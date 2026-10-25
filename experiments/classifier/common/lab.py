"""Shared harness for the stage-1 research loop (phase 7, `01-winning-classifier`). Copied from archived `01-fair-verdict`.

Up: ../README.md (protocol + experiment index) · Log: ../learning-log.md

What it holds:
- dataset loaders (4 PRD datasets, offline files in ../data, Breast Cancer via sklearn),
- per-fold preprocessing (StandardScaler -> PCA(k) -> StandardScaler, fit on train only),
- fast exact kernels (closed forms + batched numpy statevector sims, checked against Qiskit
  in ../e00-kernel-checks),
- the equal-budget tuning protocol (5-fold stratified CV inside the training split,
  30 configs per tuned model, balanced accuracy as the selection score),
- recipes `pca` (shared StandardScaler -> PCA(k) -> StandardScaler) and `native` (raw columns, model-internal scaling).
- model specs for every baseline and every candidate.

Not production code. Phase 9 re-implements the chosen pieces inside the `qlassifier` package.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

DATA = Path(__file__).resolve().parent.parent / "data"
DATASETS = ["breast_cancer", "ionosphere", "sonar", "heart"]
# Column geometry of structure-fit datasets (recipe `native`): periods per column, (lat, lon) sphere pairs.
GEOMETRY = {"deep_quakes": {"sphere_pairs": [(0, 1)]},
            "protein_strands": {"periods": {j: 360.0 for j in range(6)}},
            "protein_div6": {"periods": {j: 360.0 for j in range(6)}},
            "protein_div10": {"periods": {j: 360.0 for j in range(10)}}}
C_GRID = [0.1, 1.0, 10.0, 100.0]
BW_MULT = [0.25, 0.5, 1.0, 2.0, 4.0]  # bandwidth multipliers (scale knobs); gamma-type knobs use squares
TEST_SIZE = 0.3
N_FOLDS = 5


# ----------------------------------------------------------------------------- data
def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (float64, constant columns dropped) and y in {0, 1} (1 = minority / positive class)."""
    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data.astype(float), (1 - d.target).astype(int)  # 1 = malignant (212)
    elif name == "ionosphere":
        df = pd.read_csv(DATA / "ionosphere.data", header=None)
        X, y = df.iloc[:, :-1].to_numpy(float), (df.iloc[:, -1] == "b").astype(int).to_numpy()  # 1 = bad (126)
    elif name == "sonar":
        df = pd.read_csv(DATA / "sonar.all-data", header=None)
        X, y = df.iloc[:, :-1].to_numpy(float), (df.iloc[:, -1] == "R").astype(int).to_numpy()  # 1 = rock (97)
    elif name == "heart":
        df = pd.read_csv(DATA / "processed.cleveland.data", header=None, na_values="?").dropna()
        X, y = df.iloc[:, :13].to_numpy(float), (df.iloc[:, 13] > 0).astype(int).to_numpy()  # 1 = disease (137 of 297)
    elif name == "deep_quakes":  # USGS ComCat M>=6 2000-2024 (U.S. public domain); 1 = depth >= 70 km (795 of 3,729)
        df = pd.read_csv(DATA / "deep_quakes.csv")
        X, y = df[["latitude", "longitude"]].to_numpy(float), df["deep"].to_numpy(int)
    elif name == "protein_strands":  # RCSB PDB (CC0); 6 backbone dihedrals (deg) of residues i-1, i, i+1; 1 = strand (E)
        df = pd.read_csv(DATA / "protein_strands.csv")
        X = df[["phi_prev", "psi_prev", "phi", "psi", "phi_next", "psi_next"]].to_numpy(float)
        y = (df["ss"] == "E").astype(int).to_numpy()
    elif name in ("protein_div6", "protein_div10"):  # RCSB PDB (CC0); 4,000 residues from 1,773 chains, <= 3 per chain
        df = pd.read_csv(DATA / "protein_div.csv")
        tags = ["m1", "0", "p1"] if name == "protein_div6" else ["m2", "m1", "0", "p1", "p2"]
        X = df[[f"{a}_{t}" for t in tags for a in ("phi", "psi")]].to_numpy(float)
        y = (df["ss"] == "E").astype(int).to_numpy()
    else:
        raise KeyError(name)
    keep = X.std(axis=0) > 0
    return X[:, keep], y


def load_groups(name: str) -> np.ndarray:
    """Group ids for grouped splits (rows of one group never straddle a split). protein_strands: PDB chain id."""
    if name == "protein_strands":
        return pd.read_csv(DATA / "protein_strands.csv")["pdb"].to_numpy()
    if name in ("protein_div6", "protein_div10"):
        return pd.read_csv(DATA / "protein_div.csv")["pdb"].to_numpy()
    raise KeyError(f"no groups defined for {name}")


class Prep:
    """Shared preprocessing, fit on training rows only.
    recipe "pca": StandardScaler -> PCA(k) (only if k < d) -> StandardScaler (k=None: all features, standardized).
    recipe "native": identity (raw columns); every model then scales internally (see `_prepare`)."""

    def __init__(self, k: int | None, recipe: str = "pca"):
        self.k = k
        self.recipe = recipe

    def fit(self, X: np.ndarray) -> "Prep":
        if self.recipe == "native":
            return self
        self.s1 = StandardScaler().fit(X)
        Z = self.s1.transform(X)
        self.pca = PCA(self.k, random_state=0).fit(Z) if (self.k and self.k < Z.shape[1]) else None
        if self.pca is not None:
            Z = self.pca.transform(Z)
        self.s2 = StandardScaler().fit(Z)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.recipe == "native":
            return np.asarray(X, float)
        Z = self.s1.transform(X)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return self.s2.transform(Z)


# ----------------------------------------------------------------------------- quantum kernels (exact)
def bloch_vectors(X: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    """Product map, 2 features per qubit: |0> -RX(s*x_a)-RY(s*x_b)- on every qubit (the shipped circuit).

    Returns unit Bloch vectors, shape (N, n_qubits, 3). Odd d: last qubit gets x_b = 0. `scale` may be a scalar or
    a per-feature array (ARD). r = (cos a sin b, -sin a, cos a cos b), a = s*x_a, b = s*x_b.
    Same kernel as the |+> -RZ-RY- form (axes relabelled); both features act at first order at x = 0.
    """
    N, d = X.shape
    n = (d + 1) // 2
    Z = np.zeros((N, 2 * n))
    Z[:, :d] = X * scale
    a, b = Z[:, 0::2], Z[:, 1::2]
    ca = np.cos(a)
    return np.stack([ca * np.sin(b), -np.sin(a), ca * np.cos(b)], axis=-1)


def gamma0(F: np.ndarray) -> float:
    """Data-derived Gaussian scale: 1 / total variance of the (training) features, sum over columns.
    Invariant to sign flips / permutations of components and to a uniform shrink of all Bloch vectors."""
    return 1.0 / F.reshape(len(F), -1).var(axis=0).sum()


def product_fidelity_from_bloch(rA: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """K_ij = prod_q (1 + r_iq . r_jq) / 2 (exact fidelity for pure product states; PSD for any r)."""
    K = np.ones((rA.shape[0], rB.shape[0]))
    for q in range(rA.shape[1]):
        K *= 0.5 * (1.0 + rA[:, q, :] @ rB[:, q, :].T)
    return K


def gaussian_from_features(FA: np.ndarray, FB: np.ndarray, gamma: float) -> np.ndarray:
    a2 = (FA**2).sum(1)[:, None]
    b2 = (FB**2).sum(1)[None, :]
    D = np.maximum(a2 + b2 - 2 * FA @ FB.T, 0.0)
    return np.exp(-gamma * D)


def _bits(n: int) -> np.ndarray:
    return (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1  # little-endian, (dim, n)


def wht(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply H on every qubit to a batch of states (N, 2^n)."""
    N = psi.shape[0]
    out = psi.copy()
    for q in range(n):
        out = out.reshape(N, 2 ** (n - q - 1), 2, 2**q)
        a, b = out[:, :, 0, :], out[:, :, 1, :]
        out = np.stack([(a + b), (a - b)], axis=2) / np.sqrt(2.0)
    return out.reshape(N, 2**n)


def zz_states(X: np.ndarray, reps: int = 2, entanglement: str = "full") -> np.ndarray:
    """Statevectors of qiskit `zz_feature_map(n, reps, entanglement)` (alpha = 2, pi-shifted pairs).

    Per rep: H on all qubits, then the diagonal D(x): phase(b) = 2 sum_i x_i b_i + 2 sum_{(i,j)} (pi-x_i)(pi-x_j) (b_i xor b_j).
    Checked against qiskit Statevector in e00.
    """
    N, n = X.shape
    bits = _bits(n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if entanglement == "full" else [(i, i + 1) for i in range(n - 1)]
    phase = 2.0 * X @ bits.T
    for i, j in pairs:
        phase += 2.0 * ((np.pi - X[:, i]) * (np.pi - X[:, j]))[:, None] * (bits[:, i] ^ bits[:, j])[None, :]
    D = np.exp(1j * phase)
    psi = np.full((N, 2**n), 2 ** (-n / 2), dtype=complex) * D
    for _ in range(reps - 1):
        psi = wht(psi, n) * D
    return psi


def fidelity_from_states(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    return np.abs(SA @ SB.conj().T) ** 2


def _apply_1q(psi: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    """U: (N, 2, 2) per-sample or (2, 2) shared."""
    N = psi.shape[0]
    s = psi.reshape(N, 2 ** (n - q - 1), 2, 2**q)
    if U.ndim == 2:
        out = np.einsum("ab,mibj->miaj", U, s)
    else:
        out = np.einsum("mab,mibj->miaj", U, s)
    return out.reshape(N, 2**n)


def _rz(theta: np.ndarray) -> np.ndarray:
    U = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    U[:, 0, 0] = np.exp(-0.5j * theta)
    U[:, 1, 1] = np.exp(0.5j * theta)
    return U


def _ry(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    U = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    U[:, 0, 0], U[:, 0, 1], U[:, 1, 0], U[:, 1, 1] = c, -s, s, c
    return U


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def local_paulis(psi: np.ndarray, n: int) -> np.ndarray:
    """Return (N, n, 3) expectation values <X_q>, <Y_q>, <Z_q>."""
    N = psi.shape[0]
    out = np.zeros((N, n, 3))
    for q in range(n):
        s = psi.reshape(N, 2 ** (n - q - 1), 2, 2**q)
        a, b = s[:, :, 0, :], s[:, :, 1, :]
        ab = (a.conj() * b).sum(axis=(1, 2))
        out[:, q, 0] = 2 * ab.real
        out[:, q, 1] = 2 * ab.imag
        out[:, q, 2] = (np.abs(a) ** 2).sum(axis=(1, 2)) - (np.abs(b) ** 2).sum(axis=(1, 2))
    return out


def _rx(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    U = np.zeros((theta.shape[0], 2, 2), dtype=complex)
    U[:, 0, 0], U[:, 0, 1], U[:, 1, 0], U[:, 1, 1] = c, -1j * s, -1j * s, c
    return U


def native_states(X: np.ndarray, scale: float, layers: int) -> tuple[np.ndarray, int]:
    """Native encoding block E(x) = [RX(s x_a), RY(s x_b)] on every qubit, from |0>^n.

    layers = 0: E(x) only (product state, same kernel family as `bloch_vectors`).
    layers = 1: E(x), then CZ chain (n-1 CZ).
    layers = 2: E(x), CZ chain, E(x) again (data re-upload). Still n-1 two-qubit gates.
    Starting at |0>^n keeps the CZ chain weak near the origin (CZ|00> = |00>), so local readout stays informative.
    """
    N, d = X.shape
    n = (d + 1) // 2
    Z = np.zeros((N, 2 * n))
    Z[:, :d] = X * scale
    psi = np.zeros((N, 2**n), dtype=complex)
    psi[:, 0] = 1.0

    def enc(p):
        for q in range(n):
            p = _apply_1q(p, n, q, _rx(Z[:, 2 * q]))
            p = _apply_1q(p, n, q, _ry(Z[:, 2 * q + 1]))
        return p

    psi = enc(psi)
    if layers >= 1:
        bits = _bits(n)
        cz = np.ones(2**n)
        for q in range(n - 1):
            cz *= 1 - 2 * (bits[:, q] & bits[:, q + 1])
        psi = psi * cz[None, :]
        if layers >= 2:
            psi = enc(psi)
    return psi, n


def heisenberg_pqk_features(X: np.ndarray, t: float, trotter: int = 3, seed: int = 1234) -> np.ndarray:
    """Huang et al. (arXiv:2011.01938) style PQK embedding as in qml-benchmarks (arXiv:2403.07059):
    d+1 qubits in fixed Haar-random single-qubit states, then `trotter` steps of
    prod_j exp(-i (t/T) x_j (XX+YY+ZZ)_{j,j+1}). Returns (N, 3(d+1)) local Pauli expectations.
    exp(-i th (XX+YY+ZZ)) = e^{i th} (cos 2th I - i sin 2th SWAP).
    """
    N, d = X.shape
    n = d + 1
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    psi0 = np.array([1.0 + 0j])
    for q in range(n):  # little-endian: qubit q is bit q -> kron(v_q, previous)
        psi0 = np.kron(v[q], psi0)
    psi = np.tile(psi0, (N, 1))
    idx = np.arange(2**n)
    for _ in range(trotter):
        for j in range(d):
            bj, bk = (idx >> j) & 1, (idx >> (j + 1)) & 1
            perm = idx ^ ((bj ^ bk) << j) ^ ((bj ^ bk) << (j + 1))
            th = (t / trotter) * X[:, j]
            psi = np.cos(2 * th)[:, None] * psi - 1j * np.sin(2 * th)[:, None] * psi[:, perm]
    return local_paulis(psi, n).reshape(N, -1)


# ----------------------------------------------------------------------------- model specs
@dataclass
class Spec:
    """A model with a tuning grid. `kernel(Xtr, Xte, bw, ctx)` returns (Ktr, Kte) for kernel models;
    `make(cfg)` returns an sklearn estimator for feature models."""

    name: str
    family: str  # classical | quantum | trivial
    grid: list[dict]
    kernel: Callable | None = None
    make: Callable | None = None
    resources: Callable | None = None  # (n_features) -> dict
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _svc(C: float, class_weight: str | None = None) -> SVC:
    return SVC(C=C, kernel="precomputed", class_weight=class_weight)


M_GRID = [1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0]  # bandwidth multipliers around the data-derived anchor
CT_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]  # C in "linear-limit units": SVM C = CT / m^2 (ridge-following)
BUDGET = len(M_GRID) * len(CT_GRID)  # 30 configs for every tuned model


def kgrid(base: float, square: bool = False, c_grid: list[float] | None = None, bw_mult: list[float] | None = None) -> list[dict]:
    """Kernel-model grid. Default: ridge-following, C = CT / m^2, because for small bandwidth every RBF-like kernel
    K ~ 1 - b^2 D(x, x') and the SVM then depends only on C * b^2 (it tends to a linear SVM).
    `square=True` for gamma-type knobs (gamma = base * m^2). Rectangular grid if c_grid is given (pilot e01)."""
    if c_grid is not None:
        return [{"bw": base * (m**2 if square else m), "C": C} for m in (bw_mult or BW_MULT) for C in c_grid]
    return [{"bw": base * (m**2 if square else m), "C": ct / m**2, "m": m, "ct": ct} for m in M_GRID for ct in CT_GRID]


def specs(which: list[str] | None = None, d: int = 8, c_grid: list[float] | None = None, bw_mult: list[float] | None = None,
          pqk_t: float = 0.1) -> list[Spec]:
    """All model specs, 30 configs per tuned model (pinned ZZ: 4, PRD FR-2). Inputs are standardized (Var = 1),
    so data-derived anchors reduce to functions of d."""
    S: list[Spec] = []

    def g(base, square=False):
        return kgrid(base, square, c_grid, bw_mult)

    S.append(Spec("trivial", "trivial", [{}], make=lambda c: DummyClassifier(strategy="most_frequent")))
    S.append(Spec("logreg", "classical", [{"C": c} for c in np.logspace(-4, 4, BUDGET)],
                  make=lambda c: LogisticRegression(C=c["C"], max_iter=5000)))
    S.append(Spec("rbf_svm", "classical", g(1.0 / d, square=True),
                  make=lambda c: SVC(C=c["C"], gamma=c["bw"], kernel="rbf")))
    S.append(Spec("random_forest", "classical",
                  [{"max_features": mf, "min_samples_leaf": msl} for mf in ["sqrt", 0.1, 0.2, 0.4, 0.7, 1.0] for msl in [1, 2, 4, 8, 16]],
                  make=lambda c: RandomForestClassifier(n_estimators=200, max_features=c["max_features"],
                                                        min_samples_leaf=c["min_samples_leaf"], random_state=0, n_jobs=1)))
    S.append(Spec("grad_boost", "classical",
                  [{"learning_rate": lr, "max_leaf_nodes": ml} for lr in [0.01, 0.03, 0.1, 0.3, 1.0] for ml in [2, 3, 7, 15, 31, 63]],
                  make=lambda c: HistGradientBoostingClassifier(learning_rate=c["learning_rate"], max_leaf_nodes=c["max_leaf_nodes"],
                                                                max_iter=200, random_state=0)))
    S.append(Spec("mlp", "classical",
                  [{"hidden": h, "alpha": a} for h in [(16,), (64,), (256,), (16, 16), (64, 64)] for a in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]],
                  make=lambda c: MLPClassifier(hidden_layer_sizes=c["hidden"], alpha=c["alpha"], solver="lbfgs",
                                               max_iter=2000, random_state=0)))

    # --- standard QML baseline (pinned, PRD FR-2): ZZ full reps 2, MinMax [0, pi] fit on train, C in {0.1,1,10,100}
    def k_zz_pinned(Xtr, Xte, bw, ctx):
        mm = MinMaxScaler((0, np.pi)).fit(Xtr)
        A, B = mm.transform(Xtr), np.clip(mm.transform(Xte), 0, np.pi)
        SA, SB = zz_states(A, 2, "full"), zz_states(B, 2, "full")
        return fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)

    S.append(Spec("zz_qsvc", "quantum", [{"bw": None, "C": C} for C in [0.1, 1.0, 10.0, 100.0]], kernel=k_zz_pinned))

    # --- bandwidth-tuned fidelity kernel (arXiv:2111.05451, 2206.06686): same ZZ map, x -> c x on z-scores
    for ent in ["full", "linear"]:
        E = d * (d - 1) / 2 if ent == "full" else d - 1
        c0 = 1.0 / np.sqrt(d + 2 * np.pi**2 * E)  # 1-layer twin anchor (tuned-quantum-kernels 05 §3)

        def k_fqk(Xtr, Xte, bw, ctx, ent=ent):
            SA, SB = zz_states(bw * Xtr, 2, ent), zz_states(bw * Xte, 2, ent)
            return fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)

        S.append(Spec(f"fqk_bw_{ent}", "quantum", g(c0), kernel=k_fqk))

    # --- separable kernel (Bowles et al. arXiv:2403.07059, qml-benchmarks SeparableKernelClassifier, L=1):
    # per feature qubit RX(pi/4) then RY(s*x), x MinMax to [-pi/2, pi/2]. Per-feature kernel 3/4 + cos(s dx)/4.
    def k_sep(Xtr, Xte, bw, ctx):
        mm = MinMaxScaler((-np.pi / 2, np.pi / 2)).fit(Xtr)
        A, B = bw * mm.transform(Xtr), bw * mm.transform(Xte)

        def K(P, Q):
            out = np.ones((P.shape[0], Q.shape[0]))
            for j in range(P.shape[1]):
                out *= 0.75 + 0.25 * np.cos(P[:, j][:, None] - Q[:, j][None, :])
            return out

        return K(A, A), K(B, A)

    S.append(Spec("separable", "quantum", g(1.0), kernel=k_sep))

    # --- projected quantum kernel (Huang et al. arXiv:2011.01938; qml-benchmarks implementation), d+1 qubits
    def k_pqk(Xtr, Xte, bw, ctx):
        t = ctx.get("pqk_t", pqk_t)
        key = ("pqk", t)
        if key not in ctx:
            ctx[key] = (heisenberg_pqk_features(Xtr, t), heisenberg_pqk_features(Xte, t))
        FA, FB = ctx[key]
        g0 = 1.0 / (FA.shape[1] * FA.var())
        return gaussian_from_features(FA, FA, bw * g0), gaussian_from_features(FB, FA, bw * g0)

    S.append(Spec("pqk", "quantum", g(1.0, square=True), kernel=k_pqk))

    # --- candidates (ours)
    c0 = 2.0 / np.sqrt(d)  # RBF bridge: gamma_scale = 1/(d Var) <=> c0 = 2/sqrt(d Var), Var = 1

    def k_bloch(Xtr, Xte, bw, ctx):
        rA, rB = bloch_vectors(Xtr, bw), bloch_vectors(Xte, bw)
        return product_fidelity_from_bloch(rA, rA), product_fidelity_from_bloch(rB, rA)

    S.append(Spec("bloch_fid", "quantum", g(c0), kernel=k_bloch, notes="H1 product map, exact fidelity from Bloch vectors"))

    def k_bloch_gauss(Xtr, Xte, bw, ctx):
        rA, rB = bloch_vectors(Xtr, c0).reshape(len(Xtr), -1), bloch_vectors(Xte, c0).reshape(len(Xte), -1)
        g0 = gamma0(rA)
        return gaussian_from_features(rA, rA, bw * g0), gaussian_from_features(rB, rA, bw * g0)

    S.append(Spec("bloch_gauss", "quantum", g(1.0, square=True), kernel=k_bloch_gauss,
                  notes="H1 variant: Gaussian on Bloch features, c = c0, gamma tuned"))

    def k_bloch_gauss_c(Xtr, Xte, bw, ctx):
        rA, rB = bloch_vectors(Xtr, bw).reshape(len(Xtr), -1), bloch_vectors(Xte, bw).reshape(len(Xte), -1)
        g0 = gamma0(rA)
        return gaussian_from_features(rA, rA, g0), gaussian_from_features(rB, rA, g0)

    S.append(Spec("bloch_gauss_c", "quantum", g(c0), kernel=k_bloch_gauss_c,
                  notes="H1 variant: Gaussian on Bloch features, gamma = data-derived, c tuned"))

    for L in [1, 2]:
        for readout in ["fid", "gauss"]:
            def k_native(Xtr, Xte, bw, ctx, L=L, readout=readout):
                pa, n = native_states(Xtr, bw, L)
                pb, _ = native_states(Xte, bw, L)
                rA, rB = local_paulis(pa, n), local_paulis(pb, n)
                if readout == "fid":
                    return product_fidelity_from_bloch(rA, rA), product_fidelity_from_bloch(rB, rA)
                FA, FB = rA.reshape(len(Xtr), -1), rB.reshape(len(Xte), -1)
                g0 = 1.0 / (FA.shape[1] * FA.var())
                return gaussian_from_features(FA, FA, g0), gaussian_from_features(FB, FA, g0)

            S.append(Spec(f"native_L{L}_{readout}", "quantum", g(c0), kernel=k_native,
                          notes=f"H2/H3 entangled: encode, CZ chain{', re-encode' if L == 2 else ''}; local readout {readout}"))

    if which is not None:
        S = [s for s in S if s.name in which]
    return S


# ----------------------------------------------------------------------------- protocol
def _metrics(y, pred, score) -> dict:
    out = {"acc": accuracy_score(y, pred), "ba": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}
    try:
        out["auc"] = roc_auc_score(y, score)
    except ValueError:
        out["auc"] = np.nan
    return out


def _prepare(spec: Spec, A: np.ndarray, B: np.ndarray, recipe: str) -> tuple[np.ndarray, np.ndarray]:
    """Model-internal scaling for the `native` recipe: StandardScaler fit on the training rows A, unless the spec reads
    raw columns (spec.extra["raw"], e.g. structure-aware encodings). Recipe `pca`: inputs are already standardized."""
    if recipe != "native" or spec.family == "trivial" or spec.extra.get("raw"):
        return A, B
    sc = StandardScaler().fit(A)
    return sc.transform(A), sc.transform(B)


def _grid_scores(spec: Spec, Xtr, ytr, Xva, yva, ctx, recipe: str = "pca") -> np.ndarray:
    """Balanced accuracy for every config of `spec` on one (train, validation) pair."""
    Xtr, Xva = _prepare(spec, Xtr, Xva, recipe)
    scores = np.zeros(len(spec.grid))
    cw = spec.extra.get("class_weight")
    if spec.kernel is not None:
        cache: dict = {}
        for i, cfg in enumerate(spec.grid):
            bw = cfg["bw"]
            if bw not in cache:
                cache[bw] = spec.kernel(Xtr, Xva, bw, ctx)
            Ktr, Kva = cache[bw]
            m = _svc(cfg["C"], cw).fit(Ktr, ytr)
            scores[i] = balanced_accuracy_score(yva, m.predict(Kva))
    else:
        for i, cfg in enumerate(spec.grid):
            m = spec.make(cfg).fit(Xtr, ytr)
            scores[i] = balanced_accuracy_score(yva, m.predict(Xva))
    return scores


def run_seed(dataset: str, seed: int, k: int | None, model_names: list[str] | None = None, ctx_extra: dict | None = None,
             c_grid: list[float] | None = None, bw_mult: list[float] | None = None, recipe: str = "pca",
             extra_specs: Callable | None = None, data: tuple[np.ndarray, np.ndarray] | None = None,
             test_size: float = TEST_SIZE, groups: np.ndarray | None = None) -> list[dict]:
    """One outer split (stratified 70/30, random_state=seed). Inner 5-fold CV tunes every model with its grid.
    Preprocessing is re-fit inside every inner fold and on the outer training split.
    `extra_specs(d) -> list[Spec]` adds candidate models (d = feature count after shared preprocessing).
    `data=(X, y)` overrides the bundled loader (new datasets, subsets).
    `groups` (e.g. protein chain ids): outer GroupShuffleSplit(test_size) and inner StratifiedGroupKFold(5), so no group
    appears on both sides of any split (leakage check, iteration 4)."""
    X, y = data if data is not None else load(dataset)
    if groups is None:
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y))
    else:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    Xtr_raw, Xte_raw, ytr, yte = X[tr], X[te], y[tr], y[te]
    if recipe == "native":
        d_eff = X.shape[1]
    else:
        d_eff = min(k, X.shape[1]) if k else X.shape[1]
    S = specs(model_names, d_eff, c_grid, bw_mult, (ctx_extra or {}).get("pqk_t", 0.1)) if model_names != [] else []
    if extra_specs is not None:
        S = S + list(extra_specs(d_eff))
    if groups is None:
        inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xtr_raw, ytr)
    else:
        inner = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xtr_raw, ytr, groups[tr])
    folds = []
    for a, b in inner:
        p = Prep(k, recipe).fit(Xtr_raw[a])
        folds.append((p.transform(Xtr_raw[a]), ytr[a], p.transform(Xtr_raw[b]), ytr[b]))
    p = Prep(k, recipe).fit(Xtr_raw)
    Xtr0, Xte0 = p.transform(Xtr_raw), p.transform(Xte_raw)
    rows = []
    for spec in S:
        t0 = time.perf_counter()
        if len(spec.grid) > 1:
            cv = np.zeros(len(spec.grid))
            for (A, ya, B, yb) in folds:
                ctx = dict(ctx_extra or {})
                cv += _grid_scores(spec, A, ya, B, yb, ctx, recipe)
            cv /= len(folds)
            best = int(np.argmax(cv))
        else:
            cv, best = np.array([np.nan]), 0
        t_tune = time.perf_counter() - t0
        cfg = spec.grid[best]
        ctx = dict(ctx_extra or {})
        Xtr, Xte = _prepare(spec, Xtr0, Xte0, recipe)
        t1 = time.perf_counter()
        if spec.kernel is not None:
            Ktr, Kte = spec.kernel(Xtr, Xte, cfg["bw"], ctx)
            m = _svc(cfg["C"], spec.extra.get("class_weight")).fit(Ktr, ytr)
            t2 = time.perf_counter()
            pred, score = m.predict(Kte), m.decision_function(Kte)
            n_sv = int(m.n_support_.sum())
        else:
            m = spec.make(cfg).fit(Xtr, ytr)
            t2 = time.perf_counter()
            pred = m.predict(Xte)
            score = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
            n_sv = -1
        t3 = time.perf_counter()
        row = {"dataset": dataset, "seed": seed, "k": k or 0, "recipe": recipe, "grouped": groups is not None,
               "d": Xtr.shape[1], "model": spec.name,
               "family": spec.family,
               "cfg": repr({k_: (float(v_) if isinstance(v_, (float, np.floating)) else v_) for k_, v_ in cfg.items()}),
               "cfg_idx": best, "cv_ba": float(cv[best]) if len(spec.grid) > 1 else np.nan,
               "edge": bool(len(spec.grid) > 1 and _on_edge(spec.grid, best)),
               "t_tune": t_tune, "t_fit": t2 - t1, "t_pred": t3 - t2, "n_train": len(ytr), "n_test": len(yte), "n_sv": n_sv}
        row.update(_metrics(yte, pred, score))
        rows.append(row)
    return rows


def _on_edge(grid: list[dict], i: int) -> bool:
    """True if the chosen config sits on the edge of a numeric axis of the grid (grid may need widening)."""
    cfg = grid[i]
    skip = {"bw", "C"} if "m" in cfg else set()
    for key, val in cfg.items():
        if key in skip:
            continue
        if not isinstance(val, (int, float)) or val is None or isinstance(val, bool):
            continue
        vals = sorted({g[key] for g in grid if isinstance(g.get(key), (int, float))})
        if len(vals) > 2 and val in (vals[0], vals[-1]):
            return True
    return False


# ----------------------------------------------------------------------------- statistics
def corrected_ci(diff: np.ndarray, n_train: int, n_test: int, level: float = 0.95) -> tuple[float, float, float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test over J repeated random splits.
    Returns (mean, lo, hi, p_two_sided)."""
    from scipy import stats

    diff = np.asarray(diff, float)
    J = len(diff)
    m = diff.mean()
    v = diff.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * v)
    if se == 0:
        return m, m, m, 0.0 if m != 0 else 1.0
    t = m / se
    q = stats.t.ppf(0.5 + level / 2, J - 1)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return m, m - q * se, m + q * se, p
