"""Model registry: every model on the leaderboard, its tuning grid and its exact kernel (TDD §7.4).

Exact paths are closed forms (Qlassifier, separable, twin) and batched numpy statevectors (ZZ, FQK, PQK), ported
from the phase-7 harness (``experiments/common/lab.py``, ``cands.py``) and checked against Qiskit in the unit
tests. Grids are lab-style dicts (``bw`` = bandwidth knob, ``C``; ridge grids also carry ``m`` and ``ct``) so the
engine reproduces phase-7 rows exactly; :func:`describe` turns them into readable configs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from qlassifier.encoding import Encoding, Geometry, twin_features
from qlassifier.tuning import CT_GRID, M_GRID, gaussian_gram

__all__ = [
    "MODEL_IDS",
    "MODEL_NAMES",
    "ModelSpec",
    "describe",
    "fidelity_from_states",
    "heisenberg_pqk_features",
    "local_paulis",
    "model_specs",
    "on_edge",
    "separable_kernel",
    "zz_states",
]

FloatArray = NDArray[np.float64]
Family = Literal["quantum", "classical", "trivial", "control"]
Role = Literal["ours", "standard_qml", "published_quantum", "classical", "trivial", "twin"]
Kind = Literal["trivial", "estimator", "kernel", "qlassifier"]
KernelFn = Callable[[FloatArray, FloatArray, Any, dict[Any, Any]], tuple[FloatArray, FloatArray]]

MODEL_IDS: tuple[str, ...] = (
    "trivial",
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "gradient_boosting",
    "mlp",
    "zz_qsvc",
    "fqk_bandwidth_full",
    "fqk_bandwidth_linear",
    "pqk",
    "separable",
    "qlassifier",
    "classical_twin",
)
MODEL_NAMES: dict[str, str] = {
    "trivial": "Majority class",
    "logistic_regression": "Logistic regression",
    "rbf_svm": "RBF-SVM",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "mlp": "MLP",
    "zz_qsvc": "ZZ QSVC",
    "fqk_bandwidth_full": "FQK-bw full",
    "fqk_bandwidth_linear": "FQK-bw linear",
    "pqk": "PQK",
    "separable": "Separable",
    "qlassifier": "Qlassifier",
    "classical_twin": "Classical twin",
}
CLASSICAL: tuple[str, ...] = ("logistic_regression", "rbf_svm", "random_forest", "gradient_boosting", "mlp")
PQK_T = 0.1
PQK_TROTTER = 3
PQK_SEED = 1234


@dataclass(frozen=True)
class ModelSpec:
    """A leaderboard model: identity, tuning grid and how to build it.

    ``kind``: ``"kernel"`` models return Gram matrices per bandwidth (C swept on cached Grams); ``"estimator"``
    models are scikit-learn estimators built per config; ``"qlassifier"`` uses the shipped estimator and its own
    tuning routine. ``raw``: reads raw columns under the ``native`` recipe (no StandardScaler before the model).
    """

    id: str
    family: Family
    role: Role
    kind: Kind
    grid: tuple[dict[str, Any], ...]
    kernel: KernelFn | None = None
    make: Callable[[dict[str, Any]], Any] | None = None
    raw: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Display name."""
        return MODEL_NAMES[self.id]


# ----------------------------------------------------------------------------- grids
def ridge_grid(base: float, square: bool) -> tuple[dict[str, Any], ...]:
    """30-config ridge-following grid: bw = base * m (or base * m^2), C = C~ / m^2 (protocol §2.4)."""
    return tuple(
        {"bw": base * (m**2 if square else m), "C": ct / m**2, "m": m, "ct": ct} for m in M_GRID for ct in CT_GRID
    )


def on_edge(grid: tuple[dict[str, Any], ...], i: int) -> bool:
    """True if the chosen config sits on an extreme of a numeric grid axis (m or C~ for ridge grids)."""
    cfg = grid[i]
    skip = {"bw", "C"} if "m" in cfg else set()
    for key, val in cfg.items():
        if key in skip or isinstance(val, bool) or not isinstance(val, int | float):
            continue
        vals = sorted({g[key] for g in grid if isinstance(g.get(key), int | float)})
        if len(vals) > 2 and val in (vals[0], vals[-1]):
            return True
    return False


def describe(model: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Readable hyperparameters of a chosen config (stored in ``RunRecord.config``)."""
    if model in ("rbf_svm", "pqk", "qlassifier", "classical_twin"):
        return {"gamma": float(cfg["bw"]), "C": float(cfg["C"])}
    if model.startswith("fqk_bandwidth"):
        return {"scale": float(cfg["bw"]), "C": float(cfg["C"])}
    if model == "separable":
        return {"s": float(cfg["bw"]), "C": float(cfg["C"])}
    if model in ("zz_qsvc", "logistic_regression"):
        return {"C": float(cfg["C"])}
    if model == "mlp":
        return {"hidden_layer_sizes": list(cfg["hidden"]), "alpha": float(cfg["alpha"])}
    return {k: (float(v) if isinstance(v, float) else v) for k, v in cfg.items()}


# ----------------------------------------------------------------------------- exact quantum kernels
def _bits(n: int) -> NDArray[np.int64]:
    return (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1  # little-endian, (dim, n)


def wht(psi: NDArray[np.complex128], n: int) -> NDArray[np.complex128]:
    """Apply H on every qubit to a batch of states (N, 2^n)."""
    N = psi.shape[0]
    out = psi.copy()
    for q in range(n):
        out = out.reshape(N, 2 ** (n - q - 1), 2, 2**q)
        a, b = out[:, :, 0, :], out[:, :, 1, :]
        out = np.stack([(a + b), (a - b)], axis=2) / np.sqrt(2.0)
    result: NDArray[np.complex128] = out.reshape(N, 2**n)
    return result


def zz_states(X: FloatArray, reps: int = 2, entanglement: str = "full") -> NDArray[np.complex128]:
    """Statevectors of Qiskit ``zz_feature_map(n, reps, entanglement)`` (alpha = 2, pi-shifted pairs).

    Per rep: H on all qubits, then the diagonal phase 2 sum_i x_i b_i + 2 sum_(i,j) (pi - x_i)(pi - x_j)(b_i xor b_j).
    """
    N, n = X.shape
    bits = _bits(n)
    if entanglement == "full":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [(i, i + 1) for i in range(n - 1)]
    phase = 2.0 * X @ bits.T
    for i, j in pairs:
        phase += 2.0 * ((np.pi - X[:, i]) * (np.pi - X[:, j]))[:, None] * (bits[:, i] ^ bits[:, j])[None, :]
    D = np.exp(1j * phase)
    psi = np.full((N, 2**n), 2 ** (-n / 2), dtype=complex) * D
    for _ in range(reps - 1):
        psi = wht(psi, n) * D
    return psi


def fidelity_from_states(SA: NDArray[np.complex128], SB: NDArray[np.complex128]) -> FloatArray:
    """|<a|b>|^2 for every pair of rows."""
    K: FloatArray = np.abs(SA @ SB.conj().T) ** 2
    return K


def local_paulis(psi: NDArray[np.complex128], n: int) -> FloatArray:
    """(N, n, 3) expectation values <X_q>, <Y_q>, <Z_q> of a batch of states."""
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


def pqk_start(n: int, seed: int = PQK_SEED) -> NDArray[np.complex128]:
    """Fixed Haar-random single-qubit states (n, 2) of the PQK embedding."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    out: NDArray[np.complex128] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return out


def heisenberg_pqk_features(X: FloatArray, t: float = PQK_T, trotter: int = PQK_TROTTER) -> FloatArray:
    """Projected quantum kernel embedding (arXiv:2011.01938, qml-benchmarks arXiv:2403.07059).

    d + 1 qubits in fixed Haar-random product states, then ``trotter`` steps of prod_j exp(-i (t/T) x_j
    (XX + YY + ZZ)_(j, j+1)); returns (N, 3(d + 1)) local Pauli expectations.
    exp(-i th (XX+YY+ZZ)) = e^(i th) (cos 2th I - i sin 2th SWAP).
    """
    N, d = X.shape
    n = d + 1
    v = pqk_start(n)
    psi0 = np.array([1.0 + 0j])
    for q in range(n):  # little-endian: qubit q is bit q
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


def separable_kernel(P: FloatArray, Q: FloatArray) -> FloatArray:
    """Separable kernel (arXiv:2403.07059): prod_j (3/4 + cos(P_j - Q_j) / 4) on scaled angles."""
    out = np.ones((P.shape[0], Q.shape[0]))
    for j in range(P.shape[1]):
        out *= 0.75 + 0.25 * np.cos(P[:, j][:, None] - Q[:, j][None, :])
    return out


def _minmax(A: FloatArray, B: FloatArray, lo: float, hi: float, clip: bool) -> tuple[FloatArray, FloatArray]:
    from sklearn.preprocessing import MinMaxScaler

    mm = MinMaxScaler((lo, hi)).fit(A)
    Bt = mm.transform(B)
    return mm.transform(A), (np.clip(Bt, lo, hi) if clip else Bt)


# ----------------------------------------------------------------------------- registry
def model_specs(d: int, geometry: Mapping[int, float] | None = None) -> list[ModelSpec]:
    """Every model of the protocol for d input features (the twin only when the dataset declares geometry)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    specs: list[ModelSpec] = [
        ModelSpec(
            "trivial", "trivial", "trivial", "trivial", ({},), make=lambda c: DummyClassifier(strategy="most_frequent")
        ),
        ModelSpec(
            "logistic_regression", "classical", "classical", "estimator",
            tuple({"C": float(c)} for c in np.logspace(-4, 4, 30)),
            make=lambda c: LogisticRegression(C=c["C"], max_iter=5000),
        ),
        ModelSpec(
            "rbf_svm", "classical", "classical", "estimator", ridge_grid(1.0 / d, square=True),
            make=lambda c: SVC(C=c["C"], gamma=c["bw"], kernel="rbf"),
        ),
        ModelSpec(
            "random_forest", "classical", "classical", "estimator",
            tuple(
                {"max_features": mf, "min_samples_leaf": msl}
                for mf in ("sqrt", 0.1, 0.2, 0.4, 0.7, 1.0)
                for msl in (1, 2, 4, 8, 16)
            ),
            make=lambda c: RandomForestClassifier(
                n_estimators=200, max_features=c["max_features"], min_samples_leaf=c["min_samples_leaf"],
                random_state=0, n_jobs=1,
            ),
        ),
        ModelSpec(
            "gradient_boosting", "classical", "classical", "estimator",
            tuple(
                {"learning_rate": lr, "max_leaf_nodes": ml}
                for lr in (0.01, 0.03, 0.1, 0.3, 1.0)
                for ml in (2, 3, 7, 15, 31, 63)
            ),
            make=lambda c: HistGradientBoostingClassifier(
                learning_rate=c["learning_rate"], max_leaf_nodes=c["max_leaf_nodes"], max_iter=200, random_state=0
            ),
        ),
        ModelSpec(
            "mlp", "classical", "classical", "estimator",
            tuple(
                {"hidden": h, "alpha": a}
                for h in ((16,), (64,), (256,), (16, 16), (64, 64))
                for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
            ),
            make=lambda c: MLPClassifier(
                hidden_layer_sizes=c["hidden"], alpha=c["alpha"], solver="lbfgs", max_iter=2000, random_state=0
            ),
        ),
    ]  # fmt: skip

    def k_zz(A: FloatArray, B: FloatArray, bw: Any, ctx: dict[Any, Any]) -> tuple[FloatArray, FloatArray]:
        Am, Bm = _minmax(A, B, 0.0, np.pi, clip=True)
        SA, SB = zz_states(Am, 2, "full"), zz_states(Bm, 2, "full")
        return fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)

    zz_grid = tuple({"bw": None, "C": C} for C in (0.1, 1.0, 10.0, 100.0))  # pinned by PRD FR-2
    specs.append(ModelSpec("zz_qsvc", "quantum", "standard_qml", "kernel", zz_grid, kernel=k_zz))

    for ent in ("full", "linear"):
        E = d * (d - 1) / 2 if ent == "full" else d - 1
        c_zz = 1.0 / np.sqrt(d + 2 * np.pi**2 * E)  # one-layer anchor of the bandwidth (tuned-quantum-kernels)

        def k_fqk(
            A: FloatArray, B: FloatArray, bw: Any, ctx: dict[Any, Any], ent: str = ent
        ) -> tuple[FloatArray, FloatArray]:
            SA, SB = zz_states(bw * A, 2, ent), zz_states(bw * B, 2, ent)
            return fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)

        fqk_grid = ridge_grid(c_zz, False)
        specs.append(
            ModelSpec(f"fqk_bandwidth_{ent}", "quantum", "published_quantum", "kernel", fqk_grid, kernel=k_fqk)
        )

    def k_pqk(A: FloatArray, B: FloatArray, bw: Any, ctx: dict[Any, Any]) -> tuple[FloatArray, FloatArray]:
        if "pqk" not in ctx:
            ctx["pqk"] = (heisenberg_pqk_features(A), heisenberg_pqk_features(B))
        FA, FB = ctx["pqk"]
        g0 = 1.0 / (FA.shape[1] * FA.var())
        return gaussian_gram(FA, FA, bw * g0), gaussian_gram(FB, FA, bw * g0)

    def k_sep(A: FloatArray, B: FloatArray, bw: Any, ctx: dict[Any, Any]) -> tuple[FloatArray, FloatArray]:
        Am, Bm = _minmax(A, B, -np.pi / 2, np.pi / 2, clip=False)
        P, Q = bw * Am, bw * Bm
        return separable_kernel(P, P), separable_kernel(Q, P)

    specs.append(ModelSpec("pqk", "quantum", "published_quantum", "kernel", ridge_grid(1.0, True), kernel=k_pqk))
    specs.append(ModelSpec("separable", "quantum", "published_quantum", "kernel", ridge_grid(1.0, False), kernel=k_sep))
    specs.append(
        ModelSpec("qlassifier", "quantum", "ours", "qlassifier", ridge_grid(1.0, True), raw=geometry is not None,
                  extra={"geometry": None if geometry is None else dict(geometry)})
    )  # fmt: skip

    if geometry is not None:
        plan = Encoding.plan(d, cast(Geometry, dict(geometry)), None)

        def k_twin(A: FloatArray, B: FloatArray, bw: Any, ctx: dict[Any, Any]) -> tuple[FloatArray, FloatArray]:
            if "twin" not in ctx:
                enc = plan.fit(A)
                ctx["twin"] = (twin_features(A, enc), twin_features(B, enc))
            FA, FB = ctx["twin"]
            g0 = 1.0 / FA.var(axis=0).sum()
            return gaussian_gram(FA, FA, bw * g0), gaussian_gram(FB, FA, bw * g0)

        specs.append(
            ModelSpec("classical_twin", "control", "twin", "kernel", ridge_grid(1.0, True), kernel=k_twin, raw=True)
        )
    return specs
