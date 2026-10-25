"""Noise path on IBM fake backends (archived e05 code, reused). Breast Cancer, PCA-8 (product model: 4 qubits).

Up: ../README.md (protocol §2.8) · Log: ../learning-log.md
Stage-1 changes: outputs go to $E05_DIR (default: this folder); Part A models from $E05_MODELS_A (comma list).

Part A (full data, 569 samples, seeds 0-4) — product Bloch model:
  - circuits: |0> -RX(c x_a)-RY(c x_b)- per qubit, measured in X, Y, Z (3 circuits per sample) + 2 readout-calibration circuits;
  - SamplerV2 on AerSimulator.from_backend(FakeTorino | FakeFez | FakeBrisbane), 1,024 shots; "shots" = ideal
    sampling (binomial, exact for product states); "exact" = closed form;
  - hyperparameters from exact-mode CV on the training split (same protocol as e02); C optionally retuned by CV on the
    measured training kernel (zero extra circuits);
  - post-processing variants: raw, mit (readout affine correction), norm (unit Bloch vectors), mit+norm;
  - readouts from the same measurements: fidelity product kernel; Gaussian with data-derived gamma.
Part B (seeds 0-2, stratified subset 60 train / 40 test of the same split) — same points for every model:
  product model vs bandwidth-tuned fidelity kernel (ZZ linear reps 2, compute-uncompute, O(N^2) circuits)
  vs projected quantum kernel (Heisenberg, d+1 qubits, 3 circuits per sample). Each tuned in exact mode on the subset.

Run: .venv/bin/python noise.py [partA|partB] [backend ...]
Part B options (env): E05_SEEDS="0,1,2", E05_MODELS="bloch_gauss,bloch_fid,fqk_bw_linear,pqk" (or "zz_qsvc"),
E05_OUT=partB_runs/<name>.csv, E05_NTR / E05_NTE (subset sizes, default 60 / 40). Phase 7 ran one process per (seed, backend) in parallel and merged into partB.csv.
Writes partA.csv + partA_stats.json, and Part B csv + .stats.json, next to this file.
"""

from __future__ import annotations

import ast
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez, FakeTorino
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import os  # noqa: E402
OUT = Path(os.environ.get("E05_DIR", str(HERE)))
import lab  # noqa: E402

SHOTS = 1024
CAL_SHOTS = 8192
BACKENDS = {"fake_torino": FakeTorino, "fake_fez": FakeFez, "fake_brisbane": FakeBrisbane}
DATASET, K = "breast_cancer", 8


# ----------------------------------------------------------------------------- helpers
def ideal_bloch(Z: np.ndarray) -> np.ndarray:
    """|0> -RX(a)-RY(b)-: r = (cos a sin b, -sin a, cos a cos b). Z: (N, 2n) angles."""
    a, b = Z[:, 0::2], Z[:, 1::2]
    return np.stack([np.cos(a) * np.sin(b), -np.sin(a), np.cos(a) * np.cos(b)], axis=-1)


def angles(X: np.ndarray, c: float) -> np.ndarray:
    N, d = X.shape
    n = (d + 1) // 2
    Z = np.zeros((N, 2 * n))
    Z[:, :d] = c * X
    return Z


def gauss_kernel(FA, FB, mult, ref):
    """exp(-gamma ||f - f'||^2), gamma = mult / total variance of the reference (training) features."""
    g0 = 1.0 / ref.reshape(len(ref), -1).var(axis=0).sum()
    A, B = FA.reshape(len(FA), -1), FB.reshape(len(FB), -1)
    return lab.gaussian_from_features(A, B, mult * g0)


def kernel_of(model: str, RA, RB, mult, ref):
    if model == "bloch_fid":
        return lab.product_fidelity_from_bloch(RA, RB)
    return gauss_kernel(RA, RB, mult, ref)


def eval_svm(Ktr, ytr, Kte, yte, C):
    m = SVC(C=C, kernel="precomputed").fit(Ktr, ytr)
    pred, sc = m.predict(Kte), m.decision_function(Kte)
    return {"acc": accuracy_score(yte, pred), "ba": balanced_accuracy_score(yte, pred), "auc": roc_auc_score(yte, sc)}


def retune_C(Ktr, ytr, m, seed):
    """5-fold CV over the C axis of the protocol grid at fixed bandwidth (C = CT / m^2). Classical, no circuits."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best, bestC = -1, None
    for ct in lab.CT_GRID:
        C = ct / m**2
        s = []
        for a, b in skf.split(Ktr, ytr):
            mm = SVC(C=C, kernel="precomputed").fit(Ktr[np.ix_(a, a)], ytr[a])
            s.append(balanced_accuracy_score(ytr[b], mm.predict(Ktr[np.ix_(b, a)])))
        if np.mean(s) > best:
            best, bestC = np.mean(s), C
    return bestC


def kstats(Kn, Ke):
    iu = np.triu_indices_from(Ke, 1)
    a, b = Kn[iu], Ke[iu]
    return {"k_pearson": float(np.corrcoef(a, b)[0, 1]), "k_mae": float(np.abs(a - b).mean())}


def split(seed):
    X, y = lab.load(DATASET)
    tr, te = next(StratifiedShuffleSplit(1, test_size=lab.TEST_SIZE, random_state=seed).split(X, y))
    p = lab.Prep(K).fit(X[tr])
    return p.transform(X[tr]), y[tr], p.transform(X[te]), y[te], X[tr], y[tr]


def exact_cfg(model: str, seed: int, Xtr_raw=None, ytr=None):
    """Exact-mode tuning with the e02 protocol (5-fold CV, per-fold preprocessing). Returns the chosen config dict."""
    if Xtr_raw is None:
        rows = lab.run_seed(DATASET, seed, K, [model])
        return ast.literal_eval(rows[0]["cfg"])
    # subset tuning (Part B): same CV machinery on the given raw training rows
    spec = lab.specs([model], K)[0]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv = np.zeros(len(spec.grid))
    for a, b in skf.split(Xtr_raw, ytr):
        p = lab.Prep(K).fit(Xtr_raw[a])
        cv += lab._grid_scores(spec, p.transform(Xtr_raw[a]), ytr[a], p.transform(Xtr_raw[b]), ytr[b], {"pqk_t": 0.1})
    return spec.grid[int(np.argmax(cv))]


# ----------------------------------------------------------------------------- circuits
def bloch_circuits(n):
    p = ParameterVector("p", 2 * n)
    out = {}
    for basis in "XYZ":
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.rx(p[2 * q], q)
            qc.ry(p[2 * q + 1], q)
        for q in range(n):
            if basis == "X":
                qc.h(q)
            elif basis == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        out[basis] = qc
    return out


def cal_circuits(n):
    c0 = QuantumCircuit(n)
    c0.measure_all()
    c1 = QuantumCircuit(n)
    c1.x(range(n))
    c1.measure_all()
    return c0, c1


def _values(tc, P):
    """Order parameter columns as in tc.parameters (sorted by name)."""
    idx = [int(str(prm).split("[")[1][:-1]) for prm in tc.parameters]
    return P[:, idx]


def _p1(ba, n):
    """Per-qubit probability of reading 1, shape (N, n)."""
    return np.stack([ba.slice_bits([q]).bitcount().mean(axis=-1) for q in range(n)], axis=-1)


class Device:
    def __init__(self, name):
        self.name = name
        self.be = BACKENDS[name]()
        self.sampler = SamplerV2.from_backend(AerSimulator.from_backend(self.be), seed=11)
        self.stats = {}

    def bloch(self, Z: np.ndarray, tag: str):
        """Measure Bloch vectors for all rows of Z (angles). Returns raw (N,n,3), calibration (e0, e1), stats."""
        N, two_n = Z.shape
        n = two_n // 2
        circs = bloch_circuits(n)
        pm0 = generate_preset_pass_manager(optimization_level=3, backend=self.be, seed_transpiler=0)
        tz = pm0.run(circs["Z"])
        layout = tz.layout.initial_index_layout(filter_ancillas=True)[:n]
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.be, initial_layout=layout, seed_transpiler=0)
        tcs = {b: (tz if b == "Z" else pm.run(circs[b])) for b in "XYZ"}
        c0, c1 = (pm.run(c) for c in cal_circuits(n))
        for b, tc in tcs.items():
            assert list(tc.layout.final_index_layout(filter_ancillas=True)[:n]) == list(layout), "layout drift"
        pubs = [(tcs[b], _values(tcs[b], Z)) for b in "XYZ"]
        res = self.sampler.run(pubs, shots=SHOTS).result()
        cal = self.sampler.run([(c0,), (c1,)], shots=CAL_SHOTS).result()
        R = np.stack([1 - 2 * _p1(res[i].data.meas, n) for i in range(3)], axis=-1)  # (N, n, 3) in X, Y, Z
        e0 = np.asarray(_p1(cal[0].data.meas, n)).reshape(-1, n)[0]  # P(1 | prepared 0)
        e1 = 1 - np.asarray(_p1(cal[1].data.meas, n)).reshape(-1, n)[0]  # P(0 | prepared 1)
        ops = {b: dict(tc.count_ops()) for b, tc in tcs.items()}
        self.stats[tag] = {"layout": [int(v) for v in layout], "ops": ops, "depth": {b: tc.depth() for b, tc in tcs.items()},
                           "two_qubit_gates": int(max(sum(v for k, v in o.items() if k in ("cz", "ecr", "cx")) for o in ops.values())),
                           "e0": e0.round(4).tolist(), "e1": e1.round(4).tolist(), "circuits": 3 * N + 2}
        return R, e0, e1

    def fidelity_kernel(self, fm, A: np.ndarray, B: np.ndarray | None, tag: str):
        """Compute-uncompute P(0^n) for all pairs (i<j of A if B is None, else all B x A)."""
        n = fm.num_qubits
        pa, pb = ParameterVector("a", n), ParameterVector("b", n)
        cu = fm.assign_parameters(pa).compose(fm.assign_parameters(pb).inverse())
        cu.measure_all()
        tc = generate_preset_pass_manager(optimization_level=3, backend=self.be, seed_transpiler=0).run(cu)
        names = [str(p) for p in tc.parameters]

        def row(x, xp):
            m = {f"a[{i}]": x[i] for i in range(n)} | {f"b[{i}]": xp[i] for i in range(n)}
            return [m[k] for k in names]

        if B is None:
            iu = np.triu_indices(len(A), 1)
            vals = np.array([row(A[i], A[j]) for i, j in zip(*iu)])
        else:
            vals = np.array([row(b, a) for b in B for a in A])
        res = self.sampler.run([(tc, vals)], shots=SHOTS).result()[0].data.meas
        p0 = (res.bitcount() == 0).mean(axis=-1)
        if B is None:
            K = np.eye(len(A))
            K[iu] = p0
            K[(iu[1], iu[0])] = p0
            w, v = np.linalg.eigh(K)  # PSD repair: symmetric eigen-clip (correct version)
            K = (v * np.maximum(w, 0)) @ v.T
        else:
            K = p0.reshape(len(B), len(A))
        ops = dict(tc.count_ops())
        self.stats[tag] = {"ops": ops, "depth": tc.depth(), "circuits": int(len(vals)),
                           "two_qubit_gates": int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx")))}
        return K

    def pqk_features(self, X: np.ndarray, t: float, tag: str, trotter: int = 3, seed: int = 1234):
        """Heisenberg PQK (d+1 qubits) local Paulis, 3 circuits per sample."""
        N, d = X.shape
        n = d + 1
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        p = ParameterVector("x", d)
        out = []
        for basis in "XYZ":
            qc = QuantumCircuit(n)
            for q in range(n):
                th = 2 * np.arccos(np.clip(abs(v[q, 0]), 0, 1))
                qc.u(th, np.angle(v[q, 1]) - np.angle(v[q, 0]), 0, q)
            for _ in range(trotter):
                for j in range(d):
                    ang = 2 * (t / trotter) * p[j]
                    qc.rxx(ang, j, j + 1)
                    qc.ryy(ang, j, j + 1)
                    qc.rzz(ang, j, j + 1)
            for q in range(n):
                if basis == "X":
                    qc.h(q)
                elif basis == "Y":
                    qc.sdg(q)
                    qc.h(q)
            qc.measure_all()
            out.append(qc)
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.be, seed_transpiler=0)
        tcs = [pm.run(c) for c in out]
        res = self.sampler.run([(tc, _values(tc, X)) for tc in tcs], shots=SHOTS).result()
        F = np.stack([1 - 2 * _p1(res[i].data.meas, n) for i in range(3)], axis=-1)  # (N, n, 3)
        ops = dict(tcs[2].count_ops())
        self.stats[tag] = {"ops": ops, "depth": tcs[2].depth(), "circuits": 3 * N,
                           "two_qubit_gates": int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx")))}
        return F.reshape(N, -1)


def post(R, e0, e1, variant):
    out = R.copy()
    if "mit" in variant:
        out = (out - (e1 - e0)[None, :, None]) / (1 - e0 - e1)[None, :, None]
    if "norm" in variant:
        out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-3)
    return out


# ----------------------------------------------------------------------------- part A
MODELS_A = os.environ.get("E05_MODELS_A", "bloch_fid,bloch_gauss,bloch_gauss_c").split(",")
VARIANTS = ["raw", "mit", "norm", "mit+norm"]


def part_a(backends, seeds=None):
    seeds = [int(v) for v in os.environ.get("E05_SEEDS_A", "0,1,2,3,4").split(",")] if seeds is None else seeds
    rows, stats = [], {}
    devs = {b: Device(b) for b in backends}
    rng = np.random.default_rng(5)
    for seed in seeds:
        Xtr, ytr, Xte, yte, _, _ = split(seed)
        Xall = np.vstack([Xtr, Xte])
        ntr = len(Xtr)
        d = Xtr.shape[1]
        c0 = 2 / np.sqrt(d)
        for model in MODELS_A:
            cfg = exact_cfg(model, seed)
            m = cfg["m"]
            c = cfg["bw"] if model in ("bloch_fid", "bloch_gauss_c") else c0
            mult = cfg["bw"] if model == "bloch_gauss" else 1.0  # gamma multiplier (bloch_gauss tunes gamma)
            Z = angles(Xall, c)
            Re = ideal_bloch(Z)
            Ke_tr = kernel_of(model, Re[:ntr], Re[:ntr], mult, Re[:ntr])
            Ke_te = kernel_of(model, Re[ntr:], Re[:ntr], mult, Re[:ntr])
            base = {"seed": seed, "model": model, "cfg": repr(cfg), "n_qubits": Z.shape[1] // 2}
            r = eval_svm(Ke_tr, ytr, Ke_te, yte, cfg["C"])
            rows.append(base | {"mode": "exact", "variant": "-", "retune": False, **r, "k_pearson": 1.0, "k_mae": 0.0, "bloch_norm": 1.0})
            # shots only (binomial sampling of each Pauli, exact for product states)
            Rs = 1 - 2 * rng.binomial(SHOTS, (1 - Re) / 2) / SHOTS
            modes = {"shots": (Rs, np.zeros(Z.shape[1] // 2), np.zeros(Z.shape[1] // 2))}
            for b in backends:
                t0 = time.time()
                modes[b] = devs[b].bloch(Z, f"{model}-s{seed}")
                stats.setdefault(b, {})["bloch_seconds"] = round(time.time() - t0, 2)
            for mode, (R, e0, e1) in modes.items():
                for variant in (VARIANTS if mode != "shots" else ["raw", "norm"]):
                    Rp = post(R, e0, e1, variant)
                    Kn_tr = kernel_of(model, Rp[:ntr], Rp[:ntr], mult, Rp[:ntr])
                    Kn_te = kernel_of(model, Rp[ntr:], Rp[:ntr], mult, Rp[:ntr])
                    ks = kstats(Kn_tr, Ke_tr)
                    bn = float(np.linalg.norm(Rp, axis=-1).mean())
                    for retune in (False, True):
                        C = retune_C(Kn_tr, ytr, m, seed) if retune else cfg["C"]
                        r = eval_svm(Kn_tr, ytr, Kn_te, yte, C)
                        rows.append(base | {"mode": mode, "variant": variant, "retune": retune, "C_used": C, **r, **ks, "bloch_norm": bn})
            print(f"seed {seed} {model} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "partA.csv", index=False)
    for b, dv in devs.items():
        stats.setdefault(b, {})["transpile_example"] = next(iter(dv.stats.values()))
    (OUT / "partA_stats.json").write_text(json.dumps(stats, indent=2, default=str))


# ----------------------------------------------------------------------------- part B
def part_b(backends, seeds=None, n_tr=60, n_te=40, models=None):
    import os
    seeds = [int(v) for v in os.environ.get("E05_SEEDS", "0,1,2").split(",")]
    n_tr, n_te = int(os.environ.get("E05_NTR", n_tr)), int(os.environ.get("E05_NTE", n_te))
    models = os.environ.get("E05_MODELS", "bloch_gauss,bloch_fid,fqk_bw_linear,pqk").split(",")
    out_csv = OUT / os.environ.get("E05_OUT", "partB.csv")
    rows, stats = [], {}
    devs = {b: Device(b) for b in backends}
    for seed in seeds:
        X, y = lab.load(DATASET)
        tr, te = next(StratifiedShuffleSplit(1, test_size=lab.TEST_SIZE, random_state=seed).split(X, y))
        tr_s, _ = train_test_split(tr, train_size=n_tr, stratify=y[tr], random_state=seed)
        te_s, _ = train_test_split(te, train_size=n_te, stratify=y[te], random_state=seed)
        Xr, yr, Xs, ys = X[tr_s], y[tr_s], X[te_s], y[te_s]
        p = lab.Prep(K).fit(Xr)
        A, B = p.transform(Xr), p.transform(Xs)
        d = A.shape[1]
        for model in models:
            cfg = exact_cfg(model, seed, Xr, yr)
            m = cfg.get("m", 1.0)  # pinned ZZ grid has no bandwidth axis
            base = {"seed": seed, "model": model, "cfg": repr(cfg), "n_train": n_tr, "n_test": n_te}
            if model.startswith("bloch"):
                c = cfg["bw"] if model == "bloch_fid" else 2 / np.sqrt(d)
                mult = cfg["bw"] if model == "bloch_gauss" else 1.0
                Z = angles(np.vstack([A, B]), c)
                Re = ideal_bloch(Z)
                Ke_tr = kernel_of(model, Re[:n_tr], Re[:n_tr], mult, Re[:n_tr])
                Ke_te = kernel_of(model, Re[n_tr:], Re[:n_tr], mult, Re[:n_tr])
            elif model == "zz_qsvc":
                from sklearn.preprocessing import MinMaxScaler
                mm = MinMaxScaler((0, np.pi)).fit(A)
                Az, Bz = mm.transform(A), np.clip(mm.transform(B), 0, np.pi)
                fm = zz_feature_map(d, reps=2, entanglement="full")
                SA, SB = lab.zz_states(Az, 2, "full"), lab.zz_states(Bz, 2, "full")
                Ke_tr, Ke_te = lab.fidelity_from_states(SA, SA), lab.fidelity_from_states(SB, SA)
                cfg = dict(cfg, m=1.0)
            elif model == "fqk_bw_linear":
                fm = zz_feature_map(d, reps=2, entanglement="linear")
                SA, SB = lab.zz_states(cfg["bw"] * A, 2, "linear"), lab.zz_states(cfg["bw"] * B, 2, "linear")
                Ke_tr, Ke_te = lab.fidelity_from_states(SA, SA), lab.fidelity_from_states(SB, SA)
            else:  # pqk
                FA, FB = lab.heisenberg_pqk_features(A, 0.1), lab.heisenberg_pqk_features(B, 0.1)
                g0 = 1.0 / (FA.shape[1] * FA.var())
                Ke_tr = lab.gaussian_from_features(FA, FA, cfg["bw"] * g0)
                Ke_te = lab.gaussian_from_features(FB, FA, cfg["bw"] * g0)
            r = eval_svm(Ke_tr, yr, Ke_te, ys, cfg["C"])
            rows.append(base | {"mode": "exact", "retune": False, **r, "k_pearson": 1.0, "k_mae": 0.0, "circuits": 0})
            for b in backends:
                t0 = time.time()
                dv = devs[b]
                if model.startswith("bloch"):
                    R, e0, e1 = dv.bloch(Z, f"B-{model}-s{seed}")
                    Rp = post(R, e0, e1, "norm")  # shipped post-processing: unit Bloch vectors (Part A ablation)
                    Kn_tr = kernel_of(model, Rp[:n_tr], Rp[:n_tr], mult, Rp[:n_tr])
                    Kn_te = kernel_of(model, Rp[n_tr:], Rp[:n_tr], mult, Rp[:n_tr])
                    circuits = 3 * (n_tr + n_te) + 2
                elif model == "zz_qsvc":
                    Kn_tr = dv.fidelity_kernel(fm, Az, None, f"B-zz-tr-s{seed}")
                    Kn_te = dv.fidelity_kernel(fm, Az, Bz, f"B-zz-te-s{seed}")
                    circuits = n_tr * (n_tr - 1) // 2 + n_tr * n_te
                elif model == "fqk_bw_linear":
                    Kn_tr = dv.fidelity_kernel(fm, cfg["bw"] * A, None, f"B-fqk-tr-s{seed}")
                    Kn_te = dv.fidelity_kernel(fm, cfg["bw"] * A, cfg["bw"] * B, f"B-fqk-te-s{seed}")
                    circuits = n_tr * (n_tr - 1) // 2 + n_tr * n_te
                else:
                    F = dv.pqk_features(np.vstack([A, B]), 0.1, f"B-pqk-s{seed}")
                    FA_, FB_ = F[:n_tr], F[n_tr:]
                    g0 = 1.0 / (FA_.shape[1] * FA_.var())
                    Kn_tr = lab.gaussian_from_features(FA_, FA_, cfg["bw"] * g0)
                    Kn_te = lab.gaussian_from_features(FB_, FA_, cfg["bw"] * g0)
                    circuits = 3 * (n_tr + n_te)
                ks = kstats(Kn_tr, Ke_tr)
                for retune in (False, True):
                    C = retune_C(Kn_tr, yr, m, seed) if retune else cfg["C"]
                    r = eval_svm(Kn_tr, yr, Kn_te, ys, C)
                    rows.append(base | {"mode": b, "retune": retune, **r, **ks, "circuits": circuits,
                                        "seconds": round(time.time() - t0, 1)})
                print(f"B seed {seed} {model} {b} {time.time() - t0:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    for b, dv in devs.items():
        stats[b] = dv.stats
    (out_csv.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "partA"
    bks = sys.argv[2:] or list(BACKENDS)
    t0 = time.time()
    part_a(bks) if which == "partA" else part_b(bks)
    print(f"{which} done in {time.time() - t0:.0f} s")
