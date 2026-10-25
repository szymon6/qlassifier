"""e08: geometry mode under IBM device noise (structure-fit datasets), fresh seeds.

Up: README.md · Protocol: ../README.md §2.8 · Exact results + chosen configs: ../e07-structure-fit/
Circuits (0 two-qubit gates): circle column -> RY(2 pi x / P); (lat, lon) -> RY(90 deg - lat) then RZ(lon).
Readout: all qubits in X, Y, Z (3 circuits per sample). Post-processing: unit renormalisation per qubit.
Hyperparameters (m, C) are the exact-mode CV choices stored in e07 (training data only); no re-tuning.
Usage: .venv/bin/python geo_noise.py DATASET SEEDS(comma) DEVICE [DEVICE ...]   -> writes <dataset>_<device>.csv
"""
from __future__ import annotations

import ast
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez, FakeTorino
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cands  # noqa: E402
import lab  # noqa: E402

SHOTS = 1024
BACKENDS = {"fake_torino": FakeTorino, "fake_fez": FakeFez, "fake_brisbane": FakeBrisbane}


def angles_of(X: np.ndarray, geom: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per qubit: (kind, theta_y, phi_z) angles in radians for every row."""
    out = []
    for j, P in geom.get("periods", {}).items():
        out.append(("circle", 2 * np.pi * X[:, j] / P, np.zeros(len(X))))
    for la, lo in geom.get("sphere_pairs", []):
        out.append(("sphere", np.deg2rad(90.0 - X[:, la]), np.deg2rad(X[:, lo])))
    return out


def circuits(n: int):
    th, ph = ParameterVector("t", n), ParameterVector("p", n)
    out = {}
    for basis in "XYZ":
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(th[q], q)
            qc.rz(ph[q], q)
            if basis == "X":
                qc.h(q)
            elif basis == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        out[basis] = qc
    return out


def measure(be, X: np.ndarray, geom: dict) -> np.ndarray:
    qa = angles_of(X, geom)
    n = len(qa)
    T = np.stack([a[1] for a in qa], 1)
    Pz = np.stack([a[2] for a in qa], 1)
    circs = circuits(n)
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0)
    tz = pm.run(circs["Z"])
    layout = tz.layout.initial_index_layout(filter_ancillas=True)[:n]
    pm2 = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=layout, seed_transpiler=0)
    tcs = {b: (tz if b == "Z" else pm2.run(circs[b])) for b in "XYZ"}
    two_q = max(sum(v for k, v in tc.count_ops().items() if k in ("cz", "ecr", "cx")) for tc in tcs.values())
    sampler = SamplerV2.from_backend(AerSimulator.from_backend(be), seed=11)

    def vals(tc):
        names = [str(p) for p in tc.parameters]
        cols = []
        for nm in names:
            k, i = nm[0], int(nm.split("[")[1][:-1])
            cols.append(T[:, i] if k == "t" else Pz[:, i])
        return np.stack(cols, 1)

    res = sampler.run([(tcs[b], vals(tcs[b])) for b in "XYZ"], shots=SHOTS).result()
    R = np.stack([1 - 2 * np.stack([res[i].data.meas.slice_bits([q]).bitcount().mean(axis=-1) for q in range(n)], -1)
                  for i in range(3)], -1)  # (N, n, 3)
    R = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), 1e-3)  # unit renormalisation
    return R, two_q, int(max(tc.depth() for tc in tcs.values()))


def main():
    ds, seeds, devs = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3:]
    geom = lab.GEOMETRY[ds]
    X, y = lab.load(ds)
    e07 = pd.read_csv(HERE.parent / "e07-structure-fit" / ("quakes.csv" if ds == "deep_quakes" else "protein.csv"))
    rows = []
    for seed in seeds:
        tr, te = next(StratifiedShuffleSplit(1, test_size=lab.TEST_SIZE, random_state=seed).split(X, y))
        cfg = ast.literal_eval(e07[(e07.seed == seed) & (e07.model == "qlass_geo")].cfg.iloc[0])
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Ftr, Fte = cands.geo_bloch(Xtr, Xtr, **geom), cands.geo_bloch(Xte, Xtr, **geom)
        Ktr, Kte = cands._gauss(Ftr, Fte, cfg["bw"])
        m = SVC(C=cfg["C"], kernel="precomputed").fit(Ktr, ytr)
        p = m.predict(Kte)
        rows.append({"dataset": ds, "seed": seed, "mode": "exact", "acc": accuracy_score(yte, p), "ba": balanced_accuracy_score(yte, p),
                     "k_pearson": 1.0, "cfg": repr(cfg)})
        for dv in devs:
            t0 = time.time()
            be = BACKENDS[dv]()
            Rtr, q2, depth = measure(be, Xtr, geom)
            Rte, _, _ = measure(be, Xte, geom)
            Kn_tr, Kn_te = cands._gauss(Rtr, Rte, cfg["bw"])
            mn = SVC(C=cfg["C"], kernel="precomputed").fit(Kn_tr, ytr)
            pn = mn.predict(Kn_te)
            iu = np.triu_indices(len(Ktr), 1)
            rows.append({"dataset": ds, "seed": seed, "mode": dv, "acc": accuracy_score(yte, pn), "ba": balanced_accuracy_score(yte, pn),
                         "k_pearson": float(np.corrcoef(Kn_tr[iu], Ktr[iu])[0, 1]), "cfg": repr(cfg), "two_qubit_gates": q2,
                         "depth": depth, "circuits": 3 * len(X), "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{ds}_{'_'.join(devs)}.csv", index=False)


if __name__ == "__main__":
    main()
