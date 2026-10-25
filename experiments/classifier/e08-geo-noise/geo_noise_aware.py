"""e08b: noise-aware tuning for geometry mode (coordinator's suggestion), deep_quakes, FakeFez, fresh seeds.

Up: README.md · Base: geo_noise.py (circuits, measurement) · Protocol grid: ../common/lab.py (kgrid, 30 configs)
Compares, on the same measured features (train and test both measured on the device, 1,024 shots):
  exact_cfg  : (m, C) from exact-mode CV (e07), as in the shipped design;
  aware_cfg  : (m, C) from 5-fold CV on the *measured* training features (0 extra circuits: geometry features
               need no per-fold preprocessing, so the measured rows are reused by every fold).
Usage: .venv/bin/python geo_noise_aware.py DATASET SEEDS(comma) DEVICE
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import cands  # noqa: E402
import geo_noise as gn  # noqa: E402
import lab  # noqa: E402


def cv_pick(R: np.ndarray, y: np.ndarray, seed: int) -> dict:
    grid = lab.kgrid(1.0, square=True)
    skf = StratifiedKFold(lab.N_FOLDS, shuffle=True, random_state=seed)
    sc = np.zeros(len(grid))
    for a, b in skf.split(R, y):
        cache = {}
        for i, cfg in enumerate(grid):
            if cfg["bw"] not in cache:
                cache[cfg["bw"]] = cands._gauss(R[a], R[b], cfg["bw"])
            Ka, Kb = cache[cfg["bw"]]
            sc[i] += balanced_accuracy_score(y[b], SVC(C=cfg["C"], kernel="precomputed").fit(Ka, y[a]).predict(Kb))
    return grid[int(np.argmax(sc))]


def main():
    ds, seeds, dv = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    geom = lab.GEOMETRY[ds]
    X, y = lab.load(ds)
    e07 = pd.read_csv(HERE.parent / "e07-structure-fit" / ("quakes.csv" if ds == "deep_quakes" else "protein.csv"))
    be = gn.BACKENDS[dv]()
    rows = []
    for seed in seeds:
        tr, te = next(StratifiedShuffleSplit(1, test_size=lab.TEST_SIZE, random_state=seed).split(X, y))
        ytr, yte = y[tr], y[te]
        cfg_x = ast.literal_eval(e07[(e07.seed == seed) & (e07.model == "qlass_geo")].cfg.iloc[0])
        Ftr, Fte = cands.geo_bloch(X[tr], X[tr], **geom), cands.geo_bloch(X[te], X[tr], **geom)
        Rtr, _, _ = gn.measure(be, X[tr], geom)
        Rte, _, _ = gn.measure(be, X[te], geom)
        cfg_a = cv_pick(Rtr, ytr, seed)
        for tag, (A, B, cfg) in {"exact": (Ftr, Fte, cfg_x), "noisy_exact_cfg": (Rtr, Rte, cfg_x),
                                 "noisy_aware_cfg": (Rtr, Rte, cfg_a)}.items():
            Ka, Kb = cands._gauss(A, B, cfg["bw"])
            p = SVC(C=cfg["C"], kernel="precomputed").fit(Ka, ytr).predict(Kb)
            rows.append({"dataset": ds, "seed": seed, "device": dv, "variant": tag, "acc": accuracy_score(yte, p),
                         "ba": balanced_accuracy_score(yte, p), "cfg": repr(cfg)})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"aware_{ds}_{dv}.csv", index=False)


if __name__ == "__main__":
    main()
