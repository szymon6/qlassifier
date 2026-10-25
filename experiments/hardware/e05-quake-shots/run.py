"""e05: do more shots close the deep_quakes device loss? Shipped Qlassifier on FakeFez at 1,024 .. 65,536 shots.

Up: ../README.md §2.2 (H6) · Log: ../learning-log.md · Data: stage-1 deep_quakes.csv (registered by extra_datasets)
Run: PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python \
         experiments/hardware/e05-quake-shots/run.py
Fresh seeds 200-204 (stage-1 e07 seeds, so the RF reference pairs by seed); recipe native; geometry (lat, lon) = sphere;
exact-mode self-tuning on the training split (0 circuits), then measured features at S shots for fit and predict.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import extra_datasets  # noqa: E402,F401

from qlassifier import Qlassifier  # noqa: E402
from qlassifier.bench.protocol import fold_inputs  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402

OUT = Path(__file__).resolve().parent
SEEDS = (200, 201, 202, 203, 204)
SHOTS = (1024, 4096, 16384, 65536)
GEOM = {(0, 1): "sphere"}

rows = []
for seed in SEEDS:
    fi = fold_inputs("deep_quakes", seed, "stratified")
    t0 = time.perf_counter()
    ex = Qlassifier(geometry=GEOM, random_state=seed).fit(fi.X_train, fi.y_train)
    g, C = ex.best_params_["gamma"], ex.best_params_["C"]
    pred = ex.predict(fi.X_test)
    rows.append(dict(seed=seed, shots=0, backend="exact", gamma=g, C=C, accuracy=accuracy_score(fi.y_test, pred),
                     balanced_accuracy=balanced_accuracy_score(fi.y_test, pred), seconds=time.perf_counter() - t0))
    print(rows[-1], flush=True)
    for S in SHOTS:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = Qlassifier(geometry=GEOM, backend="fake_fez", shots=S, gamma=g, C=C, random_state=seed).fit(fi.X_train, fi.y_train)
            pred = clf.predict(fi.X_test)
        rows.append(dict(seed=seed, shots=S, backend="fake_fez", gamma=g, C=C, accuracy=accuracy_score(fi.y_test, pred),
                         balanced_accuracy=balanced_accuracy_score(fi.y_test, pred), seconds=time.perf_counter() - t0,
                         qubits=clf.resources_.qubits, two_qubit_gates=clf.resources_.two_qubit_gates))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

df = pd.DataFrame(rows)
ref = Path(__file__).resolve().parents[2] / "classifier" / "e07-structure-fit" / "quakes.csv"
summary = df.groupby(["backend", "shots"])[["accuracy", "balanced_accuracy"]].agg(["mean", "std"]).round(4)
print(summary.to_string())
if ref.exists():
    e07 = pd.read_csv(ref)
    rf = e07[(e07.dataset == "deep_quakes") & (e07.model == "random_forest") & (e07.seed.isin(SEEDS))]
    geo = e07[(e07.dataset == "deep_quakes") & (e07.model == "qlass_geo") & (e07.seed.isin(SEEDS))]
    print("e07 reference, seeds 200-204: RF BA", rf.ba.mean().round(4), "geometry exact BA", geo.ba.mean().round(4))
    json.dump({"rf_ba_mean": float(rf.ba.mean()), "rf_ba_by_seed": dict(zip(rf.seed.astype(str), rf.ba)),
               "e07_geo_ba_mean": float(geo.ba.mean())}, (OUT / "reference_e07.json").open("w"), indent=1)
