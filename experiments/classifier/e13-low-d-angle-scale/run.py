"""e13: angle scale of the core encoding below 8 plain columns (phase 9, found by scikit-learn's estimator checks).

Up: README.md · Log: ../learning-log.md (phase-9 addendum) · Code under test: src/qlassifier/encoding.py (MIN_SCALE_DIM)
Compares c0 = 2/sqrt(d) (phase-7 rule) with c0 = 2/sqrt(max(d, 8)) (shipped) at d < 8; at d >= 8 both are identical.
Run: .venv/bin/python experiments/classifier/e13-low-d-angle-scale/run.py
Writes results.csv next to this file.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

import qlassifier.encoding as enc_mod
from qlassifier import Qlassifier, load_dataset

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PLAN = enc_mod.Encoding.plan.__func__  # shipped rule: 2 / sqrt(max(d_plain, 8))


def phase7_rule(cls, n_features, geometry, names):  # type: ignore[no-untyped-def]
    e = PLAN(cls, n_features, geometry, names)
    return replace(e, scale=2.0 / math.sqrt(len(e.plain)) if e.plain else 1.0)


def tasks() -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    X_m, y_m = make_blobs(n_samples=300, random_state=0)  # scikit-learn check_classifiers_train problem
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    out = {"blobs (train acc)": (X_m[y_m != 2], y_m[y_m != 2], 0)}
    out["moons"] = (*make_moons(400, noise=0.2, random_state=0), 0)
    out["circles"] = (*make_circles(400, noise=0.1, factor=0.5, random_state=0), 0)
    X, y = load_dataset("heart", return_X_y=True)
    for k in (2, 4, 6, 8):
        out[f"heart PCA-{k}"] = (X.to_numpy(), y.to_numpy(), k)
    return out


def score(X: np.ndarray, y: np.ndarray, k: int, model: object) -> float:
    if k:
        model = make_pipeline(StandardScaler(), PCA(k, random_state=0), model)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(model, X, y, cv=cv).mean())


def main() -> None:
    rows = []
    for name, (X, y, k) in tasks().items():
        rbf = score(X, y, k, SVC())
        for rule, plan in (("2/sqrt(d) (phase 7)", classmethod(phase7_rule)), ("2/sqrt(max(d,8)) (shipped)", classmethod(PLAN))):
            enc_mod.Encoding.plan = plan  # type: ignore[method-assign]
            if name.startswith("blobs"):
                acc = float(Qlassifier().fit(X, y).score(X, y))
            else:
                acc = score(X, y, k, Qlassifier(random_state=0))
            rows.append({"task": name, "rule": rule, "qlassifier": round(acc, 3), "rbf_svm_default": round(rbf, 3)})
    enc_mod.Encoding.plan = classmethod(PLAN)  # type: ignore[method-assign]
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)
    print(df.pivot(index="task", columns="rule", values="qlassifier").join(df.groupby("task").rbf_svm_default.first()))


if __name__ == "__main__":
    main()
