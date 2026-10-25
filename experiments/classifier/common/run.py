"""Parallel runner over (dataset, seed) for the exact-mode protocol in lab.py.

Usage:
  .venv/bin/python run.py OUT.csv --datasets breast_cancer ionosphere --seeds 0-9 --k 8 [--recipe pca|native] [--models a b] [--pqk-t 0.3] [--jobs 10]
Up: ../README.md (protocol §2) · Log: ../learning-log.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


def _seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--datasets", nargs="+", default=lab.DATASETS)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--k", type=int, default=8, help="PCA components; 0 = all features")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--pqk-t", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--recipe", default="pca", choices=["pca", "native"])
    a = ap.parse_args()
    t0 = time.time()
    tasks = [(ds, s) for ds in a.datasets for s in _seeds(a.seeds)]
    res = Parallel(n_jobs=a.jobs, verbose=0)(
        delayed(lab.run_seed)(ds, s, a.k or None, a.models, {"pqk_t": a.pqk_t}, recipe=a.recipe) for ds, s in tasks
    )
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows in {time.time() - t0:.1f} s")
    print(df.groupby(["dataset", "model"])[["ba", "acc", "t_tune", "t_fit"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
