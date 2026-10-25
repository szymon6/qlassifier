"""Parallel runner over (dataset, seed) with candidate specs from cands.py (protocol §2 of ../README.md).

Usage:
  .venv/bin/python run_x.py OUT.csv --datasets ionosphere --seeds 100-109 --k 8 [--recipe pca|native]
        [--models logreg rbf_svm ... | --models none] [--extra cands:spectrum_specs cands:balanced_specs] [--grouped] [--jobs 10]
Up: ../README.md · Log: ../learning-log.md
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
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


def _extra(names: list[str]):
    fns = []
    for n in names:
        mod, fn = n.split(":")
        fns.append(getattr(importlib.import_module(mod), fn))
    return (lambda d: [s for f in fns for s in f(d)]) if fns else None


def _task(ds, seed, k, models, recipe, extra_names, pqk_t, grouped=False):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    groups = lab.load_groups(ds) if grouped else None
    return lab.run_seed(ds, seed, k, models, {"pqk_t": pqk_t}, recipe=recipe, extra_specs=_extra(extra_names), groups=groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--datasets", nargs="+", default=lab.DATASETS)
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--k", type=int, default=8, help="PCA components; 0 = all features")
    ap.add_argument("--recipe", default="pca", choices=["pca", "native"])
    ap.add_argument("--models", nargs="*", default=None, help="base specs from lab.specs; 'none' = only --extra")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--pqk-t", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--grouped", action="store_true", help="grouped outer split + grouped inner CV (lab.load_groups)")
    a = ap.parse_args()
    models = [] if a.models == ["none"] else a.models
    t0 = time.time()
    tasks = [(ds, s) for ds in a.datasets for s in _seeds(a.seeds)]
    res = Parallel(n_jobs=a.jobs, verbose=0)(
        delayed(_task)(ds, s, a.k or None, models, a.recipe, a.extra, a.pqk_t, a.grouped) for ds, s in tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows in {time.time() - t0:.1f} s")
    print(df.groupby(["dataset", "model"])[["ba", "acc", "t_tune"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
