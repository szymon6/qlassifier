"""Golden check: the shipped engine reproduces every row of the phase-7 reference leaderboard (e01).

Up: README.md · Reference: ../experiments/e01-reproduce/results.csv (480 rows: 4 UCI datasets x seeds 0-9 x 12 models)
Pass: chosen config index identical, edge flag identical, BA / accuracy / F1 / ROC-AUC within 1e-9.
Run: .venv/bin/python benchmarks/golden_e01.py   (about 5 min on 10 cores)
Writes golden_e01.txt next to this file. CI runs a 21-row subset (tests/integration/test_golden_e01.py).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
E01 = HERE.parent / "experiments" / "e01-reproduce" / "results.csv"
NAMES = {"logreg": "logistic_regression", "grad_boost": "gradient_boosting", "fqk_bw_full": "fqk_bandwidth_full",
         "fqk_bw_linear": "fqk_bandwidth_linear", "bloch_gauss": "qlassifier"}  # fmt: skip


def run(dataset: str, seed: int, model: str):  # type: ignore[no-untyped-def]
    from threadpoolctl import threadpool_limits

    from qlassifier.bench.protocol import run_model

    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_model(dataset, seed, model, "stratified")


def main() -> int:
    ref = pd.read_csv(E01)
    ref["model"] = ref["model"].map(lambda m: NAMES.get(m, m))
    t0 = time.time()
    records = Parallel(n_jobs=-1)(delayed(run)(r.dataset, int(r.seed), r.model) for r in ref.itertuples())
    bad = []
    for r, rec in zip(ref.itertuples(), records, strict=True):
        dm = max(abs(rec.balanced_accuracy - r.ba), abs(rec.accuracy - r.acc), abs(rec.f1 - r.f1),
                 abs((rec.roc_auc if rec.roc_auc is not None else float("nan")) - r.auc))  # fmt: skip
        if rec.config_index != r.cfg_idx or rec.edge != bool(r.edge) or not dm <= 1e-9:
            bad.append(f"{r.dataset} seed {r.seed} {r.model}: index {rec.config_index}/{r.cfg_idx} max diff {dm}")
    text = f"{len(ref)} rows, {len(bad)} mismatches, {time.time() - t0:.0f} s\n" + "".join(f"{b}\n" for b in bad)
    (HERE / "golden_e01.txt").write_text(text)
    print(text, end="")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
