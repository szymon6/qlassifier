"""Golden test (TDD §7.13): the shipped engine reproduces the phase-7 reference leaderboard rows (e01) exactly.

CI-sized subset: Heart seed 0 with every model except the MLP, Breast Cancer seed 0 with the kernel and linear models.
Chosen config identical, edge flag identical, every metric within 1e-9 on the reference platform (macOS arm64,
Accelerate BLAS: where e01 was produced). On other platforms (Linux CI, OpenBLAS) matmul rounding differs in the
last bits, which can swap the rank of two test points with near-equal SVM margins: ROC-AUC then moves by exactly one
resolution step, 1 / (n_pos * n_neg) (1.46e-4 on Breast Cancer seed 0, seen for `separable` and `pqk`). Label-based
metrics, the chosen config and the edge flag stay exact everywhere. The MLP is left out because lbfgs follows a
different path under different BLAS rounding; its rows are checked exactly on the reference platform with the rest
of the full e01 table (480 / 480: benchmarks/golden_e01.py, .txt).
"""

from __future__ import annotations

import ast
import platform
import sys
from typing import Any

import pandas as pd
import pytest
from joblib import Parallel, delayed

from qlassifier.bench.protocol import fold_inputs, run_model


def _run(dataset: str, seed: int, model: str) -> Any:
    import warnings

    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_model(dataset, seed, model, "stratified")


PLATFORM_SENSITIVE = {"mlp"}
REFERENCE_PLATFORM = sys.platform == "darwin" and platform.machine() == "arm64"


def auc_tolerance(dataset: str, seed: int) -> float:
    """One ROC-AUC resolution step, 1 / (n_pos * n_neg) of the test split, allowed off the reference platform."""
    if REFERENCE_PLATFORM:
        return 0.0
    y_test = fold_inputs(dataset, seed, "stratified").y_test
    n_pos = int(y_test.sum())
    return 1.0 / (n_pos * (len(y_test) - n_pos))


@pytest.fixture(scope="module")
def engine_rows(e01_rows: pd.DataFrame) -> list[tuple[Any, Any]]:
    rows = e01_rows[~e01_rows.model.isin(PLATFORM_SENSITIVE)]
    records = Parallel(n_jobs=2)(delayed(_run)(r.dataset, int(r.seed), r.model) for r in rows.itertuples())
    return list(zip(rows.itertuples(), records, strict=True))


def test_golden_rows_are_reproduced_exactly(engine_rows: list[tuple[Any, Any]]) -> None:
    assert len(engine_rows) == 20
    mismatches = []
    for ref, rec in engine_rows:
        labels = max(abs(rec.balanced_accuracy - ref.ba), abs(rec.accuracy - ref.acc), abs(rec.f1 - ref.f1))
        auc = abs((rec.roc_auc if rec.roc_auc is not None else float("nan")) - ref.auc)
        exact = rec.config_index == ref.cfg_idx and rec.edge == bool(ref.edge) and labels <= 1e-9
        if not exact or not auc <= 1e-9 + auc_tolerance(ref.dataset, int(ref.seed)):
            mismatches.append(f"{ref.dataset} seed {ref.seed} {ref.model}: index {rec.config_index} vs {ref.cfg_idx}, "
                              f"label-metric diff {labels:.3g}, ROC-AUC diff {auc:.3g}")  # fmt: skip
    assert not mismatches, "\n".join(mismatches)


def test_golden_qlassifier_row_in_process(e01_rows: pd.DataFrame) -> None:
    ref = e01_rows[(e01_rows.dataset == "heart") & (e01_rows.model == "qlassifier")].iloc[0]
    rec = run_model("heart", 0, "qlassifier", "stratified")
    assert rec.config_index == ref.cfg_idx
    assert rec.balanced_accuracy == pytest.approx(ref.ba, abs=1e-9)
    cfg = ast.literal_eval(ref.cfg)  # phase-7 grid entry: {"bw": gamma multiplier, "C": ..., "m": ..., "ct": ...}
    assert rec.config == {"gamma": cfg["bw"], "C": cfg["C"]}
