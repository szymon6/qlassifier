"""Records -> ``BenchmarkResult``: leaderboard rows, paired comparisons, ties and highlights (TDD §5.5, §7.7).

Runs on the caller's thread after every finished task, so partial results carry the same statistics as the final
one (the dashboard never recomputes them). Records are sorted by (seed, model order) before aggregation, so the
numbers never depend on scheduling.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Literal

import numpy as np

from qlassifier.bench import stats
from qlassifier.bench.cost import fqk_pipeline_ratio
from qlassifier.bench.models import CLASSICAL, MODEL_IDS, MODEL_NAMES, ModelSpec
from qlassifier.bench.noise import max_device_drop, noise_study
from qlassifier.bench.protocol import CI_LEVEL, N_FOLDS, PUBLISHED
from qlassifier.bench.result import (
    METRICS,
    BenchmarkResult,
    Comparison,
    CostStudy,
    Environment,
    Highlights,
    MetricSummary,
    ModelMetrics,
    ModelSummary,
    NoiseRecord,
    PairedDiff,
    PairedMetrics,
    Protocol,
    RunRecord,
    Times,
    Versions,
)
from qlassifier.datasets import TEST_SIZE, DatasetInfo, SplitMode

__all__ = ["build_result", "versions"]

_DISTS = {
    "qlassifier": "qlassifier",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "pandas": "pandas",
    "joblib": "joblib",
    "qiskit": "qiskit",
    "qiskit_aer": "qiskit-aer",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
}


def versions() -> Versions:
    """Installed versions of the packages that produce the numbers (read from metadata, nothing imported)."""
    return Versions(python=platform.python_version(), **{k: version(v) for k, v in _DISTS.items()})


def _metric_values(runs: Sequence[RunRecord], metric: str) -> dict[int, float]:
    return {r.seed: float(v) for r in runs if (v := getattr(r, metric)) is not None}


def _paired_diff(a: dict[int, float], b: dict[int, float], ratio: float) -> PairedDiff:
    common = sorted(set(a) & set(b))
    if not common:
        return PairedDiff(mean_diff=0.0, ci_low=None, ci_high=None, p_value=None, win_rate=0.0, n=0)
    m, lo, hi, p, w, n = stats.paired([a[s] for s in common], [b[s] for s in common], ratio, CI_LEVEL)
    return PairedDiff(mean_diff=m, ci_low=lo, ci_high=hi, p_value=p, win_rate=w, n=n)


def build_result(
    *,
    info: DatasetInfo,
    seeds: Sequence[int],
    specs: Sequence[ModelSpec],
    mode: SplitMode,
    ratio: float,
    fingerprint: str,
    n_features_used: int,
    runs: Sequence[RunRecord],
    cost: CostStudy | None,
    noise_records: Sequence[NoiseRecord],
    status: Literal["partial", "complete", "cancelled"],
    wall_s: float,
    n_jobs: int,
) -> BenchmarkResult:
    """Assemble a ``BenchmarkResult`` from finished records (any subset of the tasks)."""
    order = {m: i for i, m in enumerate(MODEL_IDS)}
    runs = sorted(runs, key=lambda r: (r.seed, order[r.model]))
    by_model: dict[str, list[RunRecord]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)
    present = [s for s in specs if s.id in by_model]

    ties = {
        metric: stats.tied_with_best({s.id: _metric_values(by_model[s.id], metric) for s in present}, ratio, CI_LEVEL)
        for metric in METRICS
    }
    rows: list[ModelSummary] = []
    for s in present:
        recs = by_model[s.id]
        summaries = {}
        for metric in METRICS:
            vals = list(_metric_values(recs, metric).values())
            mean, sd, lo, hi = stats.mean_ci(vals, ratio, CI_LEVEL) if vals else (0.0, None, None, None)
            summaries[metric] = MetricSummary(
                mean=mean, std=sd, ci_low=lo, ci_high=hi, n=len(vals), tied_with_best=ties[metric].get(s.id, False)
            )
        rows.append(
            ModelSummary(
                id=s.id,
                name=MODEL_NAMES[s.id],
                family=s.family,
                role=s.role,
                metrics=ModelMetrics(**summaries),
                time_s=Times(
                    tune=float(np.median([r.t_tune for r in recs])),
                    fit=float(np.median([r.t_fit for r in recs])),
                    predict=float(np.median([r.t_predict for r in recs])),
                ),
                n_seeds_done=len(recs),
                rank=0,
                edge_share=float(np.mean([r.edge for r in recs])),
            )
        )
    by_ba = sorted(rows, key=lambda m: -m.metrics.balanced_accuracy.mean)
    ranks = {m.id: i + 1 for i, m in enumerate(by_ba)}
    rows = [m.model_copy(update={"rank": ranks[m.id]}) for m in rows]

    comparisons: list[Comparison] = []
    q = by_model.get("qlassifier")
    if q is not None:
        for s in present:
            if s.id == "qlassifier":
                continue
            other = by_model[s.id]
            diffs = {
                metric: _paired_diff(_metric_values(q, metric), _metric_values(other, metric), ratio)
                for metric in METRICS
            }
            comparisons.append(Comparison(model=s.id, metrics=PairedMetrics(**diffs)))
    comp = {c.model: c.metrics.balanced_accuracy for c in comparisons}

    classical = [m for m in rows if m.id in CLASSICAL]
    best_classical = max(classical, key=lambda m: m.metrics.balanced_accuracy.mean).id if classical else None
    published = PUBLISHED.get(info.id)
    q_row = next((m for m in rows if m.id == "qlassifier"), None)
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    noise = noise_study(info.id, seeds, noise_records)
    if noise is not None:
        noise = noise.model_copy(update={"created_at": created_at})
    highlights = Highlights(
        best_classical=best_classical,
        vs_standard_qml=comp.get("zz_qsvc"),
        vs_best_classical=comp.get(best_classical) if best_classical else None,
        published=published,
        vs_published_pp=(
            100.0 * (q_row.metrics.balanced_accuracy.mean - published.value) if published and q_row else None
        ),
        noise_drop_max_pp=max_device_drop(noise),
        circuit_ratio_vs_fqk_pipeline=fqk_pipeline_ratio(cost),
    )
    protocol = Protocol(
        fingerprint=fingerprint,
        seeds=[int(s) for s in seeds],
        test_size=TEST_SIZE,
        split=mode,
        test_train_ratio=ratio,
        cv_folds=N_FOLDS,
        recipe=info.recipe,
        n_features_used=n_features_used,
        configs_per_model={s.id: (len(s.grid) if len(s.grid) > 1 else 0) for s in specs},
        selection_metric="balanced_accuracy",
        ci_level=CI_LEVEL,
        statistics="nadeau_bengio_corrected_t",
        models=[s.id for s in specs],
    )
    return BenchmarkResult(
        status=status,
        source="run",
        created_at=created_at,
        versions=versions(),
        environment=Environment(
            platform=platform.platform(),
            machine=platform.machine(),
            cpu_count=os.cpu_count() or 1,
            n_jobs=n_jobs,
            wall_s=wall_s,
        ),
        dataset=info,
        protocol=protocol,
        models=rows,
        comparisons=comparisons,
        highlights=highlights,
        runs=list(runs),
        noise=noise,
        cost=cost,
    )
