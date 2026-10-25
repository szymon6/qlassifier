"""Benchmark engine: ``benchmark()`` reruns the leaderboard under the fair protocol (TDD §4.5, §5.5, §7).

Tasks: one per (seed, model), one cost task, and the noise study cells with ``noise=True``. They run on a joblib
(loky) pool, longest first, one BLAS / OpenMP thread per worker; results are aggregated on the caller's thread
after every task (partial results for the dashboard). Modules: ``models`` (registry), ``protocol`` (splits,
folds, tuning, tasks), ``stats``, ``noise``, ``cost``, ``aggregate``, ``result``, ``plots``, ``report``.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qlassifier.bench.result import BenchmarkResult
from qlassifier.datasets import SplitMode, csv_dataset, is_csv_name, looks_like_csv
from qlassifier.errors import DATASET_IDS, InvalidParameterError, UnknownDatasetError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["BenchmarkResult", "ProgressEvent", "benchmark", "default_jobs"]

# Noise-study memory budget: one 10-qubit device cell peaks at about 2.4 GB (TDD §7.11), so one worker per core
# overran a 16 GB laptop twice. Default workers with ``noise=True``: (RAM - reserve) / per-worker, at least 1.
NOISE_WORKER_GIB = 4
NOISE_RESERVE_GIB = 4

# Expected seconds per task when no shipped result exists (Breast Cancer scale, one core); longest first.
_DEFAULT_WEIGHTS = {
    "trivial": 0.01, "logistic_regression": 0.3, "rbf_svm": 0.4, "random_forest": 25.0, "gradient_boosting": 8.0,
    "mlp": 30.0, "zz_qsvc": 0.5, "fqk_bandwidth_full": 2.0, "fqk_bandwidth_linear": 2.0, "pqk": 1.0,
    "separable": 0.5, "qlassifier": 0.6, "classical_twin": 0.5, "cost": 20.0,
}  # fmt: skip


@dataclass(frozen=True)
class ProgressEvent:
    """Sent to ``progress`` after each finished task (on the caller's thread).

    Attributes:
        done: finished tasks.
        total: all tasks of the run.
        seed: seed of the finished task (None for the cost task).
        model: model id of the finished task (``"cost"`` for the cost task).
        eta_s: estimated seconds left (None before the first estimate).
        result: the partial result after this task.
    """

    done: int
    total: int
    seed: int | None
    model: str
    eta_s: float | None
    result: BenchmarkResult


def _resolve_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, bool):
        raise InvalidParameterError(
            f"seeds must be an int >= 2 or a list of ints; got {seeds!r}.", "Use e.g. seeds=10."
        )
    if isinstance(seeds, int):
        if seeds < 2:
            raise InvalidParameterError(f"seeds must be at least 2; got {seeds}.", "Use e.g. seeds=10 (seeds 0-9).")
        return list(range(seeds))
    try:
        out = [int(s) for s in seeds]
    except (TypeError, ValueError):
        raise InvalidParameterError(
            f"seeds must be an int >= 2 or a list of ints; got {seeds!r}.", "Use e.g. seeds=10 or seeds=[0, 1, 2]."
        ) from None
    if len(out) < 2 or len(set(out)) != len(out) or any(s < 0 for s in out):
        raise InvalidParameterError(
            f"seeds must hold at least 2 distinct non-negative ints; got {out}.", "Use e.g. seeds=[0, 1, 2]."
        )
    return out


def _physical_memory() -> int | None:
    """Installed RAM in bytes, or None where the platform does not say."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def default_jobs(noise: bool, *, cores: int | None = None, memory_bytes: int | None = None) -> int:
    """Worker processes when ``n_jobs`` is not given: every core; with the noise study, capped by RAM.

    The cap is ``(RAM - 4 GiB) // 4 GiB`` workers (3 on a 16 GB laptop, 7 on 32 GB), never below 1 and never above
    the core count. ``cores`` and ``memory_bytes`` override the machine's values (tests); unknown RAM means no cap.
    """
    cpus = cores if cores is not None else (os.cpu_count() or 1)
    if not noise:
        return cpus
    mem = memory_bytes if memory_bytes is not None else _physical_memory()
    if mem is None:
        return cpus
    cap = int((mem / 2**30 - NOISE_RESERVE_GIB) // NOISE_WORKER_GIB)
    return max(1, min(cpus, cap))


def _resolve_jobs(n_jobs: int | None, noise: bool = False) -> int:
    if n_jobs is None or n_jobs == -1:
        return default_jobs(noise)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise InvalidParameterError(f"n_jobs must be a positive int or None; got {n_jobs!r}.", "Use e.g. n_jobs=4.")
    return n_jobs


def _expected_weights(dataset: str) -> dict[tuple[str, str | None, str | None], float]:
    """Expected seconds per (model, part, device) from the shipped result; empty when none is shipped."""
    import numpy as np

    from qlassifier.errors import NoResultError

    if dataset not in DATASET_IDS:
        return {}
    try:
        shipped = BenchmarkResult.cached(dataset)
    except NoResultError:
        return {}
    w: dict[tuple[str, str | None, str | None], float] = {}
    for m in {r.model for r in shipped.runs}:
        w[(m, None, None)] = float(np.median([r.t_tune + r.t_fit + r.t_predict for r in shipped.runs if r.model == m]))
    if shipped.noise is not None:
        for key in {(r.model, r.part, r.device) for r in shipped.noise.records}:
            secs = [r.seconds for r in shipped.noise.records if (r.model, r.part, r.device) == key]
            w[key] = float(np.median(secs))
    return w


def _execute(task: Any, mode: SplitMode, models: tuple[str, ...]) -> tuple[Any, Any]:
    """Run one task (in a worker or in-process) with one BLAS / OpenMP thread and quiet warnings."""
    from threadpoolctl import threadpool_limits

    from qlassifier.bench.cost import cost_study
    from qlassifier.bench.noise import run_noise_task
    from qlassifier.bench.protocol import run_model

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task.kind == "run":
            return task, run_model(task.dataset, task.seed, task.model, mode)
        if task.kind == "cost":
            return task, cost_study(task.dataset, task.seed, mode, models)
        return task, run_noise_task(task, mode)


def _schedule(
    tasks: Sequence[Any], mode: SplitMode, models: tuple[str, ...], n_jobs: int, cancel: threading.Event | None
) -> Iterator[tuple[Any, Any]]:
    """Yield (task, output) as tasks finish; stop dispatching once ``cancel`` is set."""
    if n_jobs == 1:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                return
            yield _execute(t, mode, models)
        return
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1), warnings.catch_warnings():
        # loky replaces a worker whose memory grew (large Grams, Aer simulators) and warns; results are unaffected
        warnings.filterwarnings("ignore", message="A worker stopped while some jobs", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*still being processed by the workers have been cancelled")
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_execute)(t, mode, models) for t in tasks
        )
        try:
            for item in gen:
                yield item
                if cancel is not None and cancel.is_set():
                    return
        finally:
            gen.close()


def _resolve_dataset(dataset: Any, target: str | None) -> str:
    """A bundled id, or the registered name of a CSV path / DataFrame (written to a temporary CSV for the workers)."""
    import pandas as pd

    if isinstance(dataset, pd.DataFrame):
        tmp = Path(tempfile.mkdtemp(prefix="qlassifier-")) / "dataframe.csv"
        dataset.to_csv(tmp, index=False)
        return csv_dataset(tmp, target=target)
    if is_csv_name(dataset):
        if target is not None:
            raise InvalidParameterError("target cannot be combined with a registered CSV name.", "Drop target=.")
        return str(dataset)
    if isinstance(dataset, str) and dataset in DATASET_IDS:
        if target is not None:
            raise InvalidParameterError(
                f"target applies to CSV files only; '{dataset}' is a bundled dataset.", "Drop target=."
            )
        return dataset
    if target is not None or looks_like_csv(dataset):
        return csv_dataset(dataset, target=target)
    raise UnknownDatasetError(dataset)


def benchmark(
    dataset: str | Path | pd.DataFrame = "breast_cancer",
    *,
    target: str | None = None,
    seeds: int | Sequence[int] = 10,
    models: Sequence[str] | None = None,
    noise: bool = False,
    n_jobs: int | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
    cancel: threading.Event | None = None,
) -> BenchmarkResult:
    """Rerun the leaderboard: Qlassifier vs every baseline under the fair protocol (TDD §7).

    Same splits, same per-fold preprocessing and features, 30 configs per tuned model, 5-fold CV on balanced
    accuracy, corrected 95% CIs and paired tests over seeds.

    Args:
        dataset: a bundled dataset id, the path of your own CSV file, or a DataFrame (numeric feature columns and
            one binary target column; rules in :func:`qlassifier.datasets.csv_dataset`).
        target: for a CSV file or DataFrame, the target column (default: the last column).
        seeds: 10 -> seeds 0-9 (the protocol); or an explicit list (at least 2).
        models: subset of model ids (None -> the full protocol set for the dataset; BT checks need the full set).
        noise: add the noise study (exact vs shots vs IBM device noise models; slow).
        n_jobs: worker processes (None -> all cores; with ``noise=True`` capped by RAM, see :func:`default_jobs`).
        progress: called after each finished task with a :class:`ProgressEvent` (on the caller's thread).
        cancel: set it to stop after the running tasks; the partial result is returned with status "cancelled".

    Returns:
        The :class:`BenchmarkResult`.

    Example:
        >>> from qlassifier import benchmark
        >>> print(benchmark("breast_cancer").summary())  # doctest: +SKIP
        >>> print(benchmark("my_data.csv", target="label").summary())  # doctest: +SKIP
    """
    from qlassifier.bench.aggregate import build_result
    from qlassifier.bench.cost import QUANTUM_MODELS
    from qlassifier.bench.noise import PART_B_MODELS, noise_tasks
    from qlassifier.bench.protocol import (
        Task,
        arrays,
        decide_split,
        fingerprint,
        protocol_models,
        test_train_ratio,
    )
    from qlassifier.datasets import load_dataset

    dataset = _resolve_dataset(dataset, target)
    seed_list = _resolve_seeds(seeds)
    jobs = _resolve_jobs(n_jobs, noise)
    specs = protocol_models(dataset, models)
    ids = tuple(s.id for s in specs)
    mode = decide_split(dataset, seed_list)
    ratio = test_train_ratio(dataset, seed_list, mode)
    fp = fingerprint(dataset, seed_list, ids, mode)
    info = load_dataset(dataset).info

    expected = _expected_weights(dataset)
    tasks = [
        Task("run", dataset, s, m, weight=expected.get((m, None, None), _DEFAULT_WEIGHTS[m]))
        for s in seed_list
        for m in ids
    ]
    if any(m in QUANTUM_MODELS for m in ids):
        tasks.append(Task("cost", dataset, seed_list[0], "cost", weight=_DEFAULT_WEIGHTS["cost"]))
    if noise:
        tasks += [
            Task(
                t.kind,
                t.dataset,
                t.seed,
                t.model,
                t.device,
                t.part,
                expected.get((t.model, t.part, t.device), t.weight),
            )
            for t in noise_tasks(dataset, seed_list)
            if t.model in ids and (t.part == "A" or t.model in PART_B_MODELS)
        ]
    tasks.sort(key=lambda t: -t.weight)  # stable: ties keep (seed, model) order
    total, total_w = len(tasks), sum(t.weight for t in tasks)

    runs: list[Any] = []
    noise_records: list[Any] = []
    cost = None
    done, done_w = 0, 0.0
    t0 = time.perf_counter()

    def snapshot(status: Any) -> BenchmarkResult:
        return build_result(
            info=info, seeds=seed_list, specs=specs, mode=mode, ratio=ratio, fingerprint=fp,
            n_features_used=arrays(dataset).d_used, runs=runs, cost=cost, noise_records=noise_records,
            status=status, wall_s=time.perf_counter() - t0, n_jobs=jobs,
        )  # fmt: skip

    for task, out in _schedule(tasks, mode, ids, jobs, cancel):
        if task.kind == "run":
            runs.append(out)
        elif task.kind == "cost":
            cost = out
        else:
            noise_records.append(out)
        done += 1
        done_w += task.weight
        if progress is not None:
            elapsed = time.perf_counter() - t0
            eta = 0.0 if done == total else max(0.0, elapsed * (total_w - done_w) / done_w) if done_w > 0 else None
            seed = None if task.kind == "cost" else task.seed
            progress(ProgressEvent(done, total, seed, task.model, eta, snapshot("partial")))
    status = "complete" if done == total else "cancelled"
    return snapshot(status)
