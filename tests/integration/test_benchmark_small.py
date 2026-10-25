"""benchmark() end to end on real data (TDD §4.5, §5.5, US-2): events, partials, determinism across n_jobs, cancel."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qlassifier import InvalidParameterError, UnknownDatasetError, UnknownModelError, benchmark, load_dataset
from qlassifier.bench import ProgressEvent

MODELS = ["trivial", "logistic_regression", "zz_qsvc", "pqk", "qlassifier"]


def _numbers(result: Any) -> list[tuple[Any, ...]]:
    """Per-run numbers without timings."""
    return [
        (r.seed, r.model, r.config_index, r.balanced_accuracy, r.accuracy, r.roc_auc, r.f1, r.edge) for r in result.runs
    ]


@pytest.fixture(scope="module")
def sequential() -> tuple[Any, list[ProgressEvent]]:
    events: list[ProgressEvent] = []
    result = benchmark("heart", seeds=[0, 1], models=MODELS, n_jobs=1, progress=events.append)
    return result, events


def test_progress_events_and_partials(sequential: tuple[Any, list[ProgressEvent]]) -> None:
    result, events = sequential
    total = 2 * len(MODELS) + 1
    assert [e.done for e in events] == list(range(1, total + 1))
    assert {e.total for e in events} == {total}
    assert [e.model for e in events].count("cost") == 1
    assert next(e for e in events if e.model == "cost").seed is None
    assert all(e.result.status == "partial" for e in events)
    n_runs = [len(e.result.runs) for e in events]
    assert n_runs == sorted(n_runs)
    assert n_runs[-1] == 2 * len(MODELS)
    assert all(e.eta_s is not None and e.eta_s >= 0 for e in events)
    assert events[-1].eta_s == 0
    assert result.status == "complete"
    assert result.protocol.models == MODELS
    assert result.environment.n_jobs == 1
    assert result.cost is not None
    assert result.noise is None


def test_numbers_do_not_depend_on_scheduling(sequential: tuple[Any, list[ProgressEvent]]) -> None:
    parallel = benchmark("heart", seeds=[0, 1], models=MODELS, n_jobs=2)
    assert _numbers(parallel) == _numbers(sequential[0])
    assert parallel.protocol.fingerprint == sequential[0].protocol.fingerprint
    assert parallel.model("qlassifier").metrics == sequential[0].model("qlassifier").metrics


def test_cancel_returns_the_partial_result() -> None:
    cancel = threading.Event()
    seen: list[ProgressEvent] = []

    def stop_after_three(ev: ProgressEvent) -> None:
        seen.append(ev)
        if ev.done == 3:
            cancel.set()

    result = benchmark("heart", seeds=[0, 1], models=MODELS, n_jobs=1, progress=stop_after_three, cancel=cancel)
    assert result.status == "cancelled"
    assert len(seen) == 3
    assert len(result.runs) + (result.cost is not None) == 3


def test_cancel_with_worker_processes() -> None:
    cancel = threading.Event()
    result = benchmark(
        "heart", seeds=[0, 1, 2], models=MODELS, n_jobs=2, progress=lambda ev: cancel.set(), cancel=cancel
    )
    assert result.status == "cancelled"
    assert 1 <= len(result.runs) + (result.cost is not None) < 3 * len(MODELS) + 1


@pytest.fixture
def small_protein(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """protein_angles cut to its first 600 rows (real rows, real chains) so the grouped engine path runs fast."""
    from qlassifier.bench import protocol

    full = protocol.arrays("protein_angles")
    cut = protocol.Arrays(full.X[:600], full.y[:600], full.groups[:600], full.recipe, full.geometry)  # type: ignore[index]
    original = protocol.arrays

    def arrays(name: str) -> protocol.Arrays:
        return cut if name == "protein_angles" else original(name)

    protocol.fold_inputs.cache_clear()
    monkeypatch.setattr(protocol, "arrays", arrays)
    yield
    protocol.fold_inputs.cache_clear()


@pytest.mark.usefixtures("small_protein")
def test_grouped_dataset_runs_geometry_mode_and_twin() -> None:
    result = benchmark("protein_angles", seeds=[0, 1], models=["trivial", "qlassifier", "classical_twin"], n_jobs=1)
    assert result.protocol.split == "stratified_grouped"  # 600 rows: test shares drift > 3 pp -> stratified groups
    assert result.protocol.recipe == "native"
    assert result.protocol.n_features_used == 10
    q = {r.seed: r for r in result.runs if r.model == "qlassifier"}
    t = {r.seed: r for r in result.runs if r.model == "classical_twin"}
    for seed in (0, 1):  # identical Gram matrix -> identical choices and predictions
        assert q[seed].config_index == t[seed].config_index
        assert q[seed].balanced_accuracy == pytest.approx(t[seed].balanced_accuracy, abs=1e-12)
    assert result.model("classical_twin").role == "twin"
    assert result.cost is not None
    assert result.cost.models[0].model == "qlassifier"
    assert result.cost.models[0].qubits == 10
    assert result.cost.models[0].two_qubit_gates == dict.fromkeys(("fake_torino", "fake_fez", "fake_brisbane"), 0)


def test_argument_errors() -> None:
    with pytest.raises(UnknownDatasetError):
        benchmark("iris")
    with pytest.raises(UnknownModelError):
        benchmark("heart", models=["classical_twin"])  # the twin exists only for declared geometry
    for seeds in (1, [3], [0, 0], [-1, 2], True, ["a", "b"]):
        with pytest.raises(InvalidParameterError):
            benchmark("heart", seeds=seeds)  # type: ignore[arg-type]
    for n_jobs in (0, -2, 1.5, True):
        with pytest.raises(InvalidParameterError):
            benchmark("heart", n_jobs=n_jobs)  # type: ignore[arg-type]


def test_noise_and_cost_on_a_small_run(heart_result: Any) -> None:
    noise = heart_result.noise
    assert noise is not None
    assert [p.id for p in noise.parts] == ["A"]  # part B runs on Breast Cancer only
    assert noise.parts[0].seeds == [0, 1]
    assert len(noise.records) == 2 * 4
    for rec in noise.records:
        assert rec.part == "A"
        assert rec.model == "qlassifier"
        assert rec.circuits == 3 * (rec.n_train + rec.n_test)
        assert rec.two_qubit_gates == 0
        assert rec.kernel_r is not None
        assert rec.kernel_r > 0.99
        assert abs(rec.drop_pp) < 5
    exact = [s for s in noise.summary if s.device == "exact"]
    assert len(exact) == 1
    assert exact[0].drop_mean_pp == 0


def test_benchmark_accepts_a_dataframe_and_a_csv_path(tmp_path: Path) -> None:
    """US-1 (stage 2, library part): the fair leaderboard on the user's own data."""
    X_train, _, y_train, _ = load_dataset("heart").split(seed=0)
    df = pd.DataFrame(X_train).assign(label=np.where(y_train == 1, "yes", "no"))
    r = benchmark(df, target="label", seeds=[0, 1], models=["trivial", "qlassifier"], n_jobs=1)
    assert r.status == "complete"
    assert r.dataset.id == "dataframe"
    assert r.dataset.positive_class == "yes"
    assert [m.id for m in r.models] == ["trivial", "qlassifier"]
    assert r.protocol.n_features_used == 8
    csv = tmp_path / "heart.csv"
    df.to_csv(csv, index=False)
    r2 = benchmark(str(csv), target="label", seeds=[0, 1], models=["trivial", "qlassifier"], n_jobs=2)
    assert r2.dataset.id == "heart_csv"
    assert r2.dataset.sha256 is not None
    assert r2.models[1].metrics.balanced_accuracy.mean == r.models[1].metrics.balanced_accuracy.mean  # same rows
    with pytest.raises(InvalidParameterError, match="CSV files only"):
        benchmark("heart", target="label", seeds=[0, 1])
