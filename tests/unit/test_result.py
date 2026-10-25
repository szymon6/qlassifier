"""BenchmarkResult (TDD §5.5, §7.12): JSON round trip + schema, views, plots, save, cached, shipped marker."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from matplotlib.figure import Figure

from qlassifier import BenchmarkResult, InvalidParameterError, NoResultError, UnknownDatasetError
from tests.conftest import ROOT

SCHEMA = json.loads((ROOT / "src" / "qlassifier" / "schemas" / "benchmark-result.json").read_text())


def test_json_round_trip_and_schema(heart_result: Any, tmp_path: Path) -> None:
    text = heart_result.to_json()
    jsonschema.validate(json.loads(text), SCHEMA)
    assert BenchmarkResult.from_json(text) == heart_result
    path = tmp_path / "r.json"
    heart_result.to_json(path)
    assert BenchmarkResult.from_json(path) == heart_result
    assert BenchmarkResult.from_json(str(path)) == heart_result


def test_result_contents(heart_result: Any) -> None:
    r = heart_result
    assert r.status == "complete"
    assert r.source == "run"
    assert r.protocol.seeds == [0, 1]
    assert r.protocol.split == "stratified"
    assert r.protocol.recipe == "pca8"
    assert r.protocol.n_features_used == 8
    assert r.protocol.configs_per_model["zz_qsvc"] == 4
    assert r.protocol.configs_per_model["trivial"] == 0
    assert len(r.protocol.fingerprint) == 64
    assert len(r.runs) == 2 * 5
    assert [m.id for m in r.models] == [
        "trivial",
        "logistic_regression",
        "zz_qsvc",
        "fqk_bandwidth_linear",
        "qlassifier",
    ]
    assert sorted(m.rank for m in r.models) == [1, 2, 3, 4, 5]
    assert {c.model for c in r.comparisons} == {"trivial", "logistic_regression", "zz_qsvc", "fqk_bandwidth_linear"}
    assert r.highlights.best_classical == "logistic_regression"
    assert r.highlights.published is not None
    assert r.highlights.published.value == 0.804
    assert r.highlights.vs_standard_qml is not None
    assert r.highlights.vs_standard_qml.mean_diff > 0.2
    assert r.highlights.circuit_ratio_vs_fqk_pipeline is not None
    assert r.noise is not None
    assert r.noise.source == "run"
    assert r.noise.created_at == r.created_at
    assert {s.device for s in r.noise.summary} == {"exact", "shots", "fake_torino", "fake_fez", "fake_brisbane"}
    assert r.highlights.noise_drop_max_pp is not None
    assert r.cost is not None
    assert [m.model for m in r.cost.models] == ["zz_qsvc", "fqk_bandwidth_linear", "qlassifier"]
    q = r.model("qlassifier")
    assert q.metrics.balanced_accuracy.n == 2
    with pytest.raises(KeyError):
        r.model("svm")


def test_summary_frame_and_repr(heart_result: Any, bt10_result: Any) -> None:
    text = heart_result.summary()
    assert "Heart Disease (Cleveland)" in text
    assert "Qlassifier" in text
    # 2 seeds: no tie star anywhere, the legend says why, and printed CIs stay inside [0, 1]
    assert "ties need ≥ 5 seeds (none marked at 2)" in text
    assert " *" not in text
    assert "Qlassifier − model" in text
    assert "Qlassifier wins" in text
    for token in re.findall(r"\[(-?\d\.\d{3}), (-?\d\.\d{3})\]", text):
        assert all(0.0 <= float(v) <= 1.0 for v in token)
    for metric in ("accuracy", "roc_auc", "f1"):
        assert "Majority class" in heart_result.summary(metric)
    # 10 seeds: ties are marked; Δ is Qlassifier − model (the MLP row reads +1.3 pp for Qlassifier −1.3 below it)
    full = bt10_result.summary()
    assert "* statistically tied with the best" in full
    assert "Qlassifier *" in full
    mlp = next(line for line in full.splitlines() if line.startswith("MLP"))
    comp = next(c for c in bt10_result.comparisons if c.model == "mlp").metrics.balanced_accuracy
    assert f"{100 * comp.mean_diff:+.1f} pp" in mlp
    assert comp.mean_diff < 0
    frame = heart_result.to_frame()
    assert len(frame) == 10
    assert {"seed", "model", "balanced_accuracy", "config", "t_tune"} <= set(frame.columns)
    rep = repr(heart_result)
    assert rep.startswith("BenchmarkResult(heart: 5 models × 2 seeds, complete, Qlassifier BA 0.")
    assert "Δ vs ZZ +" in rep
    assert "1.060" not in rep  # the raw upper CI end (above 1) is clipped in the repr
    assert str(heart_result) == rep


def test_plots(heart_result: Any, bt10_result: Any) -> None:
    for kind in ("leaderboard", "noise", "cost"):
        assert isinstance(heart_result.plot(kind), Figure)
    assert isinstance(heart_result.plot("leaderboard", metric="f1"), Figure)
    ax = Figure().add_subplot()
    assert heart_result.plot("noise", ax=ax) is ax.figure
    ax2 = Figure().add_subplot()
    assert heart_result.plot("cost", ax=ax2) is ax2.figure
    assert isinstance(bt10_result.plot("noise"), Figure)  # no noise study: an empty, labelled figure
    no_cost = bt10_result.model_copy(update={"cost": None})
    assert isinstance(no_cost.plot("cost"), Figure)
    with pytest.raises(InvalidParameterError):
        heart_result.plot("pie")  # type: ignore[arg-type]


def test_save_writes_the_output_layout(heart_result: Any, tmp_path: Path) -> None:
    out = heart_result.save(tmp_path / "heart")
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert names == [
        "noise.csv", "plots/cost.pdf", "plots/cost.png", "plots/leaderboard.pdf", "plots/leaderboard.png",
        "plots/noise.pdf", "plots/noise.png", "result.json", "runs.csv",
    ]  # fmt: skip
    runs = (out / "runs.csv").read_text().splitlines()
    assert len(runs) == 11
    assert '"{""C"":' in runs[2] or "{" in runs[2]  # configs as JSON text


def test_shipped_marker_and_cache(heart_result: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import qlassifier.bench.result as result_mod

    shipped = heart_result.as_shipped()
    assert shipped.source == "shipped"
    assert shipped.noise.source == "shipped"
    assert shipped.noise.created_at == heart_result.created_at
    assert shipped.runs == heart_result.runs
    monkeypatch.setattr(result_mod, "RESULTS_DIR", tmp_path)
    with pytest.raises(NoResultError):
        BenchmarkResult.cached("heart")
    shipped.to_json(tmp_path / "heart.json")
    assert BenchmarkResult.cached("heart") == shipped
    with pytest.raises(UnknownDatasetError):
        BenchmarkResult.cached("iris")


def test_partial_result_without_runs_is_valid() -> None:
    from qlassifier.server.runs import RunManager, _RunState

    empty = RunManager()._empty_result(_RunState(id="r-1", dataset="sonar", seeds=[0, 1], total=25, n_models=12))
    jsonschema.validate(json.loads(empty.to_json()), SCHEMA)
    assert empty.models == []
    assert empty.status == "partial"
    assert repr(empty) == "BenchmarkResult(sonar: 0 models × 0 seeds, partial)"
    assert isinstance(empty.plot("leaderboard"), Figure)  # a run stopped before any model finished
