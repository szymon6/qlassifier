"""Benchmark target rules BT-1...BT-10 (TDD §7.10) on the shipped results, plus every missed / not-measured branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qlassifier.bench.report import STATUSES, bt10_from, evaluate, geometry_qubit_limit, write_report
from qlassifier.bench.result import BenchmarkResult
from qlassifier.encoding import Encoding
from qlassifier.errors import DATASET_IDS


@pytest.fixture(scope="module")
def shipped() -> dict[str, BenchmarkResult]:
    return {ds: BenchmarkResult.cached(ds) for ds in DATASET_IDS}


def _status(targets: list[Any]) -> dict[str, str]:
    return {t.id: t.status for t in targets}


def _with_mean(result: BenchmarkResult, model: str, metric: str, value: float) -> BenchmarkResult:
    """A copy of ``result`` where one model's mean of one metric is replaced."""
    models = []
    for m in result.models:
        if m.id == model:
            s = getattr(m.metrics, metric).model_copy(update={"mean": value})
            m = m.model_copy(update={"metrics": m.metrics.model_copy(update={metric: s})})
        models.append(m)
    return result.model_copy(update={"models": models})


def test_shipped_results_meet_the_stage_targets(shipped: dict[str, BenchmarkResult]) -> None:
    bt10 = {"wall_s": 60.2, "cpu": "Apple M4", "cores": 10, "platform": "macOS"}
    status = _status(evaluate(shipped, bt10))
    assert status["BT-6"] == "report-only"
    for bt in ("BT-1", "BT-2", "BT-3", "BT-4", "BT-5", "BT-7", "BT-8", "BT-9", "BT-10"):
        assert status[bt] == "met", bt
    assert set(status.values()) <= set(STATUSES)


def test_measured_values_are_spelled_out(shipped: dict[str, BenchmarkResult]) -> None:
    t = {x.id: x for x in evaluate(shipped, None)}
    assert t["BT-1"].measured.endswith("pp")
    assert "1,420×" in t["BT-7"].measured
    assert t["BT-2"].measured.count("10/10") == 5
    assert t["BT-8"].measured.count("(core, ceil(8/2) = 4), 0 two-qubit gates") == 4
    assert "protein_angles: 10 qubits (geometry, ADR 0005 limit 10), 0 two-qubit gates" in t["BT-8"].measured
    assert t["BT-10"].status == "not-measured"


def _with_qubits(result: BenchmarkResult, qubits: int) -> BenchmarkResult:
    """A copy of ``result`` where Qlassifier's cost row has ``qubits`` qubits."""
    assert result.cost is not None
    models = [m.model_copy(update={"qubits": qubits}) if m.model == "qlassifier" else m for m in result.cost.models]
    return result.model_copy(update={"cost": result.cost.model_copy(update={"models": models})})


def test_bt8_geometry_rule_comes_from_the_encoding_plan(shipped: dict[str, BenchmarkResult]) -> None:
    """ADR 0005: 1 qubit per circle, 1 per (lat, lon) pair, ceil(d_plain / 2) for the plain columns."""
    assert geometry_qubit_limit(Encoding.plan(10, dict.fromkeys(range(10), 360.0), None)) == 10
    assert geometry_qubit_limit(Encoding.plan(8, None, None)) == 4
    assert geometry_qubit_limit(Encoding.plan(5, {0: 360.0, (1, 2): "sphere"}, None)) == 3  # circle + sphere + 1 pair
    protein = shipped["protein_angles"]
    only = {"protein_angles": protein}
    assert _status(evaluate(only))["BT-8"] == "met"  # 10 qubits for 10 circles: the ADR 0005 limit, above ceil(10/2)
    assert _status(evaluate({"protein_angles": _with_qubits(protein, 5)}))["BT-8"] == "met"
    too_many = evaluate({"protein_angles": _with_qubits(protein, 11)})
    t = {x.id: x for x in too_many}["BT-8"]
    assert t.status == "missed"
    assert "11 qubits (geometry, ADR 0005 limit 10)" in t.measured
    # a missed core dataset (5 qubits for d = 8) fails the target whatever the geometry dataset does
    bc = _with_qubits(shipped["breast_cancer"], 5)
    assert _status(evaluate({"breast_cancer": bc, "protein_angles": protein}))["BT-8"] == "missed"


def test_missed_branches(shipped: dict[str, BenchmarkResult]) -> None:
    bc = shipped["breast_cancer"]
    worse = dict(shipped, breast_cancer=_with_mean(bc, "qlassifier", "accuracy", 0.70))
    assert _status(evaluate(worse))["BT-1"] == "missed"
    low = {ds: _with_mean(r, "qlassifier", "balanced_accuracy", 0.5) for ds, r in shipped.items()}
    status = _status(evaluate(low))
    assert (status["BT-3"], status["BT-4"], status["BT-5"]) == ("missed", "missed", "missed")
    zz_better = dict(shipped)
    comps = [
        c.model_copy(
            update={
                "metrics": c.metrics.model_copy(
                    update={"balanced_accuracy": c.metrics.balanced_accuracy.model_copy(update={"win_rate": 0.5})}
                )
            }
        )
        for c in bc.comparisons
    ]
    zz_better["breast_cancer"] = bc.model_copy(update={"comparisons": comps})
    assert _status(evaluate(zz_better))["BT-2"] == "missed"
    assert _status(evaluate(shipped, {"wall_s": 130.0}))["BT-10"] == "missed"
    assert bc.noise is not None
    noisy = [s.model_copy(update={"drop_mean_pp": 3.0}) if s.part == "A" else s for s in bc.noise.summary]
    loud = dict(shipped, breast_cancer=bc.model_copy(update={"noise": bc.noise.model_copy(update={"summary": noisy})}))
    assert _status(evaluate(loud))["BT-9"] == "missed"
    assert bc.cost is not None
    gates = [
        m.model_copy(update={"two_qubit_gates": dict.fromkeys(m.two_qubit_gates, 9)}) if m.model == "qlassifier" else m
        for m in bc.cost.models
    ]
    entangled = dict(
        shipped, breast_cancer=bc.model_copy(update={"cost": bc.cost.model_copy(update={"models": gates})})
    )
    assert _status(evaluate(entangled))["BT-8"] == "missed"
    cheap = bc.model_copy(
        update={"highlights": bc.highlights.model_copy(update={"circuit_ratio_vs_fqk_pipeline": 50.0})}
    )
    assert _status(evaluate(dict(shipped, breast_cancer=cheap)))["BT-7"] == "missed"


def test_not_measured_without_inputs(bt10_result: BenchmarkResult) -> None:
    status = _status(evaluate({"breast_cancer": bt10_result}, None))
    assert status["BT-1"] == "met"
    assert status["BT-7"] == "met"
    assert status["BT-8"] == "met"  # core encoding only: the PRD rule applies as written
    for bt in ("BT-2", "BT-3", "BT-4", "BT-5", "BT-9", "BT-10"):
        assert status[bt] == "not-measured", bt
    assert _status(evaluate({}, None)) == {
        "BT-1": "not-measured", "BT-2": "not-measured", "BT-3": "not-measured", "BT-4": "not-measured",
        "BT-5": "not-measured", "BT-6": "report-only", "BT-7": "not-measured", "BT-8": "not-measured",
        "BT-9": "not-measured", "BT-10": "not-measured",
    }  # fmt: skip


def test_write_report_and_bt10_sources(bt10_result: BenchmarkResult, tmp_path: Path) -> None:
    results = {"breast_cancer": bt10_result}
    own = bt10_from(tmp_path, results, {"breast_cancer": 61.0}, defaults=True, load_avg=(1.25, 2.0, 3.0))
    assert own is not None
    assert own["wall_s"] == 61.0
    assert own["load_avg_at_start"] == [1.25, 2.0, 3.0]
    assert bt10_from(tmp_path, results, {"breast_cancer": 61.0}, defaults=False) is None
    (tmp_path / "bt10").mkdir()
    (tmp_path / "bt10" / "environment.json").write_text(
        json.dumps({"wall_s": {"breast_cancer": 58.5}, "cpu": "Apple M4", "cores": 10, "platform": "macOS"})
    )
    other = bt10_from(tmp_path, results, {"breast_cancer": 300.0}, defaults=False)
    assert other is not None
    assert other["wall_s"] == 58.5
    report = write_report(
        tmp_path, results, "qlassifier benchmark breast_cancer", {"breast_cancer": 61.0}, 10, own, (1.25, 2.0, 3.0)
    )
    text = report.read_text()
    assert "| BT-10 | 61.0 s on" in text
    assert "1-min load 1.25 at start" in text
    assert "Rerun: `qlassifier benchmark breast_cancer`" in text
    # a newcomer reads the headline and where the leaderboards are before any target code (UAT)
    head = text.split("Rerun:")[0]
    assert "## Read this first" in head
    assert "breast_cancer: Qlassifier BA 0." in head
    assert "vs the standard quantum baseline (ZZ QSVC)" in head
    assert "vs the best classical model (mlp: a statistical tie)" in head
    assert "[summary.md](summary.md)" in head
    # targets this one-dataset run cannot measure say what they need instead of a bare status
    assert "| **not measured** in this run: needs all 5 datasets |" in text  # BT-2
    assert "| **not measured** in this run: needs `breast_cancer --noise` |" in text  # BT-9
    assert "**not-measured**" not in text
    targets = json.loads((tmp_path / "targets.json").read_text())
    assert [t["id"] for t in targets] == [f"BT-{i}" for i in range(1, 11)]
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["wall_s"] == {"breast_cancer": 61.0}
    assert env["load_avg_at_start"] == [1.25, 2.0, 3.0]
    assert env["versions"]["qlassifier"] == bt10_result.versions.qlassifier
    summary = (tmp_path / "summary.md").read_text()
    assert "## Breast Cancer (Wisconsin) (`breast_cancer`)" in summary
    assert "**Qlassifier**" in summary
    assert "| Grid edge |" in summary
    trivial = next(line for line in summary.splitlines() if line.startswith("| Majority class |"))
    assert trivial.endswith("| — |")  # untuned models have no grid edge


def test_report_rerun_line_lists_every_command_of_a_two_run_folder(
    bt10_result: BenchmarkResult, tmp_path: Path
) -> None:
    """A folder filled by two runs (benchmarks README) names both commands with their datasets."""
    results = {"breast_cancer": bt10_result, "ionosphere": bt10_result}
    saved = {"ionosphere": {"wall_s": 81.2, "n_jobs": 3, "command": "qlassifier benchmark ionosphere --jobs 3"}}
    report = write_report(
        tmp_path, results, "qlassifier benchmark breast_cancer", {"breast_cancer": 61.0, "ionosphere": 81.2}, 10,
        None, None, saved,
    )  # fmt: skip
    text = report.read_text()
    assert (
        "Rerun: `qlassifier benchmark breast_cancer` (breast_cancer) and `qlassifier benchmark ionosphere --jobs 3` "
        "(ionosphere) ·" in text
    )
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["per_dataset"]["ionosphere"]["command"] == "qlassifier benchmark ionosphere --jobs 3"
    assert env["per_dataset"]["breast_cancer"]["command"] == "qlassifier benchmark breast_cancer"


def test_fingerprint_ignores_ulp_differences_in_grid_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPUs differ by an ulp in numpy reductions and libm pow (logspace): the protocol hash must not."""
    import numpy as np

    from qlassifier.bench import protocol
    from qlassifier.bench.models import model_specs

    base = protocol.fingerprint("heart", [0, 1], ["logistic_regression", "fqk_bandwidth_full", "pqk"], "stratified")
    original = model_specs

    def perturbed(d: int, geometry: Any = None) -> Any:
        specs = []
        for spec in original(d, geometry):
            grid = tuple(
                {k: (np.nextafter(v, np.inf) if isinstance(v, float) else v) for k, v in cfg.items()}
                for cfg in spec.grid
            )
            specs.append(spec.__class__(**{**spec.__dict__, "grid": grid}))
        return specs

    monkeypatch.setattr(protocol, "model_specs", perturbed)
    assert (
        protocol.fingerprint("heart", [0, 1], ["logistic_regression", "fqk_bandwidth_full", "pqk"], "stratified")
        == base
    )
    assert protocol.fingerprint("heart", [0, 1, 2], ["logistic_regression"], "stratified") != base
    assert protocol.fingerprint("heart", [0, 1], ["logistic_regression"], "grouped") != base
