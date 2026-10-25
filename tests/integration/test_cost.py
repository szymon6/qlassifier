"""Cost study (TDD §7.9, BT-7, BT-8): transpiled resources and circuit counts on Breast Cancer (phase-7 e06)."""

from __future__ import annotations

import numpy as np
import pytest

from qlassifier.bench.cost import QUANTUM_MODELS, cost_study, fqk_pipeline_ratio, representative_circuits
from qlassifier.bench.result import CostStudy


@pytest.fixture(scope="module")
def bc_cost() -> CostStudy:
    return cost_study("breast_cancer", 0, "stratified", list(QUANTUM_MODELS))


def test_circuit_counts_match_e06(bc_cost: CostStudy) -> None:
    by = {m.model: m for m in bc_cost.models}
    assert (bc_cost.n_train, bc_cost.n_test) == (398, 171)
    assert by["qlassifier"].circuits_fit_predict == 1_707
    assert by["pqk"].circuits_fit_predict == 1_707
    for fidelity in ("zz_qsvc", "fqk_bandwidth_full", "fqk_bandwidth_linear", "separable"):
        assert by[fidelity].circuits_fit_predict == 147_061
    assert by["fqk_bandwidth_linear"].tuning_circuits_on_device == 2_276_238
    assert by["separable"].tuning_circuits_on_device == 2_276_238
    assert by["zz_qsvc"].tuning_circuits_on_device == 0
    assert by["qlassifier"].tuning_circuits_on_device == 0
    assert fqk_pipeline_ratio(bc_cost) == pytest.approx(2_423_299 / 1_707)
    assert round(fqk_pipeline_ratio(bc_cost) or 0) == 1_420


def test_transpiled_resources(bc_cost: CostStudy) -> None:
    by = {m.model: m for m in bc_cost.models}
    assert bc_cost.devices == ["fake_torino", "fake_fez", "fake_brisbane"]
    q = by["qlassifier"]
    assert q.qubits == 4
    assert q.two_qubit_gates == {"fake_torino": 0, "fake_fez": 0, "fake_brisbane": 0}
    assert max(q.depth.values()) <= 8
    assert by["separable"].two_qubit_gates["fake_fez"] == 0
    assert by["fqk_bandwidth_linear"].two_qubit_gates == {"fake_torino": 54, "fake_fez": 54, "fake_brisbane": 54}
    assert by["pqk"].qubits == 9
    assert by["pqk"].two_qubit_gates["fake_torino"] == 72
    assert by["zz_qsvc"].two_qubit_gates["fake_torino"] > 300


def test_ratio_needs_both_models(bc_cost: CostStudy) -> None:
    assert fqk_pipeline_ratio(None) is None
    only_q = bc_cost.model_copy(update={"models": [m for m in bc_cost.models if m.model == "qlassifier"]})
    assert fqk_pipeline_ratio(only_q) is None
    subset = cost_study("heart", 0, "stratified", ["qlassifier"])
    assert [m.model for m in subset.models] == ["qlassifier"]
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError):
        representative_circuits("svm", 8, None, rng)
