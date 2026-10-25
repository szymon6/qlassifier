"""Noise study cells on real rows (TDD §7.8, ADR 0006): part A, part B on a tiny paired subset, aggregation."""

from __future__ import annotations

import pytest

from qlassifier.bench import noise
from qlassifier.bench.protocol import Task
from qlassifier.bench.result import NoiseRecord


@pytest.fixture
def tiny_part_b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part B on a 20 / 10 subset of the Breast Cancer split and 256 shots (the protocol: 60 / 40, 1,024 shots)."""
    monkeypatch.setattr(noise, "PART_B_TRAIN", 20)
    monkeypatch.setattr(noise, "PART_B_TEST", 10)
    monkeypatch.setattr(noise, "SHOTS", 256)


def test_task_lists() -> None:
    bc = noise.noise_tasks("breast_cancer", range(10))
    assert len([t for t in bc if t.part == "A"]) == 5 * 4
    assert len([t for t in bc if t.part == "B"]) == 3 * 3 * 4
    assert {t.seed for t in bc if t.part == "B"} == {0, 1, 2}
    heart = noise.noise_tasks("heart", [4, 5, 6, 7, 8, 9])
    assert {t.part for t in heart} == {"A"}
    assert {t.seed for t in heart} == {4, 5, 6, 7, 8}
    assert {t.device for t in heart} == {"shots", "fake_torino", "fake_fez", "fake_brisbane"}


def test_part_a_cell() -> None:
    rec = noise.run_noise_task(Task("noise", "heart", 0, "qlassifier", "fake_brisbane", "A"), "stratified")
    assert isinstance(rec, NoiseRecord)
    assert (rec.part, rec.model, rec.device, rec.seed) == ("A", "qlassifier", "fake_brisbane", 0)
    assert (rec.n_train, rec.n_test) == (207, 90)
    assert rec.circuits == 3 * 297
    assert rec.two_qubit_gates == 0
    assert rec.kernel_r is not None
    assert rec.kernel_r > 0.99
    assert rec.drop_pp == pytest.approx(100 * (rec.exact_accuracy - rec.accuracy))
    assert rec.exact_balanced_accuracy > 0.75


@pytest.mark.usefixtures("tiny_part_b")
@pytest.mark.parametrize("model", ["qlassifier", "fqk_bandwidth_linear", "pqk", "separable"])
def test_part_b_cells(model: str) -> None:
    rec = noise.run_noise_task(Task("noise", "breast_cancer", 0, model, "fake_torino", "B"), "stratified")
    assert (rec.part, rec.model, rec.n_train, rec.n_test) == ("B", model, 20, 10)
    fidelity = model in ("fqk_bandwidth_linear", "separable")
    assert rec.circuits == (20 * 19 // 2 + 10 * 20 if fidelity else 3 * 30)
    assert rec.two_qubit_gates == (0 if model in ("qlassifier", "separable") else rec.two_qubit_gates)
    if model in ("fqk_bandwidth_linear", "pqk"):
        assert rec.two_qubit_gates > 0
    assert 0.0 <= rec.accuracy <= 1.0
    assert rec.kernel_r is not None


def test_subset_is_stratified_and_paired() -> None:
    Xr, yr, Xs, ys = noise.part_b_subset(1)
    assert (len(Xr), len(Xs)) == (60, 40)
    assert yr.sum() in (22, 23)
    assert ys.sum() in (15, 16)
    again = noise.part_b_subset(1)
    assert (again[0] == Xr).all()


def test_study_aggregation() -> None:
    def rec(part: str, model: str, device: str, seed: int, acc: float, exact: float) -> NoiseRecord:
        return NoiseRecord(
            part=part, model=model, device=device, seed=seed, accuracy=acc, balanced_accuracy=acc,  # type: ignore[arg-type]
            exact_accuracy=exact, exact_balanced_accuracy=exact, drop_pp=100 * (exact - acc), kernel_r=0.99,
            circuits=10, two_qubit_gates=0, depth=5, n_train=60, n_test=40, seconds=1.0,
        )  # fmt: skip

    records = [
        rec("A", "qlassifier", "fake_fez", 0, 0.94, 0.95),
        rec("A", "qlassifier", "fake_fez", 1, 0.95, 0.95),
        rec("A", "qlassifier", "fake_torino", 0, 0.93, 0.95),
        rec("A", "qlassifier", "shots", 0, 0.96, 0.95),
        rec("B", "fqk_bandwidth_linear", "fake_fez", 0, 0.70, 0.90),
    ]
    study = noise.noise_study("breast_cancer", range(10), records)
    assert study is not None
    assert [p.id for p in study.parts] == ["A", "B"]
    rows = {(s.part, s.model, s.device): s for s in study.summary}
    assert rows[("A", "qlassifier", "exact")].accuracy_mean == pytest.approx(0.95)
    assert rows[("A", "qlassifier", "fake_fez")].drop_mean_pp == pytest.approx(0.5)
    assert rows[("A", "qlassifier", "fake_fez")].drop_worst_pp == pytest.approx(1.0)
    assert rows[("A", "qlassifier", "shots")].drop_mean_pp == pytest.approx(-1.0)
    assert rows[("B", "fqk_bandwidth_linear", "fake_fez")].drop_mean_pp == pytest.approx(20.0)
    assert noise.max_device_drop(study) == pytest.approx(2.0)  # Torino, part A
    assert noise.max_device_drop(None) is None
    assert noise.noise_study("heart", [0, 1], []) is None
