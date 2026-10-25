"""Default worker count: every core, capped by RAM during the noise study (TDD §7.11; post-launch review)."""

from __future__ import annotations

import os

import pytest

from qlassifier.bench import NOISE_RESERVE_GIB, NOISE_WORKER_GIB, _resolve_jobs, default_jobs
from qlassifier.cli import _jobs_note
from qlassifier.errors import InvalidParameterError

GIB = 2**30


@pytest.mark.parametrize(
    ("noise", "cores", "ram_gib", "expected"),
    [
        (False, 10, 16, 10),  # exact runs are light: every core
        (True, 10, 16, 3),  # the 16 GB laptop that panicked twice with 10 workers
        (True, 10, 32, 7),
        (True, 10, 64, 10),  # never above the core count
        (True, 10, 8, 1),  # never below 1
        (True, 2, 16, 2),
    ],
)
def test_default_jobs(noise: bool, cores: int, ram_gib: int, expected: int) -> None:
    assert default_jobs(noise, cores=cores, memory_bytes=ram_gib * GIB) == expected


def test_cap_formula_matches_the_documented_budget() -> None:
    assert (NOISE_WORKER_GIB, NOISE_RESERVE_GIB) == (4, 4)
    assert default_jobs(True, cores=10, memory_bytes=16 * GIB) == (16 - NOISE_RESERVE_GIB) // NOISE_WORKER_GIB


def test_unknown_memory_means_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import qlassifier.bench as bench

    monkeypatch.setattr(bench, "_physical_memory", lambda: None)
    assert default_jobs(True, cores=6) == 6
    assert default_jobs(False) == (os.cpu_count() or 1)


def test_machine_memory_is_read() -> None:
    cores = os.cpu_count() or 1
    assert 1 <= default_jobs(True) <= cores
    assert default_jobs(False) == cores


def test_resolve_jobs_uses_the_default_only_when_unset() -> None:
    assert _resolve_jobs(None, noise=False) == (os.cpu_count() or 1)
    assert _resolve_jobs(-1, noise=True) == default_jobs(True)
    assert _resolve_jobs(2, noise=True) == 2
    with pytest.raises(InvalidParameterError):
        _resolve_jobs(0, noise=True)


def test_plan_note_only_when_the_cap_bites() -> None:
    cores = os.cpu_count() or 1
    assert _jobs_note(None, False, cores) == ""
    assert _jobs_note(3, True, 3) == ""  # the user chose 3
    assert _jobs_note(None, True, cores) == ""
    if cores > 1:
        assert "--jobs overrides" in _jobs_note(None, True, cores - 1)
