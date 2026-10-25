"""Package surface (TDD §5, §10.4): lazy heavy imports, public names, version."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import qlassifier


def _modules_after(code: str) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-c", f"{code}\nimport sys, json; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_import_loads_no_qiskit_or_matplotlib() -> None:
    mods = _modules_after("import qlassifier")
    assert not any(m == "qiskit" or m.startswith("qiskit") for m in mods)
    assert "matplotlib" not in mods
    assert "fastapi" not in mods


def test_exact_mode_fit_needs_no_qiskit() -> None:
    mods = _modules_after(
        "from qlassifier import Qlassifier, load_dataset\n"
        "X, y = load_dataset('heart', return_X_y=True)\n"
        "Qlassifier(gamma=1.0, C=1.0).fit(X, y).predict(X)"
    )
    assert "qiskit" not in mods


def test_public_names_and_lazy_benchmark_api() -> None:
    for name in qlassifier.__all__:
        assert getattr(qlassifier, name) is not None
    from qlassifier.bench import ProgressEvent, benchmark
    from qlassifier.bench.result import BenchmarkResult

    assert qlassifier.benchmark is benchmark
    assert qlassifier.ProgressEvent is ProgressEvent
    assert qlassifier.BenchmarkResult is BenchmarkResult
    with pytest.raises(AttributeError):
        _ = qlassifier.does_not_exist  # type: ignore[attr-defined]


def test_version_from_metadata() -> None:
    assert qlassifier.__version__ == "0.1.0"
