"""scripts/ship_results.py (TDD §7.12): only a complete protocol run with noise and cost studies is shipped."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import ROOT


def _ship() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ship_results", ROOT / "scripts" / "ship_results.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_flags_runs_that_are_not_the_full_protocol(heart_result: Any) -> None:
    problems = _ship().check(heart_result, "heart")
    assert "seeds [0, 1]" in problems
    assert _ship().check(heart_result.model_copy(update={"status": "cancelled"}), "sonar") == [
        "dataset id heart", "status cancelled", "seeds [0, 1]",
    ]  # fmt: skip
    no_studies = heart_result.model_copy(update={"noise": None, "cost": None})
    assert {"no noise study", "no cost study"} <= set(_ship().check(no_studies, "heart"))


def test_main_refuses_missing_or_partial_runs(
    heart_result: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ship = _ship()
    monkeypatch.setattr("sys.argv", ["ship_results.py", str(tmp_path)])
    assert ship.main() == 1
    assert "missing" in capsys.readouterr().err
    for ds in ("breast_cancer", "ionosphere", "sonar", "heart", "protein_angles"):
        (tmp_path / ds).mkdir()
        heart_result.to_json(tmp_path / ds / "result.json")
    assert ship.main() == 1
    assert "not a full protocol run" in capsys.readouterr().err
