"""API contract (TDD §6.7): committed JSON Schemas equal the pydantic models; shipped results are current."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

import qlassifier
from qlassifier.bench.protocol import decide_split, fingerprint
from qlassifier.bench.result import RESULTS_DIR, BenchmarkResult
from qlassifier.errors import DATASET_IDS
from tests.conftest import ROOT

SCHEMAS = ROOT / "src" / "qlassifier" / "schemas"


def _gen_schemas() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_schemas", ROOT / "scripts" / "gen_schemas.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_schemas_equal_the_models() -> None:
    files = _gen_schemas().build()
    committed = {p.name for p in SCHEMAS.glob("*.json")}
    assert committed == set(files)
    for name, text in files.items():
        assert (SCHEMAS / name).read_text() == text, f"{name} is stale: run python scripts/gen_schemas.py"


def test_bundle_holds_every_contract_model() -> None:
    bundle = json.loads((SCHEMAS / "api.json").read_text())
    for name in ("Meta", "DatasetList", "DatasetCard", "BenchmarkResult", "Run", "RunRequest", "ErrorResponse",
                 "ProgressData", "PartialData", "TerminalData", "FailedData", "NoiseStudy", "CostStudy"):  # fmt: skip
        assert name in bundle["$defs"], name
    jsonschema.Draft202012Validator.check_schema(bundle)


def test_check_mode_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gen = _gen_schemas()
    monkeypatch.setattr(gen, "OUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["gen_schemas.py"])
    assert gen.main() == 0
    monkeypatch.setattr("sys.argv", ["gen_schemas.py", "--check"])
    assert gen.main() == 0
    (tmp_path / "run.json").write_text("{}")
    (tmp_path / "extra.json").write_text("{}")
    assert gen.main() == 1
    assert "run.json" in capsys.readouterr().err


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_shipped_results_are_valid_and_current(dataset: str) -> None:
    path = RESULTS_DIR / f"{dataset}.json"
    assert path.is_file(), "missing shipped result: run scripts/ship_results.py after the full benchmark"
    raw = json.loads(path.read_text())
    jsonschema.validate(raw, json.loads((SCHEMAS / "benchmark-result.json").read_text()))
    result = BenchmarkResult.cached(dataset)
    assert result.source == "shipped"
    assert result.status == "complete"
    assert result.protocol.seeds == list(range(10))
    assert result.noise is not None
    assert result.noise.source == "shipped"
    assert result.cost is not None
    mode = decide_split(dataset, result.protocol.seeds)
    assert result.protocol.split == mode
    assert result.protocol.fingerprint == fingerprint(dataset, result.protocol.seeds, result.protocol.models, mode)
    major_minor = ".".join(qlassifier.__version__.split(".")[:2])
    assert result.versions.qlassifier.startswith(major_minor + ".")
    assert len(result.runs) == 10 * len(result.protocol.models)
