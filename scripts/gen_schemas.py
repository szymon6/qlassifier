"""Generate the API contract JSON Schemas from the pydantic models (TDD §6.7).

Usage:
  python scripts/gen_schemas.py          # write src/qlassifier/schemas/*.json
  python scripts/gen_schemas.py --check  # exit 1 if the committed files differ from the models

One file per top-level contract model (endpoint responses, the run request, SSE event data) plus ``api.json``,
a bundle with every model once under ``$defs`` (use it for TypeScript generation: no duplicate types).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from qlassifier.bench.result import BenchmarkResult
from qlassifier.server.models import (
    DatasetList,
    ErrorResponse,
    FailedData,
    Meta,
    PartialData,
    ProgressData,
    Run,
    RunRequest,
    TerminalData,
)

OUT = Path(__file__).resolve().parent.parent / "src" / "qlassifier" / "schemas"
Mode = Literal["validation", "serialization"]
MODELS: list[tuple[str, type[BaseModel], Mode]] = [
    ("meta", Meta, "serialization"),
    ("dataset-list", DatasetList, "serialization"),
    ("benchmark-result", BenchmarkResult, "serialization"),
    ("run", Run, "serialization"),
    ("run-request", RunRequest, "validation"),
    ("error-response", ErrorResponse, "serialization"),
    ("progress-data", ProgressData, "serialization"),
    ("partial-data", PartialData, "serialization"),
    ("terminal-data", TerminalData, "serialization"),
    ("failed-data", FailedData, "serialization"),
]


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def build() -> dict[str, str]:
    files: dict[str, str] = {}
    for name, model, mode in MODELS:
        schema = model.model_json_schema(mode=mode)
        schema["$id"] = f"qlassifier/api/v1/{name}.json"
        files[f"{name}.json"] = render(schema)
    _, bundle = models_json_schema([(m, mode) for _, m, mode in MODELS], title="Qlassifier API v1")
    bundle["$id"] = "qlassifier/api/v1/api.json"
    bundle["description"] = (
        "Every contract model once under $defs. Endpoints: GET /api/v1/meta -> Meta; GET /api/v1/datasets -> "
        "DatasetList; GET /api/v1/results/{dataset} -> BenchmarkResult; POST /api/v1/runs (RunRequest) -> Run; "
        "GET /api/v1/runs/{id} -> Run; GET /api/v1/runs/{id}/events -> SSE (progress: ProgressData, partial: "
        "PartialData, complete / cancelled: TerminalData, failed: FailedData); GET /api/v1/runs/{id}/result -> "
        "BenchmarkResult; POST /api/v1/runs/{id}/cancel -> Run; errors -> ErrorResponse."
    )
    files["api.json"] = render(bundle)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the committed schemas instead of writing them")
    args = ap.parse_args()
    files = build()
    if args.check:
        stale = [n for n, text in files.items() if not (OUT / n).is_file() or (OUT / n).read_text() != text]
        extra = sorted(p.name for p in OUT.glob("*.json") if p.name not in files)
        if stale or extra:
            print(
                f"schemas out of date: {', '.join(stale + extra)}. Run: python scripts/gen_schemas.py",
                file=sys.stderr,
            )
            return 1
        print(f"schemas in sync ({len(files)} files)")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        if old.name not in files:
            old.unlink()
    for name, text in files.items():
        (OUT / name).write_text(text)
    print(f"wrote {len(files)} schemas to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
