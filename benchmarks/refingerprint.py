"""Rewrite ``protocol.fingerprint`` of the saved and shipped results after a fingerprint-definition change.

Up: README.md · Code: src/qlassifier/bench/protocol.py (`fingerprint`)
Only the fingerprint string changes; every number stays as measured. Run from the repo root:
  .venv/bin/python benchmarks/refingerprint.py
"""

from __future__ import annotations

from pathlib import Path

from qlassifier.bench.protocol import fingerprint
from qlassifier.bench.result import RESULTS_DIR, BenchmarkResult
from qlassifier.errors import DATASET_IDS

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Rewrite the fingerprint of every saved and shipped result whose definition changed."""
    for ds in DATASET_IDS:
        for path in (HERE / ds / "result.json", RESULTS_DIR / f"{ds}.json", HERE / "bt10" / ds / "result.json"):
            if not path.is_file():
                continue
            result = BenchmarkResult.from_json(path)
            new = fingerprint(ds, result.protocol.seeds, result.protocol.models, result.protocol.split)
            if new == result.protocol.fingerprint:
                print(f"{path}: unchanged")
                continue
            updated = result.model_copy(update={"protocol": result.protocol.model_copy(update={"fingerprint": new})})
            assert updated.runs == result.runs
            assert updated.noise == result.noise
            assert updated.cost == result.cost
            updated.to_json(path)
            print(f"{path}: {result.protocol.fingerprint[:8]} -> {new[:8]}")


if __name__ == "__main__":
    main()
