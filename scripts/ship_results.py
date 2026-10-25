"""Ship a full benchmark run as the package's cached results (TDD §6.6, §7.12).

Usage:
  python scripts/ship_results.py [BENCHMARK_DIR]   # default: benchmarks

Copies ``<dir>/<dataset>/result.json`` for every bundled dataset to ``src/qlassifier/_results/<dataset>.json``
with ``source: "shipped"`` (noise study included). Refuses partial or cancelled runs, runs that are not on the
protocol seeds 0-9, and runs without the noise and cost studies: shipped numbers come only from the full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qlassifier.bench.result import RESULTS_DIR, BenchmarkResult
from qlassifier.errors import DATASET_IDS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "benchmarks"


def check(result: BenchmarkResult, dataset: str) -> list[str]:
    problems = []
    if result.dataset.id != dataset:
        problems.append(f"dataset id {result.dataset.id}")
    if result.status != "complete":
        problems.append(f"status {result.status}")
    if result.protocol.seeds != list(range(10)):
        problems.append(f"seeds {result.protocol.seeds}")
    if result.noise is None:
        problems.append("no noise study")
    if result.cost is None:
        problems.append("no cost study")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", nargs="?", type=Path, default=DEFAULT)
    args = ap.parse_args()
    results = {}
    for ds in DATASET_IDS:
        path = args.source / ds / "result.json"
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 1
        result = BenchmarkResult.from_json(path)
        problems = check(result, ds)
        if problems:
            print(f"{path}: not a full protocol run ({'; '.join(problems)})", file=sys.stderr)
            return 1
        results[ds] = result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ds, result in results.items():
        result.as_shipped().to_json(RESULTS_DIR / f"{ds}.json")
        print(
            f"shipped {ds}: {len(result.runs)} runs, {len(result.noise.records) if result.noise else 0} noise records"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
