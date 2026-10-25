"""Run the shipped benchmark engine (protocol, grids, statistics, output files) on a registered dataset.

Up: ../README.md (protocol §2) · Registry: extra_datasets.py

Usage (from the repo root, PYTHONPATH on this folder so loky workers can import the registry):
  PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python \
      experiments/hardware/common/run_bench.py spambase --seeds 0-9 --out <dir> [--jobs 10]
      [--models m1 m2 ...] [--no-cost]
Writes result.json, runs.csv, plots/ (as `qlassifier benchmark`) plus summary.txt; appends finished rows to
runs_partial.csv as they come in (a crash keeps them).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_datasets  # noqa: E402,F401  (registers on import)
from joblib import Parallel, delayed, parallel_config  # noqa: E402

from qlassifier.bench import _DEFAULT_WEIGHTS, _execute  # noqa: E402
from qlassifier.bench.aggregate import build_result  # noqa: E402
from qlassifier.bench.cost import QUANTUM_MODELS  # noqa: E402
from qlassifier.bench.protocol import Task, arrays, decide_split, fingerprint, protocol_models, test_train_ratio  # noqa: E402
from qlassifier.datasets import _info  # noqa: E402


def _seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def _run(task: Task, mode: str, models: tuple[str, ...]):  # executed in workers
    import extra_datasets  # noqa: F401  (registry in this process)

    return _execute(task, mode, models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(a.seeds)
    specs = protocol_models(a.dataset, a.models)
    ids = tuple(s.id for s in specs)
    mode = decide_split(a.dataset, seeds)
    ratio = test_train_ratio(a.dataset, seeds, mode)
    fp = fingerprint(a.dataset, seeds, ids, mode)
    info = _info(a.dataset)
    tasks = [Task("run", a.dataset, s, m, weight=_DEFAULT_WEIGHTS[m]) for s in seeds for m in ids]
    if not a.no_cost and any(m in QUANTUM_MODELS for m in ids):
        tasks.append(Task("cost", a.dataset, seeds[0], "cost", weight=_DEFAULT_WEIGHTS["cost"]))
    tasks.sort(key=lambda t: -t.weight)
    print(f"{a.dataset}: {len(tasks)} tasks, split {mode}, n_te/n_tr {ratio:.3f}, d_used {arrays(a.dataset).d_used}, jobs {a.jobs}", flush=True)
    t0 = time.perf_counter()
    runs, cost = [], None
    partial = out / "runs_partial.csv"
    with parallel_config(backend="loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(_run)(t, mode, ids) for t in tasks)
        for i, (task, res) in enumerate(gen, 1):
            if task.kind == "run":
                runs.append(res)
                with partial.open("a") as f:
                    f.write(json.dumps(res.model_dump(), default=str) + "\n")
            else:
                cost = res
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    result = build_result(
        info=info, seeds=seeds, specs=specs, mode=mode, ratio=ratio, fingerprint=fp,
        n_features_used=arrays(a.dataset).d_used, runs=runs, cost=cost, noise_records=[], status="complete",
        wall_s=time.perf_counter() - t0, n_jobs=a.jobs,
    )  # fmt: skip
    result.save(out)
    (out / "summary.txt").write_text(result.summary() + "\n" + result.summary("accuracy") + "\n")
    print(result.summary())
    print(f"wall {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
