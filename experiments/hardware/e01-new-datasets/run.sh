#!/bin/bash
# e01: BT-1 on three new UCI sets, eval seeds 0-9, full model set + cost study (shipped engine via run_bench.py).
# Up: ../README.md §2.2 · run from the repo root under the watchdog:
#   .cache/run_guarded.sh stage2-e01 bash experiments/hardware/e01-new-datasets/run.sh
set -eu
E=experiments/hardware
export PYTHONPATH=$E/common OMP_NUM_THREADS=1
for ds in spambase banknote haberman; do
  echo "== $ds $(date -u '+%FT%TZ')"
  /usr/bin/time -l .venv/bin/python $E/common/run_bench.py $ds --seeds 0-9 --out $E/e01-new-datasets/$ds --jobs 10 2>&1 | grep -v "Warning\|warn("
done
echo "== done $(date -u '+%FT%TZ')"
