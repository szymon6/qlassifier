#!/bin/bash
# e06: BT-6 regression check: shipped `qlassifier benchmark` (exact, seeds 0-9) on current main vs the stage-1 shipped rows.
# Up: ../README.md §2.6 ·   .cache/run_guarded.sh stage2-e06 bash experiments/hardware/e06-regression/run.sh
set -eu
E=experiments/hardware
export OMP_NUM_THREADS=1
for ds in breast_cancer heart; do
  echo "== $ds $(date -u '+%FT%TZ')"
  .venv/bin/qlassifier benchmark $ds --out .cache/stage2-e06 --jobs 5 2>&1 | tail -3
done
.venv/bin/python $E/e06-regression/compare.py
echo "== done $(date -u '+%FT%TZ')"
