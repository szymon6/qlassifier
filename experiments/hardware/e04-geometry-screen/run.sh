#!/bin/bash
# e04: BT-5 screen on dev seeds 100-104 (beijing_pm25, solar_boulder); confirmation on fresh seeds 200-209 only if
# the dev gate (Δ vs best classical >= +1.0 pp) passes (protocol §2.2). Up: ../README.md
#   .cache/run_guarded.sh stage2-e04 bash experiments/hardware/e04-geometry-screen/run.sh [confirm <dataset>]
set -eu
E=experiments/hardware
export PYTHONPATH=$E/common OMP_NUM_THREADS=1
if [ "${1:-screen}" = "confirm" ]; then
  ds=$2; echo "== confirm $ds $(date -u '+%FT%TZ')"
  /usr/bin/time -l .venv/bin/python $E/common/run_bench.py $ds --seeds 200-209 --out $E/e04-geometry-screen/confirm-$ds --jobs 5 --no-cost 2>&1 | grep -v "Warning\|warn("
else
  for ds in beijing_pm25 solar_boulder; do
    echo "== screen $ds $(date -u '+%FT%TZ')"
    /usr/bin/time -l .venv/bin/python $E/common/run_bench.py $ds --seeds 100-104 --out $E/e04-geometry-screen/dev-$ds --jobs 5 --no-cost 2>&1 | grep -v "Warning\|warn("
  done
fi
echo "== done $(date -u '+%FT%TZ')"
