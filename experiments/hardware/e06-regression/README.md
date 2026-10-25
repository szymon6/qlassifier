# e06 — stage-1 regression check (BT-6)

Up: [../README.md](../README.md) (protocol §2.6) · Log: [../learning-log.md](../learning-log.md) (H7) · Code: [`run.sh`](run.sh), [`compare.py`](compare.py) · Raw: [`comparison.json`](comparison.json) · Reference: [stage-1 benchmarks](../../../benchmarks/REPORT.md)

## What
BT-6 asks that the stage-1 targets still pass. Phase 7 checks the baseline: the shipped `qlassifier benchmark` (exact, seeds 0–9) on the current `main` reproduces the stage-1 shipped rows (chosen config, edge flag, BA, accuracy, ROC-AUC, F1) for `breast_cancer` and `heart`, and the stage-1 target table is re-read. Phase 9 re-runs the full `--noise` command after the stage-2 changes.

## How
```
.cache/run_guarded.sh stage2-e06 bash experiments/hardware/e06-regression/run.sh
```
(`qlassifier benchmark <dataset> --out .cache/stage2-e06`, then `compare.py` against `benchmarks/<dataset>/runs.csv`.)

## Result
| Dataset | Rows compared | Rows within 1e-9 (BA, acc, ROC-AUC, F1) | Max |Δ| | Chosen config equal | Edge flag equal |
|---|---|---|---|---|---|
| `breast_cancer` | 120 / 120 | 120 | 0.0 | 120 | 120 |
| `heart` | 120 / 120 | 120 | 0.0 | 120 | 120 |
- Stage-1 `targets.json` re-read: BT-1…BT-5, BT-7…BT-10 `met`, BT-6 `report-only` (unchanged).
- Run 2026-10-25 09:28–09:32 UTC with `--jobs 5` while e02 and e05 ran; wall time is therefore not a BT-10 measurement (phase 9 re-measures).

## Verdict
H7 confirmed: the shipped engine on current `main` reproduces every stage-1 row bit for bit on the reference platform. BT-6 baseline holds; phase 9 re-runs the full `--noise` command after the stage-2 changes (new datasets, sweep, cost model, hardware path) and compares the ten statuses again.
