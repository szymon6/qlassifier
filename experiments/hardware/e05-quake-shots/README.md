# e05 — do more shots close the `deep_quakes` device loss? (H6)

Up: [../README.md](../README.md) (protocol §2.2, §2.7) · Log: [../learning-log.md](../learning-log.md) (H6) · Code: [`run.py`](run.py) · Raw: [`results.csv`](results.csv), [`reference_e07.json`](reference_e07.json) · Stage 1: [e07](../../classifier/e07-structure-fit/README.md) (exact loss to RF), [e08](../../classifier/e08-geo-noise/README.md) (−8 pp at 1,024 shots), theory §7

## What
Stage 1 recorded that geometry mode loses to random forest on global deep-vs-shallow earthquakes (−6.6 pp BA exact) and loses another ≈ 8 pp under 1,024-shot device noise, with the hypothesis "narrow kernels need angle resolution, i.e. shots". Test: the shipped `Qlassifier(geometry={(lat, lon): "sphere"}, backend="fake_fez", shots=S)` at S ∈ {1,024; 4,096; 16,384; 65,536} on fresh seeds 200–204 (the e07 seeds, so the RF reference pairs by seed). Exact-mode self-tuning on the training split (0 circuits); fit and predict both read measured Bloch vectors (1 qubit, 3 settings, 0 two-qubit gates).

## How
`PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python experiments/hardware/e05-quake-shots/run.py` (under `.cache/run_guarded.sh stage2-e05`). Splits: stage-1 protocol, recipe `native`, stratified 70/30 (3,729 rows → 2,610 / 1,119).

## Result (fresh seeds 200–204; 1 sphere qubit, 0 two-qubit gates; wall 13 s → 280 s per fit + predict as shots grow)
| Backend / shots | Accuracy | BA | BA drop vs exact (mean ± sd over seeds) |
|---|---|---|---|
| exact | 0.881 ± 0.004 | 0.766 ± 0.008 | — |
| FakeFez, 1,024 | 0.832 ± 0.009 | 0.663 ± 0.022 | 10.35 ± 1.88 pp |
| FakeFez, 4,096 | 0.854 ± 0.006 | 0.709 ± 0.009 | 5.76 ± 0.57 pp |
| FakeFez, 16,384 | 0.872 ± 0.004 | 0.748 ± 0.012 | 1.84 ± 0.78 pp |
| FakeFez, 65,536 | 0.878 ± 0.003 | 0.760 ± 0.010 | **0.65 ± 0.66 pp** |
| Random forest, exact (e07, same seeds) | — | 0.831 | — (geometry exact − RF = −6.5 pp) |
- Tuned configuration on every seed: g = 16 (the narrowest grid corner), C = 62.5; the shipped `ShotNoiseWarning` fires at 1,024 shots on this data (predicted CV loss > 2 pp), as designed in stage 1.
- The shipped estimator's exact BA (0.766) equals the stage-1 harness value on the same seeds (e07: 0.766): same splits, same tuning.
- Per seed the curve is monotone; the drop halves roughly every 4× shots (consistent with a 1/√S angle resolution: 1.8° at 1,024 shots, 0.2° at 65,536).

## Verdict
- **H6 confirmed, not killed:** the device loss on `deep_quakes` is shot resolution: 10.4 pp at 1,024 shots, 0.65 pp at 65,536 shots (within the 2 pp kill bound). It is a shots problem, not a device-error problem (0 two-qubit gates; FakeFez gate and readout errors are the stage-1 ones).
- The loss to random forest is unchanged (−6.5 pp exact, −7.1 pp at 65,536 shots): shots cannot buy what the kernel lacks (local density adaptation). `deep_quakes` stays a teaching example, not a showcase; BT-5 does not move.
- Product rule for phase 8: when the tuned width sits on the narrow grid corner, the cost view should show the shots needed (here ≥ 16k for ≤ 2 pp) next to the QPU-time bracket (16,384 shots × 3 × 3,729 rows ≈ 15 h of QPU at 250 µs rep delay: not an Open-plan job).
