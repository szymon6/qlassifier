# e07 — hardware point on an IBM Heron device (BT-3)

Up: [../README.md](../README.md) (protocol §2.5) · PRD: BT-3, §9, change log · Log: [../learning-log.md](../learning-log.md) · Code: [`hardware_point.py`](hardware_point.py) · Dry run: [`dry-run.log`](dry-run.log), [`plan.json`](plan.json), [`prediction.json`](prediction.json), `dry-run-seed0/` (transpiled circuits on the FakeFez twin) · Run: `run-seed0/` (`record.json`, `counts.json`, `circuit_*.qasm3|qpy`, `parameters_*.csv`, `bloch_*.npy`) · Cost model: [e03](../e03-cost-model/README.md) · Noise twin: [e02](../e02-noise-sweep/README.md)

## What
The first real-device point of Qlassifier: Bloch readout (3 Pauli settings per row, 4 qubits, 0 two-qubit gates) on an IBM Heron r2 device for the Breast Cancer 60 train / 40 test subset of eval seed 0 (the stage-1 noise-study rows). Training, tuning and preprocessing stay on the exact simulator (0 circuits); hardware reads the features. One job, 3 PUBs × 100 parameter sets × 1,024 shots = 300 circuits. Two analyses from the same counts: **variant 1 (inference only)**: exact training features, hardware test features (the product's hardware mode); **variant 2 (train + test readout)**: hardware features for both.

## Pre-registered plan and prediction (written before submission; `dry-run.log`, 2026-10-25)
| Item | Value |
|---|---|
| Rows | 60 train / 40 test (`qlassifier.bench.noise.part_b_subset(0)`), `pca8` fit on the 60 training rows |
| Tuning (exact, 0 circuits) | γ-multiplier g = 0.25, C = 4.0 (`_subset_config`, 30-config ridge grid, 5-fold CV on BA) |
| Circuits | 300 (3 settings × 100 rows); inference-only would be 120 |
| Shots | 1,024 per circuit (307,200 total) |
| Circuit on the FakeFez twin | 4 qubits, physical layout [84, 96, 23, 4], 0 two-qubit gates, depth 10, duration 1.63 µs |
| QPU estimate (e03 model) | **79–110 s** for 300 circuits (33–45 s for inference only); cap 180 s → train + test readout in one job |
| Sampler options | default shots 1,024; dynamical decoupling off; gate and measurement twirling off (the simulation estimator) |
| Device rule | `ibm_fez` (twin of FakeFez) if operational with ≤ 50 pending jobs, else `ibm_kingston`, else least-busy Heron |
| Exact accuracy / BA on this subset | 0.875 / 0.873 (35 of 40 test rows correct) |

Prediction from the fake devices (per-qubit simulation, 20 shot-noise draws each, sampler seeds 11–30):

| Device model | Variant 1 acc (mean ± sd [min, max]) | Variant 1 BA | Variant 2 acc | Variant 2 BA | Kernel r (train) |
|---|---|---|---|---|---|
| FakeFez (Heron r2) | 0.880 ± 0.010 [0.875, 0.900] | 0.880 | 0.880 ± 0.013 | 0.880 | 0.9983 |
| FakeTorino (Heron r1) | 0.879 ± 0.009 [0.875, 0.900] | 0.878 | 0.880 ± 0.013 | 0.880 | 0.9983 |
| FakeBrisbane (Eagle r3) | 0.880 ± 0.010 [0.875, 0.900] | 0.880 | 0.881 ± 0.011 | 0.882 | 0.9983 |

- **Pass rule (PRD BT-3):** hardware accuracy within ± 2 pp of the FakeFez prediction, per variant: 0.880 ± 0.020 → **35 or 36 of 40 test rows correct (0.875 or 0.900)**. 
- **Kill criterion:** ≤ 34 or ≥ 37 correct rows (|Δ| ≥ 2.5 pp) in variant 1 → BT-3 missed; recorded with the calibration snapshot and kernel r. Variant 2 is reported alongside (the PRD's optional training readout).
- Resolution flag: with 40 test rows one row is 2.5 pp, so the ± 2 pp rule is a one-row tolerance; the shot-noise prediction itself spans [0.875, 0.900]. Also reported: kernel r vs exact (prediction 0.998), mean |Bloch error| (prediction ≈ 0.018), BA.

## How
```
PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python \
  experiments/hardware/e07-hardware-point/hardware_point.py --dry-run --seeds 0      # no token
PYTHONPATH=... .venv/bin/python .../hardware_point.py --go --seeds 0 [--backend ibm_fez] [--budget-s 180]   # after GO
PYTHONPATH=... .venv/bin/python .../hardware_point.py --job-id <id> --seeds 0                               # recover a submitted job
```
Credentials: `QiskitRuntimeService()` defaults only (account saved by the user with `save_account(channel="ibm_quantum_platform", …)`); the script never reads, prints or stores a token. Provenance saved per run: backend and processor type, UTC timestamps (prepared / submitted / result received), job id (written before waiting), `job.usage()` seconds and `job.metrics()`, shots, circuit count, physical qubits, calibration snapshot of those qubits before and after (T1, T2, frequency, `sx` / `x` / `rz` / `measure` / `reset` error and duration, readout p(0|1) / p(1|0) when the backend exposes properties), transpiled circuits (QASM 3 + QPY), parameter values per row, raw counts per circuit, measured and exact Bloch vectors.

## Result (run 2026-10-25, `run-seed0/record.json`)
| Provenance | Value |
|---|---|
| Device | `ibm_fez` (Heron r2, 156 qubits); chosen by the rule: operational, 0 pending jobs |
| Job id | `dbesrqnpc3tc739mb92g` (one `SamplerV2` job, job mode, no session) |
| Timestamps (UTC) | prepared / submitted 09:37:45; created 09:37:46.49; running 09:37:47.55; finished 09:39:21.74; result received 09:39:23 |
| QPU billed | **82 s** (`job.usage()`; `metrics.usage.qpu_charge_time_seconds` 82; 1.362 resource units); model estimate 79–110 s → inside |
| Circuits × shots | 300 × 1,024 = 307,200 shots (3 PUBs × 100 parameter sets) |
| Physical qubits | [4, 2, 130, 50] for logical qubits 0–3 (opt level 3, seed 0; Z layout pinned for X and Y) |
| Transpiled circuits | 4 qubits, 0 two-qubit gates, depth 10, duration 1.73 µs; `run-seed0/circuit_{X,Y,Z}.qasm3` and `.qpy`, parameters in `parameters_*.csv` |
| Sampler options | shots 1,024; dynamical decoupling off; gate and measurement twirling off |
| Calibration snapshot (properties last updated 2026-10-25 08:53 UTC; identical before and after the job) | q4: T1 117 µs, T2 104 µs, sx 2.98e-4, readout 3.8e-3 (p(0\|1) 6.3e-3, p(1\|0) 1.2e-3) · q2: T1 148, T2 128 µs, sx 1.38e-4, readout 4.5e-3 (5.9e-3 / 3.2e-3) · q130: T1 158, T2 153 µs, sx 4.11e-4, readout 2.8e-3 (5.4e-3 / 2.4e-4) · q50: T1 108, T2 94 µs, sx 4.07e-4, readout 2.3e-3 (6.3e-3 / 7.3e-4); measurement 1.66 µs; rz error 0 |
| Raw data | `counts.json` (3 settings × 100 rows × 1,024 shots), `bloch_measured.npy`, `bloch_exact.npy` |

| Analysis (40 test rows) | Accuracy | BA | Kernel r vs exact | Mean \|Bloch error\| |
|---|---|---|---|---|
| Exact simulator | 0.875 (35/40) | 0.873 | 1.000 | 0 |
| FakeFez prediction (20 draws) | 0.880 ± 0.010 [0.875, 0.900] | 0.880 | 0.9983 (train) | 0.018 |
| **Hardware, variant 1** (exact train features, `ibm_fez` test features) | **0.900 (36/40)** | **0.907** | 0.9991 (test block) | 0.020 |
| **Hardware, variant 2** (`ibm_fez` features for train and test) | **0.900 (36/40)** | **0.907** | 0.9981 (train block) | 0.020 |

- Δ vs the FakeFez prediction: **+2.0 pp** (one test row), inside the pre-registered ± 2 pp rule, inside the prediction's shot-noise range [0.875, 0.900], and within one test row. Hardware is +2.5 pp above the exact simulator (one more row correct: a shot-noise-sized move, not a systematic gain).
- The device bias is the size the twin predicted: mean |Bloch error| 0.020 on hardware vs 0.018 in simulation (≈ 0.018 is the 1,024-shot binomial floor); kernel r ≥ 0.998 on both.
- Cost: 82 billed QPU seconds for 300 circuits = 0.273 s per 1,024-shot circuit, matching the e03 bracket [0.264, 0.365] and the published 0.27–0.32 s on ibm_fez.

## Verdict
- **H8 confirmed → BT-3 met** on the pre-registered rule (both variants). Stage budget used: 82 of ≤ 180 QPU s; one job, no retry.
- For the paper: a first-class hardware section with every row of the provenance table above; the claim is "the 4-qubit, 0-two-qubit-gate readout runs on ibm_fez with device bias at the shot-noise floor and accuracy inside the simulator's prediction", never more. Limits: one seed, 40 test rows (2.5 pp per row), one device, one calibration.
- Phase 8: ship this as `qlassifier benchmark --hardware` / `Qlassifier(backend="ibm:<name>")`: the dry run (plan + prediction + cost) first, explicit confirmation, then the job; copy `run-seed0/` to `benchmarks/hardware/`.

## Seeds 1–4 (added 2026-10-25, after the pre-registered run; user request)
With QPU budget left, four more seeds of the same design ran on `ibm_fez` (`hardware_point.py --go --seeds 1 2 3 4 --backend ibm_fez`, one job per seed, same subset rule, exact training and tuning). Their FakeFez predictions were computed **after** the runs (`--dry-run --seeds 0 1 2 3 4`, same procedure and sampler seeds 11–30), so they are checks, not pre-registered tests. Records: `run-seed1/` … `run-seed4/` (same files as `run-seed0/`).

| Seed | Job | Submitted | Physical qubits | QPU s | Exact | Predicted (FakeFez) | Hardware | Δ (pp) | Within 2 pp | Kernel r (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `dbesrqnpc3tc739mb92g` | 09:37 UTC | [4, 2, 130, 50] | 82 | 0.875 | 0.880 ± 0.010 | 0.900 | +2.0 | yes | 0.999 |
| 1 | `dbeutp8m94ac738silb0` | 11:58 UTC | [86, 50, 4, 130] | 82 | 0.975 | 0.975 ± 0.000 | 0.975 | -0.0 | yes | 0.998 |
| 2 | `dbeuv97pc3tc739mf740` | 12:01 UTC | [86, 50, 4, 130] | 82 | 0.875 | 0.874 ± 0.010 | 0.875 | +0.1 | yes | 0.999 |
| 3 | `dbev0anpc3tc739mf97g` | 12:03 UTC | [86, 50, 4, 130] | 82 | 0.975 | 0.975 ± 0.000 | 0.975 | -0.0 | yes | 0.999 |
| 4 | `dbev157sstms73fcr4s0` | 12:05 UTC | [86, 50, 4, 130] | 82 | 0.950 | 0.954 ± 0.009 | 0.950 | -0.4 | yes | 0.999 |

- All 5 seeds within 2 pp of their prediction and inside the shot-noise range; mean deviation +0.3 pp; 410 QPU s in all (190 s of the Open plan's monthly 600 s left).
- Same caveats as above: one device, one day's calibration, 40 test rows per seed (one row = 2.5 pp). The paper's hardware section and `generated/hardware-seeds.tex` are built from these records by `scripts/paper_tables.py`.
