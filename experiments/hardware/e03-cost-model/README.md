# e03 — cost model (BT-4): exact counts, QPU-time bracket vs published timings

Up: [../README.md](../README.md) (protocol §2.4) · Log: [../learning-log.md](../learning-log.md) (H3) · Code: [`../common/cost_model.py`](../common/cost_model.py) · Raw: [`cost_model.json`](cost_model.json) · Sources: prior art §2

## What
BT-4 asks for exact circuit / shot / two-qubit-gate counts and a QPU-time model that lands inside the published per-circuit range (0.27–0.69 s). The model is IBM's own job-time model with device defaults (no fitted constant): **low** = Σ shots × (rep_delay + circuit duration) + 2 s per job of ≤ 300 circuits; **high** = Σ shots × 350 µs (IBM's quick per-execution figure) + the same overhead. rep_delay = `default_rep_delay` = 250 µs (FakeTorino, FakeFez, FakeBrisbane); durations from `estimate_duration` of the transpiled circuits.

## How
`PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python experiments/hardware/common/cost_model.py --out experiments/hardware/e03-cost-model` (39 s). Counts come from the shipped `cost_study` (Breast Cancer, seed 0: N_tr 398, N_te 171); the executed-count check fits `Qlassifier(backend="fake_fez", random_state=0)` on that split and counts every parameter set Aer executes.

## Result
### Published per-circuit timings vs the model (FakeFez durations of the published circuit class)
| Source | Device | Circuits × shots | Measured per circuit | Circuit class used (2Q, depth, duration) | Model per circuit [low, high] | Inside |
|---|---|---|---|---|---|---|
| arXiv:2607.20377 | ibm_fez | 900 × 1,024 | 0.271 s | 4-qubit ZZ linear compute–uncompute (22 CZ, depth 74, 4.3 µs; paper: ≤ 22 CZ, depth ≤ 102) | [0.267, 0.365] | yes |
| arXiv:2604.18837 (code archive) | ibm_fez | 1,830 × 1,024 | 0.319 s | 5-qubit product map (0 CZ, 1.6 µs) and 5-qubit ZZ linear (30 CZ, depth 90, 4.3 µs) | [0.268, 0.366] | yes |
| arXiv:2605.14672 | ibm_pittsburgh | 1,675 × 2,048 | 0.691 s | 4-qubit ZZ full (64 CZ, depth 202, 8.9 µs) / 8-qubit ZZ linear (54 CZ, 5.8 µs) | [0.537, 0.724] | yes |
- The low bound is almost all repetition delay: 1,024 × 250 µs = 0.256 s per circuit; circuit duration adds 1.6–31 µs per shot (0.6–12%); the 2 s job overhead adds ≈ 7 ms per circuit at 300 circuits per job.

### Breast Cancer fit + predict (N_tr 398, N_te 171; 1,024 shots; FakeFez)
| Model | Qubits | 2Q gates | Depth | Duration | Circuits | QPU low – high |
|---|---|---|---|---|---|---|
| Qlassifier | 4 | 0 | 6 | 1.61 µs | 1,707 | **452 – 624 s** |
| PQK | 9 | 72 | 165 | 6.26 µs | 1,707 | 460 – 624 s |
| FQK-bw linear | 8 | 54 | 133 | 5.76 µs | 147,061 (+ 2,276,238 on-device tuning) | 39,497 – 53,689 s (11.0 – 14.9 h) |
| Separable | 8 | 0 | 6 | 1.61 µs | 147,061 (+ 2,276,238) | 38,872 – 53,689 s |
| ZZ QSVC / FQK-bw full | 8 | 427 | 768 | 30.85 µs | 147,061 | 43,276 – 53,689 s |
- Per-sample readout (Qlassifier, PQK) costs ≈ 8–10 QPU minutes for the whole dataset; every fidelity kernel needs 11–15 QPU hours before tuning. Torino and Brisbane: same rep_delay, durations within 1 µs (JSON).

### Executed-count check (shipped estimator on FakeFez, Breast Cancer seed 0)
- `resources_`: circuits_fit 1,194 (= 3 × 398), circuits_predict 513 (= 3 × 171), two_qubit_gates 0, depth 10, shots 1,024. Model: 3 × 569 = 1,707. Aer executed 1,707 parameter sets per qubit column (27 experiments: 4 qubits × 3 settings × fit + predict, plus the 3 whole-register control circuits × 20 rows) → **match**.

## Verdict
- H3 confirmed: all three published measurements lie inside the model bracket; counts equal executed circuits. BT-4 (phase-7 evidence) met; phase 9 turns the executed-count check into a unit test and the model into `cost.qpu_seconds(low, high)`.
- Recommendation for phase 8: report the bracket, label it "estimate (IBM job model, default rep_delay)", and compare against `job.usage()` when a hardware run exists.
