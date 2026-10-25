# e00 — setup checks: scaled FakeFez noise model, cell costs

Up: [../README.md](../README.md) (protocol §2.3, §2.4) · Log: [../learning-log.md](../learning-log.md) · Code: [`probe.py`](probe.py) · Raw: [`probe.log`](probe.log)

## What
Before the sweep: (1) does scaling the device Target (errors × s, T1 and T2 ÷ s) scale every Aer channel by s? (2) what do the sweep cells cost (time, RSS) so the worker count and the scale set can be fixed? (3) circuit durations for the cost model.

## How
`OMP_NUM_THREADS=1 .venv/bin/python experiments/hardware/e00-setup-checks/probe.py` (≈ 1 min). Qiskit 2.x Targets hand out copies of their properties, so changes go through `Target.update_instruction_properties` and the `qubit_properties` setter; the fake backend's `_target` is replaced and `NoiseModel.from_backend` rebuilds the model.

## Result
| Check | Value |
|---|---|
| Base FakeFez model | 1,132 error entries (156 readout, 976 gate); build 1.8 s |
| s = 0.5 / 2.0 | readout flip q0 0.01147 → 0.00574 / 0.02295 (×0.50 / ×2.00); `sx` q0 non-identity probability ×0.50 / ×2.00; channel kinds unchanged (`id, x, y, z, reset`) |
| Per-qubit Qlassifier cell, Breast Cancer 569 rows × 4 qubits × 3 settings, 1,024 shots | s = 0: 1.2 s; s = 2: 2.8 s; mean |R − exact| 0.0174 / 0.0178; RSS 1.1–1.3 GB (whole process) |
| FQK-bw linear compute–uncompute, 8 qubits, FakeFez | 54 CZ, depth 169, duration 5.76 µs; 46 ms per 1,024-shot circuit (`density_matrix` = `automatic`) → 60/40 paired Gram (4,170 circuits) ≈ 3.2 min |
| Pinned ZZ full compute–uncompute, 8 qubits, FakeFez | 433 CZ, depth 861, duration 30.8 µs; 261 ms per circuit → paired Gram ≈ 18 min |
| Qlassifier settings X / Y / Z | duration 1.63 / 1.61 / 1.63 µs, depth 8 / 7 / 10 (measurement 1.56 µs dominates) |
| Peak RSS of the probe process | 1.55 GB |

## Verdict
- Target scaling is an exact per-channel scaling of the published error rates (readout and depolarizing); use it as the §2.3 definition.
- Cells fit the machine rules: ≤ 3 workers × ≤ 1.6 GB. Scale set fixed as §2.3: all six scales for Qlassifier and FQK-bw linear; ZZ full at s ∈ {0, 1, 3} on 2 seeds (6 × 18 min).
