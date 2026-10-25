# e06 — hardware resources (BT-8) and circuit cost (BT-7)

Up: [../README.md](../README.md) (protocol §2.9) · Code: [`resources.py`](resources.py) (from archived e06; BT-7 count updated) · Raw: [`resources.json`](resources.json) · Log: [learning-log](../learning-log.md)

- **Question:** qubits, transpiled two-qubit gates and depth per circuit; circuits to train + predict on Breast Cancer vs the bandwidth-tuned fidelity-kernel (FQK-bw) pipeline incl. its bandwidth search.
- **Setup:** transpile to FakeTorino (Heron r1), FakeFez (Heron r2), FakeBrisbane (Eagle r3), `optimization_level=3`, transpiler seeds 0–2 (median). Error-free shot ≈ Π(1 − gate error)·Π(1 − readout error) from `backend.target` (crude: no idling, crosstalk). Run: `.venv/bin/python resources.py` (11 s).

## Result: circuits (median of 3 transpiler seeds)
| Circuit | Qubits | 2Q gates Torino / Fez / Brisbane | Depth (Torino) | Error-free shot Torino / Fez / Brisbane |
|---|---|---|---|---|
| **Qlassifier product map, PCA-8 (1 of 3 settings)** | **4** | **0 / 0 / 0** | **6** | **0.976 / 0.990 / 0.972** |
| Qlassifier product map, Breast Cancer all 30 features | 15 | 0 / 0 / 0 | 6 | 0.898 / 0.953 / 0.871 |
| pinned ZZ QSVC (full), compute-uncompute | 8 | 393 / 428 / 394 | 631 | 0.191 / 0.233 / 0.051 |
| FQK-bw (linear ZZ), compute-uncompute | 8 | 54 / 54 / 54 | 135 | 0.677 / 0.790 / 0.617 |
| separable, compute-uncompute | 8 | 0 / 0 / 0 | 6 | 0.949 / 0.977 / 0.937 |
| PQK Heisenberg (1 of 3 settings) | 9 | 72 / 72 / 72 | 167 | 0.617 / 0.734 / 0.531 |

- **BT-8 met:** 4 = ⌈8/2⌉ qubits; 0 ≤ n − 1 = 3 two-qubit gates on all three devices (also 0 at 15 qubits).

## Result: circuit evaluations, Breast Cancer 70/30 (398 train / 171 test)
| Pipeline | Circuits |
|---|---|
| **Qlassifier: 3 settings × 569 samples; tuning is classical post-processing** | **1,707** |
| FQK-bw: bandwidth search on device, 6 bandwidths × 5 folds | 2,276,238 |
| FQK-bw: final Gram at the chosen bandwidth | 147,061 |
| FQK-bw pipeline total | 2,423,299 |
| PQK (3 settings) | 1,707 (same count; 9 qubits, 72 two-qubit gates per circuit) |

- **BT-7 met:** 2,423,299 / 1,707 = **1,420×** fewer circuits (≥ 100×). Against the final Gram alone (bandwidth tuned off-device, possible only while the fidelity kernel is classically simulable): 86×.
- **Decision:** cost and qubit targets hold for the product map at any feature count (0 two-qubit gates). No change needed.
