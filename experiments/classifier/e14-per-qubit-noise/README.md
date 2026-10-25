# e14 — per-qubit vs whole-register simulation on IBM fake backends (phase 9)

Up: [../README.md](../README.md) · Code: [`run.py`](run.py) · Raw: [`exact.csv`](exact.csv), [`shots.csv`](shots.csv) · Log: [learning-log → phase-9 addendum](../learning-log.md#phase-9-addendum--per-qubit-device-simulation-e14) · Decision: ADR 0006 amendment

- **Trigger:** the shipped noise study stalled on `protein_angles` part A (10 qubits): a fake-device cell simulates 4,000 rows × 3 settings × 1,024 shots of the whole register, and Aer's cost grows 4ⁿ (density matrix) or 2ⁿ per shot; e08's 20–36 s at 6 qubits became ≈ 2 h per cell at 10. `Qlassifier(backend="fake_fez")` on raw Breast Cancer (15 qubits) did not finish in 10 min.
- **Claim:** Qlassifier circuits are product states with 0 two-qubit gates. The fake backends' noise models act qubit by qubit (single-qubit gate errors, single-qubit readout errors, no delay / idle noise, no crosstalk), and the readout uses per-qubit marginals only. So running each qubit alone, on the physical qubit the whole-register transpile pinned, samples exactly the same marginal distribution, at linear cost.
- **Kill criterion:** any exact marginal differing by more than 1e-12, or a shot-level accuracy gap outside the binomial standard error.

## Part 1 — exact marginals (no shots)
Real rows encoded on 4 qubits (Breast Cancer, PCA-8) and 6 qubits (Sonar, PCA-12), 5 rows each; every Pauli setting (X, Y, Z); FakeTorino, FakeFez, FakeBrisbane. Noisy density matrix of the whole register (opt 3, transpiler seed 0) vs of each one-qubit circuit on the same physical qubit; then that qubit's readout matrix.

| Check | Result |
|---|---|
| Gate sequence per physical qubit (whole register vs one-qubit circuit) | identical (inspected) |
| Max \|P(1) whole − P(1) per qubit\| over 450 marginals (2 registers × 3 devices × 3 settings × 5 rows × qubits) | **3.3e-16** |
| Trace of Aer's noisy density matrices | 1 − 7.7e-10 at most (Aer's thermal-relaxation channels are trace preserving to ~1e-10); marginals are normalised by the trace, as sampling does. Raw, un-normalised marginals differ by up to 6.3e-10 for that reason only. |

## Part 2 — shot level (noise study part A, Breast Cancer, seeds 0–4, 1,024 shots, sampler seed 11)
| Device | Accuracy, whole register (phase-7 path) | Accuracy, per qubit (shipped path) | Difference (pp) | Binomial SE of the difference (pp) |
|---|---|---|---|---|
| FakeTorino | 0.9485 | 0.9485 | 0.00 | 1.08 |
| FakeFez | 0.9474 | 0.9485 | +0.12 | 1.08 |
| FakeBrisbane | 0.9450 | 0.9485 | +0.35 | 1.08 |
- Both paths draw different random samples (different circuits per PUB), so cell-level numbers differ; every device mean is within one standard error. The whole-register column equals the phase-7 e05 numbers exactly (drops 0.12 / 0.23 / 0.47 pp).

## Decision
- **Keep per-qubit execution** for product circuits on local noise models: `backends.per_qubit_exact` checks 0 two-qubit gates, no delay instruction and local noise (`backends.local_noise`), else the whole register is simulated (FQK-bw, PQK and any future entangled path).
- Applies to `Qlassifier(backend="fake_*")` (Breast Cancer, 15 qubits: fit + predict in seconds instead of > 10 min) and to the Qlassifier cells of the noise study. Phase-7 numbers (e05, e08) stay as they are: they are history, produced by the whole-register path.
