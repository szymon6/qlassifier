# e00 — kernel correctness checks (re-run after copy)

Up: [../README.md](../README.md) · Code: [`check_kernels.py`](check_kernels.py) · Raw: [`results.json`](results.json) · Harness: [`../common/lab.py`](../common/lab.py) · Log: [learning-log](../learning-log.md) (iteration 0)

- **Question:** do the copied fast numpy kernels still equal the Qiskit circuits they stand for (qiskit 2.6.1)?
- **Setup:** random inputs; Qiskit `Statevector` vs `lab.py`. Same script as archived e00.
- **Result:** all checks pass (`all_ok: true`).

| Check | Max abs error |
|---|---|
| Bloch vectors, product map RX·RY | 4.4e-16 |
| Product fidelity Π(1+r·r′)/2 vs statevector overlap | 1.3e-15 |
| ZZ map (`zz_feature_map`, full / linear, reps 2) kernel | 3.1e-15 / 2.2e-15 |
| Native entangled map (CZ chain, 1–2 layers) local Paulis | ≤ 3.3e-16 |
| Native product (layers 0) kernel = closed-form Bloch kernel | 1.8e-15 |
| Heisenberg PQK local Paulis | 1.4e-15 |
| Separable per-feature kernel 3/4 + cos(Δ)/4 | 2.2e-16 |
| Product kernel from noisy, shrunk vectors: min eigenvalue | +0.076 (PSD) |

- **Decision:** trust all fast paths. Re-run: `.venv/bin/python experiments/classifier/e00-kernel-checks/check_kernels.py`.

## e00b — candidate encodings of this loop
- Code: [`check_candidates.py`](check_candidates.py) · Raw: [`results_candidates.json`](results_candidates.json). All pass.

| Check (vs Qiskit `Statevector`) | Max abs error |
|---|---|
| Data re-uploading [RX·RY]×L, L = 1, 2, 3 (`cands.reupload_bloch`) | ≤ 6.7e-16 |
| Riemann-sphere qubit RY(2·arctan|w|)·P(arg w) (`cands.riemann_bloch`) | 3.3e-16 |
| Product fidelity of Riemann states; chordal identity |1+z̄w|²/((1+|z|²)(1+|w|²)) | 4.4e-16; 0 |
| Geometry-aware map: periodic RY(2πx/P), lat/lon RY(90°−lat)·RZ(lon), Bloch pairs (`cands.geo_bloch`) | 3.3e-16 |
