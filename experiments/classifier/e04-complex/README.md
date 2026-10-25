# e04 — complex-valued radar returns on qubits (H2), Ionosphere, recipe `native`, dev seeds

Up: [../README.md](../README.md) · Candidates: [`../common/cands.py`](../common/cands.py) (`complex_specs`, `riemann_bloch`) · Raw: [`results.csv`](results.csv) · Summary: [`summary.md`](summary.md) · Log: [learning-log](../learning-log.md) (iteration 1)

- **Hypothesis H2:** Ionosphere's 34 columns are 17 complex autocorrelation values (UCI 52). One qubit per complex value fits this structure: Riemann-sphere state |0⟩ + w|1⟩ (inverse stereographic projection, cf. arXiv:2308.03949) or the Bloch map on aligned (Re, Im) pairs with one shared scale. Predicted +0.5–2 pp over the standard encoding; kill if Δ vs best classical < +1 pp.
- **Setup:** recipe `native` (raw 33 columns; classical models and the control scale internally on training rows; complex encodings read raw values, no per-column standardisation). Dev seeds 100–109. Models: 5 classical, separable, `bloch_gauss_std` (standard encoding = all-features Bloch kernel), `cplx_riemann` (w = z / rms|z|, 17 qubits), `cplx_pairs` (RX/RY on (Re, Im), shared scale). All kernel models: same 30-config ridge grid. Encodings verified vs Qiskit (`check_candidates.py`: ≤ 4.4e-16; chordal identity exact).

## Result (BA, mean ± std, 10 dev seeds)
| Model | BA | Δ vs best classical (RBF-SVM), corrected 95% CI |
|---|---|---|
| `bloch_gauss_std` | 0.941 ± 0.022 | −0.1 [−1.4, +1.2] |
| RBF-SVM | 0.941 ± 0.024 | — |
| `cplx_riemann` | 0.935 ± 0.042 | −0.6 [−6.7, +5.4] |
| `cplx_pairs` | 0.933 ± 0.017 | −0.8 [−5.4, +3.8] |
| separable | 0.932 ± 0.022 | −0.9 [−4.3, +2.5] |
| random forest / GB | 0.917 / 0.909 | — |
| MLP / LR | 0.879 / 0.845 | — |

- `cplx_riemann` wins some splits (+2.8, +5.0 pp on seeds 101, 106) but collapses on others (0.831 on seed 102, C picked at the low edge). Inner CV picks the narrow end (m = 2) on 9/10 splits: the encoding is less stable, not better.
- All features (native) lift every kernel model vs PCA-8 (Bloch 0.941 here vs 0.910 on the same dev seeds at PCA-8), but RBF-SVM gains the same.

## Decision
- H2 dropped: Δ vs best classical −0.6 / −0.8 pp (< +1 pp kill line). Complex-aware encoding is not a BT-6 route on Ionosphere.
- Kept as a negative result: matching the data's algebra (ℂ ↔ qubit) is not enough; the label must depend on structure that generic kernels miss.
