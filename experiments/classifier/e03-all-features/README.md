# e03 — all features instead of PCA-8 (H1, PRD Q-1), dev seeds

Up: [../README.md](../README.md) · Runner: [`../common/run_x.py`](../common/run_x.py) · Raw: [`results.csv`](results.csv) · Summary: [`summary.md`](summary.md) · Log: [learning-log](../learning-log.md) (iteration 1) · Archived: `01-fair-verdict` e03 (eval seeds, fewer models)

- **Hypothesis H1:** with all features (no PCA) the product map (⌈d/2⌉ qubits, 0 two-qubit gates) gains more than the classical models → a BT-6 win. Kill: no dataset with Δ(Q − best classical) ≥ +1.5 pp.
- **Setup:** protocol §2 with k = all (StandardScaler only), 4 datasets × dev seeds 100–109; LR, RBF-SVM, RF, GB, MLP, separable, `bloch_gauss`, `bloch_ms`, `bloch_reup2`. Qubits for `bloch_gauss`: 15 / 17 / 30 / 7 (BC / Iono / Sonar / Heart). ZZ / FQK / PQK cannot run at 30–61 qubits in exact simulation (not part of this side study). Run: 671 s on shared cores.

## Result (BA, mean ± std, 10 dev seeds)
| Model | BC | Heart | Iono | Sonar | Mean |
|---|---|---|---|---|---|
| RBF-SVM | 0.975 ± 0.011 | 0.820 ± 0.041 | 0.941 ± 0.024 | 0.868 ± 0.044 | 0.901 |
| separable | 0.976 ± 0.011 | 0.801 ± 0.048 | 0.932 ± 0.022 | 0.883 ± 0.053 | 0.898 |
| **`bloch_gauss`** | **0.974 ± 0.009** | **0.801 ± 0.030** | **0.941 ± 0.022** | **0.861 ± 0.048** | **0.894** |
| `bloch_ms` / `bloch_reup2` | 0.975 / 0.975 | 0.804 / 0.789 | 0.937 / 0.938 | 0.858 / 0.853 | 0.894 / 0.889 |
| MLP | 0.970 | 0.821 | 0.879 | 0.845 | 0.879 |
| GB / RF | 0.963 / 0.953 | 0.788 / 0.798 | 0.909 / 0.917 | 0.821 / 0.760 | 0.870 / 0.857 |
| LR | 0.972 | 0.817 | 0.845 | 0.734 | 0.842 |

- Gain vs PCA-8 on the same dev seeds ([e02](../e02-dev-pca8/README.md)), pp: `bloch_gauss` +0.9 / +0.3 / +3.1 / +3.0; RBF-SVM +1.0 / +2.5 / +3.7 / +7.3; separable +0.5 / −0.6 / +3.2 / +10.8.
- Δ `bloch_gauss` vs best classical (corrected 95% CI): BC −0.1 [−1.2, +0.9], Heart −2.0 [−5.2, +1.2] (MLP), Iono −0.1 [−1.4, +1.2], Sonar −0.7 [−2.1, +0.7] (RBF-SVM).
- Breast Cancer at all features: 0.974 ± 0.009 on dev seeds (archived eval-seed run: 0.957). Above the published 0.968 on these splits, but only outside the shared PCA-8 protocol.

## Decision
- H1 dropped as a BT-6 route: RBF-SVM gains as much or more; no dataset reaches Δ ≥ +1.5 pp.
- Leaderboard stays on shared PCA-8 features (every quantum baseline can run; matches the published protocol). `Qlassifier` itself has no feature cap (⌈d/2⌉ qubits, 0 two-qubit gates); all-features numbers go to the paper as a scaling side table.
