# e11 — chain-grouped splits for the BT-6 claim (H11), `protein_strands`, fresh seeds

Up: [../README.md](../README.md) (protocol §2.2) · Harness: [`../common/lab.py`](../common/lab.py) (`run_seed(groups=…)`, `load_groups`) · Raw: [`results.csv`](results.csv) · Summary: [`summary.md`](summary.md) (from [`summarize.py`](summarize.py)) · Residue-split run: [e07](../e07-structure-fit/README.md) · Log: [learning-log](../learning-log.md) (iteration 4)

- **Hypothesis H11 (pre-registered):** the protein win survives chain-grouped splits. Predicted Δ vs best classical +0.5 to +1.5 pp. Kill: corrected 95% CI of Δ vs best classical not above 0 → BT-6 evidence downgraded, no rescue iterations.
- **Why:** e07 split residues at random; residues of one chain could sit in train and test (shared fold context). Grouped splits by chain are the stricter test.
- **Setup:** recipe `native`, fresh seeds 200–209. Outer `GroupShuffleSplit(test_size=0.3)` by `pdb` (test class share drift −2.7 to +2.7 pp, under the 3 pp limit, so no stratified-group variant); inner `StratifiedGroupKFold(5, shuffle=True)` by `pdb`. Zero chain overlap checked for the outer and inner splits. Same 30-config grids (ZZ 4). Models: trivial, LR, RBF-SVM, RF, GB, MLP, pinned ZZ, separable, generic Bloch, geometry mode. n_train 2,027–2,188, n_test 812–973; corrected CI with the mean realised n_te/n_tr (0.444). Run: `run_x.py results.csv --datasets protein_strands --seeds 200-209 --recipe native --grouped --models … --extra cands:protein_geo_specs` (510 s).
- Harness check: the ungrouped path is unchanged (e01 rows reproduce exactly).

## Result (mean over 10 fresh seeds)
| Model | BA grouped (± sd) | Acc | ROC-AUC | F1 | BA residue split (e07) |
|---|---|---|---|---|---|
| **Qlassifier geometry mode** | **0.879 ± 0.015** | 0.920 | 0.947 | 0.811 | 0.884 |
| separable | 0.877 ± 0.020 | 0.915 | 0.944 | 0.801 | 0.882 |
| MLP | 0.872 ± 0.014 | 0.920 | 0.956 | 0.807 | 0.870 |
| gradient boosting | 0.872 ± 0.018 | 0.921 | 0.963 | 0.807 | 0.870 |
| Qlassifier generic (geometry off) | 0.871 ± 0.015 | 0.914 | 0.931 | 0.798 | 0.872 |
| RBF-SVM | 0.863 ± 0.017 | 0.913 | 0.935 | 0.792 | 0.868 |
| random forest | 0.861 ± 0.014 | 0.920 | 0.960 | 0.800 | 0.857 |
| logistic regression | 0.809 ± 0.015 | 0.883 | 0.923 | 0.711 | 0.815 |
| pinned ZZ QSVC | 0.756 ± 0.017 | 0.861 | 0.879 | 0.636 | 0.767 |

| Comparison (BA unless noted) | Δ pp [corrected 95% CI] | p | Wins |
|---|---|---|---|
| **geometry − MLP (best classical)** | **+0.7 [−0.6, +2.1]** | **0.254** | 9/10 |
| geometry − MLP, accuracy / ROC-AUC / F1 | +0.0 / −0.8 / +0.4 (all CIs span 0) | — | — |
| geometry − GB | +0.8 [−0.3, +1.8] | 0.138 | 9/10 |
| geometry − RBF-SVM | +1.6 [+0.3, +2.9] | 0.021 | 10/10 |
| geometry − RF | +1.9 [+0.5, +3.2] | 0.012 | 10/10 |
| geometry − LR | +7.1 [+4.4, +9.8] | < 0.001 | 10/10 |
| geometry − pinned ZZ | +12.3 [+8.4, +16.2] | < 0.001 | 10/10 |
| geometry on − off | +0.8 [−0.6, +2.2] | 0.215 | 8/10 |

## Decision
- **H11 killed by its own criterion:** the mean prediction held (+0.7 pp, top mean BA, 9/10 wins vs MLP), but the corrected CI vs the best classical spans 0. Under the stricter split, BT-6 is **missed**: Qlassifier is top of the leaderboard on mean BA and significantly beats RBF-SVM, RF, LR and ZZ, but only ties MLP and GB.
- Residue-level evidence (e07: +1.4 [+0.3, +2.4]) is kept as context. The margin shrank from +1.4 to +0.7 pp; chain-level overlap likely inflated the residue-split margin (inference: the two runs use different test sets).
- Protocol: `protein_strands` uses chain-grouped splits from now on (README §2.2). No rescue iterations (coordinator rule).
