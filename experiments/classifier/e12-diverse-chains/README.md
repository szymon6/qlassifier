# e12 — last BT-6 attempt: diverse chains (H12) and wider context (H13), chain-grouped, fresh seeds 300–309

Up: [../README.md](../README.md) (protocol §2.2) · Code: [`fetch_pdb.py`](fetch_pdb.py), [`build_dataset.py`](build_dataset.py), [`summarize.py`](summarize.py) · Data: [`../data/README.md`](../data/README.md) (`protein_div.csv`) · Raw: [`div6.csv`](div6.csv), [`div10.csv`](div10.csv) · Summary: [`summary.md`](summary.md) · Log: [learning-log](../learning-log.md) (iteration 5, pre-registered)

- **Hypotheses (pre-registered, k = 2, corrected 97.5% CIs):**
  - H12: chain diversity gives the grouped test power; geometry mode beats every tuned classical model with 6 angles (φ/ψ of i−1 … i+1). Predicted +0.5 to +1.5 pp.
  - H13: same rows with 10 angles (i−2 … i+2, 10 circle qubits). Predicted +0.5 to +2 pp.
  - Pass rule: mean BA above all five classical models and 97.5% CI of Δ vs the best classical > 0.
- **Data:** `protein_div.csv`, 4,000 residues from 1,773 non-redundant PDB chains (≤ 3 rows per chain, CC0), strand share 24.9%. Power analysis (from e11) fixed N at the 4,000 cap.
- **Setup:** recipe `native`; outer `GroupShuffleSplit(0.3)` by chain (test class-share drift ≤ 2.0 pp), inner `StratifiedGroupKFold(5)` by chain; fresh seeds 300–309; 30-config grids (ZZ 4); 10 models. n_train 2,765–2,815, n_test 1,185–1,235 (mean n_te/n_tr 0.432). Runtime 1,575 s (H12) and 1,610 s (H13), run in parallel.

## H12 result: `protein_div6` (BA, mean over 10 fresh seeds)
| Model | BA (± sd) | Acc | ROC-AUC | F1 |
|---|---|---|---|---|
| **geometry mode** | **0.889 ± 0.011** | 0.914 | 0.960 | 0.830 |
| generic Bloch | 0.887 ± 0.013 | 0.911 | 0.950 | 0.825 |
| MLP (best classical) | 0.887 ± 0.011 | 0.914 | 0.965 | 0.829 |
| gradient boosting | 0.884 ± 0.013 | 0.915 | 0.966 | 0.829 |
| RBF-SVM / separable | 0.882 / 0.882 | 0.908 / 0.907 | 0.950 / 0.955 | 0.820 / 0.817 |
| random forest | 0.878 ± 0.013 | 0.916 | 0.967 | 0.827 |
| logistic regression / pinned ZZ | 0.844 / 0.798 | 0.883 / 0.856 | 0.925 / 0.902 | 0.766 / 0.704 |
- Geometry − MLP: **+0.25 [−1.13, +1.62] pp** (97.5%), p = 0.64, 6/10. Accuracy −0.04, ROC-AUC −0.49, F1 +0.08 pp (all CIs span 0).
- Significant: − RF +1.11 [+0.22, +2.01]; − LR +4.5; − ZZ +9.1 pp. Geometry on − off: +0.17 [−1.43, +1.78].
- **H12: FAIL** (mean above every classical, CI includes 0).

## H13 result: `protein_div10` (BA, mean over 10 fresh seeds)
| Model | BA (± sd) | Acc | ROC-AUC | F1 |
|---|---|---|---|---|
| **geometry mode** | **0.897 ± 0.009** | 0.919 | 0.968 | 0.841 |
| gradient boosting (best classical) | 0.889 ± 0.010 | 0.919 | 0.969 | 0.837 |
| MLP | 0.884 ± 0.009 | 0.913 | 0.965 | 0.827 |
| separable / random forest | 0.883 / 0.883 | 0.913 / 0.921 | 0.962 / 0.970 | 0.825 / 0.836 |
| generic Bloch | 0.877 ± 0.008 | 0.906 | 0.958 | 0.813 |
| RBF-SVM | 0.874 ± 0.010 | 0.905 | 0.951 | 0.811 |
| logistic regression | 0.843 ± 0.011 | 0.884 | 0.931 | 0.767 |
| pinned ZZ (10 qubits) | 0.543 ± 0.009 | 0.770 | 0.892 | 0.159 |
- Geometry − GB: **+0.81 [−0.61, +2.23] pp** (97.5%), p = 0.16, 8/10. Accuracy +0.03, ROC-AUC −0.12, F1 +0.43 pp (all CIs span 0).
- Other: − MLP +1.35 [−0.59, +3.29] (10/10); − RF +1.46 [−0.09, +3.00] (10/10); − RBF-SVM +2.29 [+0.16, +4.43]; − LR +5.4; − ZZ +35.5 pp (ZZ collapses at 10 qubits). Geometry on − off: **+2.04 [+0.65, +3.42]**: with 10 angles the geometry itself matters.
- **H13: FAIL** (mean above every classical, CI includes 0).

## Power check
- Realised sd of per-split Δ vs the best classical: 0.70 pp (H12) and 0.73 pp (H13), against 0.80 pp in e11: the pessimistic "residue noise" scenario of the power analysis (0.69 pp), not the chain-dominated one (0.27 pp). 1,773 chains did not buy the variance reduction.

## Decision
- Both confirmatory tests fail → **BT-6 missed**. The stop rule fires (iterations 3, 4, 5 without a measurable gain on the open target). No further iterations.
- What the evidence does support: geometry mode has the **top mean BA of all 10 models on every protein variant** (residue split, chain-grouped e11, H12, H13), and significantly beats RBF-SVM / RF / LR / ZZ on most of them; it ties gradient boosting and MLP.
