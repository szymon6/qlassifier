# e07 — geometry mode vs every tuned classical model on structure-fit data (H7, H8 → BT-6), fresh seeds

Up: [../README.md](../README.md) · Candidates: [`../common/cands.py`](../common/cands.py) (`geo_bloch`, `make_geo_specs`, `quake_geo_specs`, `protein_geo_specs`) · Data: [`../data/README.md`](../data/README.md) · Raw: [`quakes.csv`](quakes.csv), [`protein.csv`](protein.csv) · Summary: [`summary.md`](summary.md) (from [`summarize.py`](summarize.py)) · Groundwork: `bloch-sphere-angular-data.md` · Log: [learning-log](../learning-log.md) (iteration 2)

- **Hypotheses (pre-registered in the log):** H7 geometry mode (`qlass_geo`) beats every tuned classical model on raw angles; H8 the win comes from the geometry (generic Bloch encoding ≥ 2 pp lower).
- **Datasets** (chosen from the groundwork pilot on other seeds and rows; fixed before this run):
  - `deep_quakes`: 3,729 USGS M ≥ 6 events 2000–2024; inputs latitude, longitude (degrees); label depth ≥ 70 km (21.3%). Geometry: one sphere qubit, RY(90° − lat)·RZ(lon).
  - `protein_strands`: 3,000 PDB residues (label-free thinning, no shared dihedrals); inputs φ, ψ of residues i−1, i, i+1 (degrees); label strand (21.1%). Geometry: six circle qubits, RY(2π·angle/360°).
- **Setup:** protocol §2, recipe `native` (raw columns; classical models, pinned ZZ, separable and the generic Bloch encoding scale internally on training rows; geometry mode reads raw angles). **Fresh seeds 200–209.** 30-config grids (ZZ 4). FQK-bw / PQK skipped (no BT needs them here). Run: `run_x.py quakes.csv --datasets deep_quakes --seeds 200-209 --recipe native --models trivial logreg rbf_svm random_forest grad_boost mlp zz_qsvc separable bloch_gauss --extra cands:quake_geo_specs` (2,776 s on shared cores); same for `protein_strands` with `cands:protein_geo_specs` (1,283 s).
- **Twin identity:** with all columns angular, the geometry-mode Gram matrix equals an RBF Gram matrix on hand-made sin/cos (circles) or 3-D unit vectors (sphere): max difference ≤ 1.8e-15 on both datasets. The classical twin therefore scores exactly like geometry mode and is not re-run.

## Result: `protein_strands` (BA, mean ± std, 10 fresh seeds; n_train 2,100, n_test 900)
| Model | BA | Acc | ROC-AUC | F1 |
|---|---|---|---|---|
| **Qlassifier geometry mode** | **0.884 ± 0.011** | **0.918** | 0.952 | **0.810** |
| separable (Bowles) | 0.882 ± 0.010 | 0.915 | 0.947 | 0.804 |
| Qlassifier generic (geometry off) | 0.872 ± 0.014 | 0.910 | 0.934 | 0.791 |
| MLP | 0.870 ± 0.012 | 0.914 | 0.953 | 0.797 |
| gradient boosting | 0.870 ± 0.009 | 0.917 | 0.960 | 0.801 |
| RBF-SVM | 0.868 ± 0.009 | 0.909 | 0.938 | 0.787 |
| random forest | 0.857 ± 0.017 | 0.914 | 0.958 | 0.787 |
| logistic regression | 0.815 ± 0.014 | 0.880 | 0.913 | 0.712 |
| pinned ZZ QSVC | 0.767 ± 0.019 | 0.865 | 0.883 | 0.650 |
| trivial | 0.500 | 0.789 | 0.500 | 0.000 |

| Comparison | Δ BA pp [corrected 95% CI] | p | Wins |
|---|---|---|---|
| **geometry − MLP (best classical)** | **+1.4 [+0.3, +2.4]** | **0.015** | **10/10** |
| geometry − GB | +1.4 [+0.03, +2.7] | 0.046 | 10/10 |
| geometry − RBF-SVM | +1.6 [+0.7, +2.6] | 0.004 | 10/10 |
| geometry − RF | +2.7 [+1.0, +4.4] | 0.006 | 10/10 |
| geometry − LR | +6.9 [+4.7, +9.0] | < 0.001 | 10/10 |
| geometry − pinned ZZ | +11.7 [+8.9, +14.5] | < 0.001 | 10/10 |
| geometry on − off (H8) | +1.2 [−0.8, +3.1] | 0.225 | 8/10 |
- On these residue-level splits: mean BA above all five tuned classical models; corrected 95% CI vs the best classical (MLP) > 0 (p = 0.015, also under a Bonferroni correction for 2 datasets).
- **Superseded for the BT-6 verdict by [e11](../e11-grouped-split/README.md):** with chain-grouped splits (the protocol now, README §2.2) the margin is +0.7 [−0.6, +2.1] pp → BT-6 missed. These residue-level numbers stay as context.
- Other metrics: accuracy +0.4 pp vs MLP (CI spans 0); ROC-AUC −0.1 / −0.8 / −0.6 pp vs MLP / GB / RF (CIs span 0); F1 +1.3 [+0.2, +2.5] vs MLP. The edge is a better-balanced decision at equal ranking quality.
- Chosen geometry configs: m ∈ {1/8 … 1}, 3/10 at the wide edge (linear-limit corner, by design).

## Result: `deep_quakes` (BA, mean ± std, 10 fresh seeds; n_train 2,610, n_test 1,119)
| Model | BA | Acc | ROC-AUC |
|---|---|---|---|
| random forest | 0.835 ± 0.015 | 0.905 | 0.932 |
| gradient boosting | 0.835 ± 0.018 | 0.896 | 0.929 |
| **Qlassifier geometry mode** | **0.769 ± 0.016** | **0.883** | **0.912** |
| MLP | 0.764 ± 0.020 | 0.873 | 0.903 |
| Qlassifier generic (geometry off) | 0.723 ± 0.025 | 0.860 | 0.856 |
| RBF-SVM | 0.713 ± 0.017 | 0.862 | 0.884 |
| logistic regression, separable, pinned ZZ, trivial | 0.500 | 0.787 | 0.48–0.55 |
- Geometry − RF: −6.6 [−8.5, −4.8] pp → BT-6 missed here. Geometry − RBF-SVM: +5.6 [+2.3, +9.0]; geometry on − off: +4.6 [+0.1, +9.0] (H8 supported); geometry − ZZ: +26.9 pp (ZZ at majority level).
- Every kernel model (RBF, generic Bloch, geometry) picks the narrowest grid corner (m = 4, C̃ = 1,000) on 10/10 splits. Deep events sit in narrow subduction zones; a few-degree kernel width is outside the protocol grid. Trees adapt to local density. Follow-up: [e09](../e09-quake-grid/README.md).

## Decision
- H7 confirmed on `protein_strands` (residue-level splits), rejected on `deep_quakes`. Keep geometry mode (default off). `protein_strands` stays the showcase; the BT-6 verdict moved to e11 (missed under chain-grouped splits). `deep_quakes` is reported as a loss with its cause.
- Honest framing: the qubit encodes the angle's geometry natively; a classical model matches it only with hand-made sin/cos features (the twin). No quantum-advantage claim.
