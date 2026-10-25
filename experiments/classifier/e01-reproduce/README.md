# e01 — reference leaderboard (archived protocol re-run, eval seeds 0–9)

Up: [../README.md](../README.md) · Runner: [`../common/run.py`](../common/run.py) · Harness: [`../common/lab.py`](../common/lab.py) · Raw: [`results.csv`](results.csv) · Summary: [`summary.md`](summary.md) · Log: [learning-log](../learning-log.md) (iteration 1)

- **Question:** does the copied (and extended) harness reproduce the archived exact-mode benchmark? What is the reference leaderboard for this stage?
- **Setup:** protocol §2, recipe `pca8`, 4 datasets × eval seeds 0–9. 12 models: trivial, LR, RBF-SVM, RF, GB, MLP, pinned ZZ, FQK-bw (full / linear), PQK, separable, `bloch_gauss`. Run: `../common/run.py results.csv --seeds 0-9 --k 8 --models …` (449 s on 10 shared cores).
- **Reproduction:** 480 / 480 rows identical to the archived `e02-exact-benchmark/results.csv` (BA, acc, AUC, F1 max diff 0.0; same chosen configs). The harness patches (recipes, extra specs, class weights) do not change `pca8` behaviour.

## Result (BA, mean ± std over 10 seeds)
| Model | Breast Cancer | Ionosphere | Sonar | Heart | Mean |
|---|---|---|---|---|---|
| MLP | 0.961 ± 0.019 | 0.915 ± 0.030 | 0.820 ± 0.052 | 0.820 ± 0.027 | 0.879 |
| **`bloch_gauss`** | **0.948 ± 0.012** | **0.925 ± 0.018** | **0.824 ± 0.053** | **0.816 ± 0.026** | **0.878** |
| RBF-SVM | 0.949 ± 0.012 | 0.926 ± 0.021 | 0.814 ± 0.055 | 0.820 ± 0.025 | 0.877 |
| separable | 0.952 ± 0.015 | 0.898 ± 0.031 | 0.805 ± 0.040 | 0.826 ± 0.025 | 0.870 |
| FQK-bw full | 0.944 ± 0.018 | 0.891 ± 0.031 | 0.800 ± 0.036 | 0.823 ± 0.029 | 0.864 |
| PQK | 0.950 ± 0.014 | 0.900 ± 0.035 | 0.806 ± 0.043 | 0.793 ± 0.043 | 0.862 |
| FQK-bw linear | 0.946 ± 0.020 | 0.904 ± 0.031 | 0.772 ± 0.037 | 0.800 ± 0.053 | 0.856 |
| random forest | 0.923 ± 0.021 | 0.905 ± 0.026 | 0.784 ± 0.072 | 0.792 ± 0.030 | 0.851 |
| grad. boosting | 0.948 ± 0.018 | 0.886 ± 0.024 | 0.754 ± 0.040 | 0.784 ± 0.031 | 0.843 |
| logistic regression | 0.954 ± 0.018 | 0.845 ± 0.029 | 0.721 ± 0.046 | 0.826 ± 0.025 | 0.836 |
| pinned ZZ QSVC | 0.571 ± 0.023 | 0.698 ± 0.042 | 0.545 ± 0.051 | 0.533 ± 0.047 | 0.587 |
| trivial | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

## BT readout (eval seeds; archived method)
- BT-1: BC accuracy 0.956 ± 0.011 vs ZZ 0.677 ± 0.018 → **+27.9 pp** (≥ 20): met.
- BT-2: BA above ZZ on all 4 datasets (+37.6 / +22.6 / +27.9 / +28.3 pp); paired wins 10/10 on each: met.
- BT-3 / BT-4 vs published (BC 0.968, Iono 0.911, Sonar 0.813, Heart 0.804): −2.0 / +1.4 / +1.1 / +1.2 pp → BT-3 3/4 met, BT-4 3/4 met.
- BT-5: mean BA 0.878 > separable 0.870 > FQK-bw full 0.864 > PQK 0.862 > FQK-bw linear 0.856 > ZZ 0.587: met.
- BT-6: Δ vs best classical −1.3 [−3.1, +0.5] (BC, MLP), −0.2 [−3.1, +2.7] (Iono, RBF), +0.5 [−8.5, +9.5] (Sonar, MLP), −1.0 [−3.1, +1.2] (Heart, LR): **missed** (ties).
- Tuning time per split under shared load: `bloch_gauss` 0.54 s; MLP 56 s, RF 36 s dominate.
- **Decision:** reference leaderboard fixed. Candidates must beat `bloch_gauss` on dev seeds first.
