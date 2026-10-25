# e09 — is the quake loss a grid limit? (H9, exploratory, post hoc)

Up: [../README.md](../README.md) · Specs: [`../common/cands.py`](../common/cands.py) (`quake_wide_specs`, `M_WIDE`) · Raw: [`results.csv`](results.csv) · Parent: [e07](../e07-structure-fit/README.md) · Log: [learning-log](../learning-log.md) (iteration 3)

- **Hypothesis H9:** on `deep_quakes` every kernel model sat at the narrow grid corner (e07). With the same ×8-shifted ridge grid for every kernel model (m ∈ {1, 2, 4, 8, 16, 32} × C̃ ∈ {0.1 … 1,000}, still 30 configs), geometry mode closes the gap to trees. Predicted 0.80–0.83, still below RF 0.835.
- **Setup:** same splits as e07 (seeds 200–209), recipe `native`; kernel models only (`qlass_geo_wide`, `bloch_gauss_wide`, `rbf_svm_wide`); tree / MLP rows reused from e07. Exploratory: the splits were already seen, so this explains the loss and cannot add a confirmatory win. Run: `run_x.py results.csv --datasets deep_quakes --seeds 200-209 --recipe native --models none --extra cands:quake_wide_specs` (57 s).

## Result (BA, mean ± std)
| Model | Protocol grid (e07) | ×8 grid (e09) |
|---|---|---|
| Qlassifier geometry mode | 0.769 ± 0.016 | **0.814 ± 0.020** |
| Qlassifier generic | 0.723 ± 0.025 | 0.768 ± 0.017 |
| RBF-SVM | 0.713 ± 0.017 | 0.762 ± 0.017 |
| random forest / gradient boosting (e07) | 0.835 / 0.835 | — |

- Geometry ×8 − RF: −2.1 [−5.2, +1.1] pp (wins 1/10); − GB: −2.0 [−5.2, +1.2].
- Geometry ×8 − RBF-SVM ×8: +5.2 [+1.8, +8.6] (10/10); geometry − generic, both ×8: +4.7 [+1.8, +7.6] (10/10).
- Kernel models still choose the high-C̃ corner (C̃ = 1,000) at m = 16: the label needs a few-degree, hard-margin boundary.

## Decision
- H9 confirmed as predicted: the grid explained 4.6 pp of the gap, not all of it. Trees adapt to the uneven density of subduction zones; one global kernel width does not. `deep_quakes` stays a reported loss.
- Lesson for phase 8: Qlassifier's self-tuning should flag edge hits (and the benchmark should log them for every kernel model). Any grid extension must apply to all kernel models alike.
- Geometry beats the matched classical kernel (RBF-SVM) by +5 pp on every split, with either grid: the sphere encoding is the better kernel geometry for global data.
