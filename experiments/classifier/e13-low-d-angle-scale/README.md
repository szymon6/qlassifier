# e13 — core angle scale below 8 plain columns (phase 9)

Up: [../README.md](../README.md) · Code: [`run.py`](run.py) · Raw: [`results.csv`](results.csv) · Log: [learning-log → phase-9 addendum](../learning-log.md#phase-9-addendum--angle-scale-below-8-plain-columns-e13) · Decision: ADR 0001 amendment

- **Trigger:** scikit-learn's `check_classifiers_train` (200 rows, 2 standardised features, 2 well-separated blobs) failed: Qlassifier training accuracy 0.82 vs the required 0.83 (RBF-SVM 0.97).
- **Cause (mechanism):** c₀ = 2/√d grows as d shrinks (1.41 at d = 2). Standardised values beyond ≈ 1.1σ then rotate past π/2, and RX·RY folds distant points onto nearby Bloch vectors (the map is injective only for |c₀x| < π/2). At d = 8 (the benchmark's PCA-8) c₀ = 0.71 keeps ≈ 97% of values inside π/2.
- **Hypothesis:** c₀ = 2/√max(d, 8) removes the fold at small d and changes nothing at d ≥ 8. Kill: no gain at d < 8, or any change at d = 8.
- **Setup:** 5-fold stratified CV (seed 0), default self-tuning; tasks: the scikit-learn blobs check (training accuracy), moons, circles (400 rows, synthetic mechanism tests) and Heart reduced to PCA-2/4/6/8 (real data); default RBF-SVM as reference. Run: `.venv/bin/python experiments/classifier/e13-low-d-angle-scale/run.py`.

## Result (accuracy)
| Task | c₀ = 2/√d (phase 7) | c₀ = 2/√max(d, 8) (shipped) | RBF-SVM (default) |
|---|---|---|---|
| blobs, training (sklearn check) | 0.820 | **0.990** | 0.970 |
| moons | 0.920 | **0.950** | 0.940 |
| circles | 0.927 | **0.997** | 1.000 |
| Heart PCA-2 | 0.788 | **0.821** | 0.838 |
| Heart PCA-4 | 0.818 | **0.825** | 0.835 |
| Heart PCA-6 | 0.821 | **0.828** | 0.845 |
| Heart PCA-8 | 0.828 | 0.828 (identical) | 0.838 |

## Decision
- **Keep the cap** (`MIN_SCALE_DIM = 8` in `src/qlassifier/encoding.py`): +0.7 to +17 pp at d < 8, identical at d ≥ 8.
- Benchmark numbers are unaffected: the protocol feeds 8 PCA features to the UCI sets and `protein_angles` has no plain columns (geometry mode). The golden e01 test still reproduces every row exactly.
