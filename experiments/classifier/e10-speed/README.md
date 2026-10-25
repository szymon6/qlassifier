# e10 — speed proxy for BT-10 (Breast Cancer benchmark ≤ 2 min)

Up: [../README.md](../README.md) (protocol §2.9) · Runner: [`../common/run_x.py`](../common/run_x.py) · Raw: [`results.csv`](results.csv), [`time.txt`](time.txt), load: [`load_before.txt`](load_before.txt), [`load_after.txt`](load_after.txt) · Log: [learning-log](../learning-log.md) (iteration 3)

- **Question:** does the benchmark on Breast Cancer (defaults, exact mode, every baseline, 10 eval seeds) finish within 2 minutes on an Apple M-class laptop?
- **Setup:** Apple M4, 10 cores, near idle (load 2.6 at start). Protocol §2 (`pca8`, 30-config CV per tuned model). 11 models: trivial, LR, RBF-SVM, RF, GB, MLP, pinned ZZ, FQK-bw (linear), PQK, separable, Qlassifier core (`bloch_gauss`). One task per seed, 10 parallel jobs. Command: `run_x.py results.csv --datasets breast_cancer --seeds 0-9 --k 8 --models … --jobs 10`.

## Result
- **Wall time 61.5 s** (CPU 549 s). Results identical to [e01](../e01-reproduce/README.md) for the same seeds (max |ΔBA| = 0).
- MLP and RF tuning dominate (≈ 90% of CPU); every quantum model together < 2 s per split.
- **BT-10 met in proxy** (≤ 120 s). Phase 9 re-measures the shipped `qlassifier benchmark` command; finer task scheduling (split × model) can cut wall time further.
