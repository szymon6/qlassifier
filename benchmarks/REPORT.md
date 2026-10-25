# Benchmark report — targets BT-1…BT-10

## Read this first
- breast_cancer: Qlassifier BA 0.948, +37.6 pp vs the standard quantum baseline (ZZ QSVC), -1.3 pp vs the best classical model (mlp: a statistical tie); ionosphere: Qlassifier BA 0.925, +22.6 pp vs the standard quantum baseline (ZZ QSVC), -0.2 pp vs the best classical model (rbf_svm: a statistical tie); sonar: Qlassifier BA 0.824, +27.9 pp vs the standard quantum baseline (ZZ QSVC), +0.5 pp vs the best classical model (mlp: a statistical tie); heart: Qlassifier BA 0.816, +28.3 pp vs the standard quantum baseline (ZZ QSVC), -1.0 pp vs the best classical model (logistic_regression: a statistical tie); protein_angles: Qlassifier BA 0.898, +35.5 pp vs the standard quantum baseline (ZZ QSVC), +1.0 pp vs the best classical model (gradient_boosting: a statistical tie).
- Leaderboards with confidence intervals, noise and cost, per dataset: [summary.md](summary.md). Per-seed rows: `<dataset>/runs.csv`; plots: `<dataset>/plots/`.
- Below: the stage's ten benchmark targets (BT-1…BT-10), each with its measured value, pass rule and verdict. A target this run could not measure says what it needs.

Rerun: `qlassifier benchmark --noise --out benchmarks` (breast_cancer, ionosphere, sonar, heart) and `qlassifier benchmark protein_angles --noise --out benchmarks --jobs 3` (protein_angles) · Protocol: TDD §7 · Rules: PRD §6 / TDD §7.10 · Details: [summary.md](summary.md), [environment.json](environment.json), per-dataset `result.json`, `runs.csv`, `plots/`.

| BT | Measured | Rule | Status |
|---|---|---|---|
| BT-1 | 0.956 vs 0.677 → +27.9 pp | Breast Cancer: mean accuracy Qlassifier - ZZ QSVC >= 20 pp | **met** |
| BT-2 | breast_cancer +37.6 pp (10/10); ionosphere +22.6 pp (10/10); sonar +27.9 pp (10/10); heart +28.3 pp (10/10); protein_angles +35.5 pp (10/10) | mean BA Qlassifier > ZZ QSVC on all 5 datasets; paired win rate >= 0.8 on each | **met** |
| BT-3 | breast_cancer 0.948 (-2.0 pp); ionosphere 0.925 (+1.4 pp); sonar 0.824 (+1.1 pp); heart 0.816 (+1.2 pp) → 3/4 | mean BA >= published - 1 pp on >= 3 of 4 UCI sets (0.968 / 0.911 / 0.813 / 0.804) | **met** |
| BT-4 | breast_cancer 0.948 (-2.0 pp); ionosphere 0.925 (+1.4 pp); sonar 0.824 (+1.1 pp); heart 0.816 (+1.2 pp) → 3/4 | mean BA >= published on >= 2 of 4 UCI sets | **met** |
| BT-5 | qlassifier 0.878, separable 0.870, fqk_bandwidth_full 0.864, pqk 0.862, fqk_bandwidth_linear 0.856, zz_qsvc 0.587 | mean BA over the 4 UCI sets: Qlassifier highest among the in-harness quantum models | **met** |
| BT-6 | breast_cancer vs mlp: -1.3 [-3.1, +0.5] pp (tie); ionosphere vs rbf_svm: -0.2 [-3.1, +2.7] pp (tie); sonar vs mlp: +0.5 [-8.5, +9.5] pp (tie); heart vs logistic_regression: -1.0 [-3.1, +1.2] pp (tie); protein_angles vs gradient_boosting: +1.0 [-0.2, +2.2] pp (tie) | ≥ 1 dataset: mean BA above every tuned classical model and corrected 95% CI of Δ vs best classical > 0 (stretch; report-only, ADR 0004) | **report-only** |
| BT-7 | 1,707 vs 2,423,299 → 1,420× | Breast Cancer: >= 100× fewer circuits than the bandwidth-tuned fidelity-kernel pipeline (incl. its search) | **met** |
| BT-8 | breast_cancer: 4 qubits (core, ceil(8/2) = 4), 0 two-qubit gates; ionosphere: 4 qubits (core, ceil(8/2) = 4), 0 two-qubit gates; sonar: 4 qubits (core, ceil(8/2) = 4), 0 two-qubit gates; heart: 4 qubits (core, ceil(8/2) = 4), 0 two-qubit gates; protein_angles: 10 qubits (geometry, ADR 0005 limit 10), 0 two-qubit gates | per circuit on FakeTorino / FakeFez / FakeBrisbane: core encoding qubits <= ceil(d/2); geometry mode: 1 qubit per circle or (lat, lon) pair + ceil(d_plain/2) (ADR 0005); 2Q gates <= n - 1 | **met** |
| BT-9 | part A drop 0.12 / 0.12 / 0.12 pp (Torino / Fez / Brisbane); part B Qlassifier 0.8 / 0.8 / 0.8 vs FQK-bw linear 22.5 / 20.0 / 17.5 pp | Breast Cancer, 1,024 shots: mean accuracy drop <= 2 pp on each device (part A); < FQK-bw drop (part B) | **met** |
| BT-10 | 56.2 s on Apple M4 (10 cores, macOS-27.0-arm64-arm-64bit; 1-min load 1.95 at start) | `qlassifier benchmark breast_cancer`, defaults, exact, Apple M-class laptop: wall <= 120 s | **met** |

- Datasets: breast_cancer, ionosphere, sonar, heart, protein_angles; seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; versions qlassifier 0.1.0, qiskit 2.6.1, scikit-learn 1.9.1.
- Machine: Apple M4, 10 cores; wall per dataset (workers): breast_cancer 569 s (10), ionosphere 81 s (10), sonar 78 s (10), heart 94 s (10), protein_angles 1685 s (3).
- BT-6 is report-only in stage 1 (ADR 0004): the phase-7 pre-registered tests failed; nothing is upgraded here whatever these seeds show.
- BT-8 judges the core datasets by the PRD's ceil(d/2) qubits and the geometry-mode dataset (protein_angles) by the ADR 0005 rule: one qubit per circle or direction (approved 2026-10-25). Two-qubit gates: <= n - 1 everywhere.
