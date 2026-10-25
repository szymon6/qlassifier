# Benchmark report — targets BT-1…BT-10

## Read this first
- breast_cancer: Qlassifier BA 0.948, +37.6 pp vs the standard quantum baseline (ZZ QSVC), -1.3 pp vs the best classical model (mlp: a statistical tie).
- Leaderboards with confidence intervals, noise and cost, per dataset: [summary.md](summary.md). Per-seed rows: `<dataset>/runs.csv`; plots: `<dataset>/plots/`.
- Below: the stage's ten benchmark targets (BT-1…BT-10), each with its measured value, pass rule and verdict. A target this run could not measure says what it needs.

Rerun: `qlassifier benchmark breast_cancer --out benchmarks/bt10` · Protocol: TDD §7 · Rules: PRD §6 / TDD §7.10 · Details: [summary.md](summary.md), [environment.json](environment.json), per-dataset `result.json`, `runs.csv`, `plots/`.

| BT | Measured | Rule | Status |
|---|---|---|---|
| BT-1 | 0.956 vs 0.677 → +27.9 pp | Breast Cancer: mean accuracy Qlassifier - ZZ QSVC >= 20 pp | **met** |
| BT-2 | breast_cancer +37.6 pp (10/10) | mean BA Qlassifier > ZZ QSVC on all 5 datasets; paired win rate >= 0.8 on each | **not measured** in this run: needs all 5 datasets |
| BT-3 | breast_cancer 0.948 (-2.0 pp) | mean BA >= published - 1 pp on >= 3 of 4 UCI sets (0.968 / 0.911 / 0.813 / 0.804) | **not measured** in this run: needs the 4 UCI datasets |
| BT-4 | breast_cancer 0.948 (-2.0 pp) | mean BA >= published on >= 2 of 4 UCI sets | **not measured** in this run: needs the 4 UCI datasets |
| BT-5 | — | mean BA over the 4 UCI sets: Qlassifier highest among the in-harness quantum models | **not measured** in this run: needs the 4 UCI datasets |
| BT-6 | breast_cancer vs mlp: -1.3 [-3.1, +0.5] pp (tie) | ≥ 1 dataset: mean BA above every tuned classical model and corrected 95% CI of Δ vs best classical > 0 (stretch; report-only, ADR 0004) | **report-only** |
| BT-7 | 1,707 vs 2,423,299 → 1,420× | Breast Cancer: >= 100× fewer circuits than the bandwidth-tuned fidelity-kernel pipeline (incl. its search) | **met** |
| BT-8 | breast_cancer: 4 qubits (core, ceil(8/2) = 4), 0 two-qubit gates | per circuit on FakeTorino / FakeFez / FakeBrisbane: core encoding qubits <= ceil(d/2); geometry mode: 1 qubit per circle or (lat, lon) pair + ceil(d_plain/2) (ADR 0005); 2Q gates <= n - 1 | **met** |
| BT-9 | — | Breast Cancer, 1,024 shots: mean accuracy drop <= 2 pp on each device (part A); < FQK-bw drop (part B) | **not measured** in this run: needs `breast_cancer --noise` |
| BT-10 | 56.2 s on Apple M4 (10 cores, macOS-27.0-arm64-arm-64bit; 1-min load 1.95 at start) | `qlassifier benchmark breast_cancer`, defaults, exact, Apple M-class laptop: wall <= 120 s | **met** |

- Datasets: breast_cancer; seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; versions qlassifier 0.1.0, qiskit 2.6.1, scikit-learn 1.9.1.
- Machine: Apple M4, 10 cores; wall per dataset (workers): breast_cancer 56 s (10).
- BT-6 is report-only in stage 1 (ADR 0004): the phase-7 pre-registered tests failed; nothing is upgraded here whatever these seeds show.
- BT-8 judges the core datasets by the PRD's ceil(d/2) qubits and the geometry-mode dataset (protein_angles) by the ADR 0005 rule: one qubit per circle or direction (approved 2026-10-25). Two-qubit gates: <= n - 1 everywhere.
