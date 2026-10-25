# e01 — BT-1 on three new UCI datasets (eval seeds 0–9)

Up: [../README.md](../README.md) (protocol §2.2) · Log: [../learning-log.md](../learning-log.md) (H1) · Code: [`run.sh`](run.sh), [`../common/run_bench.py`](../common/run_bench.py), [`analyze.py`](analyze.py) · Data: [../data/README.md](../data/README.md) · Raw: `<dataset>/result.json`, `runs.csv`, `summary.txt`, `plots/` · Table: [`bt1_table.md`](bt1_table.md)

## What
Do Qlassifier's stage-1 wins hold on real data it has never seen? Three UCI sets with published quantum numbers (arXiv:2604.18837), chosen and declared before any run: `spambase` (4,601 × 57, `pca8`), `banknote` (1,372 × 4, d ≤ 8 → 2 qubits), `haberman` (306 × 3, 2 qubits; the one public set where that paper reports a quantum lead). Rule (README §2.2): BT-1 met when ≥ 2 of 3 have mean BA > pinned ZZ with ≥ 80% paired wins **and** mean BA ≥ the published quantum BA. All three are reported.

## How
`.cache/run_guarded.sh stage2-e01 bash experiments/hardware/e01-new-datasets/run.sh`: the shipped engine (same splits, per-fold `pca8`, 30-config grids, corrected statistics, output files as `qlassifier benchmark`) on each dataset, eval seeds 0–9, 12 models + cost study, 10 workers. Wall: spambase 1,187 s (peak RSS 0.8 GB), banknote 78 s, haberman 115 s. Then `analyze.py`.

## Result
| Dataset | Qlassifier BA ± sd | ZZ QSVC BA | Δ vs ZZ [CI] (wins) | Published quantum BA | Qlassifier − published | Best classical (BA) | Δ vs best classical [CI] | Tied with best? | BT-1 |
|---|---|---|---|---|---|---|---|---|---|
| `spambase` | 0.911 ± 0.006 | 0.899 | +1.2 [+0.8, +1.6] (10/10) | 0.888 | +2.3 pp | gradient_boosting (0.918) | −0.8 [−2.0, +0.4] | yes | pass |
| `banknote` | 1.000 ± 0.001 | 0.955 | +4.4 [+2.7, +6.2] (10/10) | 0.921 | +7.9 pp | mlp (1.000) | +0.0 [−0.2, +0.2] | yes | pass |
| `haberman` | 0.564 ± 0.031 | 0.546 | +1.9 [−4.1, +7.9] (6/10) | 0.586 | −2.2 pp | mlp (0.583) | −1.8 [−8.1, +4.4] | yes | fail |

- **BT-1: 2 of 3 pass → met.**
- Spambase leaderboard (BA): GB 0.918, MLP 0.918, RF 0.915, **Qlassifier 0.911**, RBF-SVM 0.908, PQK 0.905, FQK-bw full 0.905, FQK-bw linear 0.902, ZZ 0.899, separable 0.897, LR 0.871. Qlassifier is the best quantum model and ties GB (the best classical) but is significantly below MLP (−0.7 [−1.2, −0.2] pp). The pinned ZZ baseline does **not** collapse here (0.899): word-frequency features squeezed to [0, π] stay near 0 for most columns, so the full-entanglement map is close to a product map on this data; the stage-1 margins (+23–38 pp) came from datasets where ZZ collapses.
- Banknote: every tuned model except ZZ, RF and LR reaches 0.997–1.000; Qlassifier 1.000 ± 0.001 with 2 qubits; ZZ 0.955 (its 4-qubit full-entanglement map still loses 4.4 pp). Published Rot2DoF 0.921 is 7.9 pp below.
- Haberman: all 11 tuned models sit between 0.546 and 0.583 (3 features, 81 positives, sd 3–6 pp); every pair ties, the majority class ties on accuracy. Qlassifier 0.564 is 2.2 pp below the published QKT 0.586 and 1.8 pp below MLP; ZZ 0.546. The published "quantum lead" (0.581 vs RBF 0.549) is inside our seed spread; under equal tuning the best model is an MLP at 0.583. Grid-edge shares: see `summary.txt`.
- Cost (seed 0, N_tr 3,220 / N_te 1,381): Qlassifier 13,803 circuits (4 qubits, 0 two-qubit gates) vs 9,629,410 per fidelity-kernel Gram (+ 149,266,320 on-device tuning circuits for FQK-bw / separable); banknote and haberman: 2 qubits, 0 two-qubit gates.

## Verdict
- H1 confirmed on spambase and banknote, killed on haberman (below published and below MLP; ties everyone). BT-1 met by its pre-registered rule; haberman is reported as a tie / miss.
- Recommendation for phase 8: bundle `spambase` and `banknote` (CC BY 4.0, published quantum numbers 0.888 / 0.921 as `PUBLISHED` entries); keep `haberman` out of the wheel but in the paper as the honest check of arXiv:2604.18837's one quantum lead. Note for the paper: the standard ZZ baseline is not always a collapsed baseline; report its number per dataset.
