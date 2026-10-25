# e04 — BT-5 geometry screen: wind-direction and solar-torus data (dev seeds 100–104)

Up: [../README.md](../README.md) (protocol §2.2) · Log: [../learning-log.md](../learning-log.md) (H4, H5) · Code: [`run.sh`](run.sh), [`../common/run_bench.py`](../common/run_bench.py) · Data: [../data/README.md](../data/README.md) (`beijing_pm25_4k.csv`, `solar_boulder_4k.csv`, built label-free before any run) · Raw: `dev-<dataset>/result.json`, `runs.csv`, `summary.txt`, `plots/` · Stage 1: [learning log iteration 5](../../classifier/learning-log.md) (why BT-6 was missed), groundwork pilots in `bloch-sphere-angular-data.md` §5

## What
BT-5 (stretch): beat every tuned classical baseline on one more real dataset with a corrected 95% CI > 0. Stage 1 said the only route left is data whose label depends more strongly on geometry. Two candidates, 4,000 label-free rows each, leakage-safe grouped splits:
- `beijing_pm25` — PM2.5 > 75 µg/m³ from wind direction (circle, 360°), hour (24), month (12) and 5 plain columns (wind speed, temperature, pressure, dew point, rain); 6 qubits (3 circles + 3 pairs); splits grouped by ISO week (drift > 3 pp on these seeds → `stratified_grouped`).
- `solar_boulder` — solar radiation > 200 W/m² from (hour, day of year) only: a label that is a smooth function of the hour × day-of-year torus; 2 circle qubits; splits grouped by date.
Pre-registered gate: the dev Δ vs the best classical model (mean BA) must reach **+1.0 pp** before a confirmation run on fresh seeds 200–209; otherwise the dataset is recorded as a tie / loss and not re-run.

## How
`.cache/run_guarded.sh stage2-e04 bash experiments/hardware/e04-geometry-screen/run.sh` → the shipped engine on both datasets, dev seeds 100–104, 13 models (12 + classical twin), 5 workers. Wall 993 s (beijing) and 504 s (solar); peak RSS ≤ 1.0 GB.

## Result (dev seeds 100–104; BA; Δ = Qlassifier − model, pp, corrected 95% CI)
| Dataset | Qlassifier | Twin | Best classical | Δ vs best classical | Other classical | Published quantum / ZZ | Gate (≥ +1.0) |
|---|---|---|---|---|---|---|---|
| `beijing_pm25` | 0.755 ± 0.026 | 0.764 (−0.9 [−1.4, −0.3]) | RBF-SVM 0.761 | **−0.6 [−2.7, +1.4]** | MLP 0.760, GB 0.757, RF 0.740, LR 0.676 | PQK 0.758, FQK-bw 0.757 / 0.752, separable 0.753, ZZ 0.567 (+18.8) | fail |
| `solar_boulder` | **0.917 ± 0.005** (top mean) | 0.917 (identical Gram) | RBF-SVM 0.912 | **+0.5 [−0.4, +1.4]** | MLP 0.907 (+1.1), RF 0.901 (+1.7), GB 0.896 (+2.1), LR 0.500 | PQK 0.894 (+2.3 [+0.8, +3.9]), FQK-bw 0.893 (+2.4), separable 0.892 (+2.6), ZZ 0.500 (+41.7) | fail |

- Beijing: wind direction, hour and month as circles do not beat the plain encoding here: the classical twin (RBF on sin/cos + z-scored plain columns) is 0.9 pp above geometry mode (the twin z-scores the 5 plain columns; geometry mode puts them on Bloch pairs), and every tuned model from RBF-SVM to FQK-bw sits in 0.75–0.76. Compared with the groundwork pilot (+0.9 pp over GB at N = 1,200, random splits), week-grouped splits and N = 4,000 turn the lead into a 0.6 pp deficit.
- Solar: the torus geometry is the right prior — Qlassifier has the top mean BA, beats every other quantum model significantly (+2.3 to +2.6 pp) and LR / ZZ collapse to 0.500 (raw hour and day of year are not linearly separable) — but tuned RBF-SVM on the raw columns is only 0.5 pp behind (the label's seams at midnight and new year are far from the daylight boundary, so the plain RBF barely suffers). Below the +1.0 pp gate.

## Verdict
- **H4 and H5 killed by the pre-registered gate; no confirmation runs; BT-5 missed** (stretch). Both datasets are reported as ties: Qlassifier is top of the leaderboard on solar (mean) and mid-pack on Beijing, inside the CI of the best classical model on both.
- Lessons: (1) a smooth label on the torus gives the geometry prior an edge over every quantum baseline and over trees / MLP, but a tuned RBF on raw angles loses little unless the label straddles the seam; (2) mixing circles with several plain columns dilutes the prior (Beijing); (3) small-N pilot leads (+0.9 pp) do not survive grouped splits at N = 4,000, as in stage 1.
- Recommendation: keep `solar_boulder` as a teaching example candidate (2 qubits, LR / ZZ collapse, torus picture) if phase 8 wants a second geometry tile; do not bundle `beijing_pm25`.
