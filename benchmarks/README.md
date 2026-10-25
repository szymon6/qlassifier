# Stage 1 benchmarks — the shipped protocol run (phase 9)

Up: PRD §6 targets · TDD §7 protocol · ADR 0004 · Evidence before this run: [phase-7 experiments](../experiments/classifier/README.md) · Paper tables read these files (`scripts/paper_tables.py`)

## Read first
- **[REPORT.md](REPORT.md)** — BT-1…BT-10: measured value, rule, status.
- **[summary.md](summary.md)** — per dataset: leaderboard (mean ± sd, corrected 95% CI, Δ vs Qlassifier), noise, cost.
- **[environment.json](environment.json)** — versions, machine, command, wall time per dataset.

## What is here
| Path | Content |
|---|---|
| `<dataset>/result.json` | the full `BenchmarkResult` (JSON = source of truth): per-seed runs, statistics, noise study, cost study |
| `<dataset>/runs.csv` | one row per (seed, model): chosen config, CV score, edge flag, BA, accuracy, ROC-AUC, F1, times |
| `<dataset>/noise.csv` | one row per noise cell: part, model, device, seed, accuracy vs exact, drop, kernel r, circuits, gates |
| `<dataset>/plots/` | `leaderboard`, `noise`, `cost` as PNG + PDF |
| `targets.json` | the report as data |
| `bt10/` | the BT-10 timing run: `qlassifier benchmark breast_cancer` with defaults (exact, no noise), on an idle machine |
| `golden_e01.py`, `golden_e01.txt` | the full golden check against phase-7 e01 (480 rows) and its result |
| `refingerprint.py` | rewrites `protocol.fingerprint` in the saved and shipped results after a fingerprint-definition change (run once in phase 9 when grid floats started entering the hash at 12 significant digits; only the hash string changed, every number untouched) |

## How it was produced
```bash
qlassifier benchmark breast_cancer --out benchmarks/bt10         # BT-10, idle machine
qlassifier benchmark breast_cancer ionosphere sonar heart --noise --out benchmarks   # 10 workers
qlassifier benchmark protein_angles --noise --out benchmarks --jobs 3                # 3 workers
python scripts/ship_results.py                                                                     # -> src/qlassifier/_results/
```
- Provenance, two runs on one 16 GB Apple M4 laptop (10 cores): the four UCI datasets come from the all-datasets run of commit `335e996` with 10 workers (it completed them and died on `protein_angles`); `protein_angles` comes from a separate run with `--jobs 3`. Reason: a 10-qubit noise cell (4,000 rows × 3 settings × 1,024 shots) peaks at 2.4 GB of RSS, so 10 workers exhausted the machine's memory twice (kernel watchdog panics); 3 workers stay under 8 GB (ADR 0006, memory). The worker count changes only wall time, never a number: every task is seeded by (dataset, seed, model) and the per-dataset results are independent. `environment.json` records both runs per dataset (`per_dataset`: wall time, workers, command). The report covers every `<dataset>/result.json` in this folder.
- The code of the two runs differs only in non-numeric ways (verified bit-identical device samples; `benchmarks/golden_e01.py` for the leaderboard): the protein run reads device samples as counts instead of per-shot memory and keeps one Gram pair at a time while tuning.
- Protocol: seeds 0–9 on every dataset; same splits, same per-fold preprocessing and features for every model; 30 configs per tuned model (pinned ZZ 4); 5-fold CV on balanced accuracy; corrected resampled t (Nadeau–Bengio) for every CI and paired test.
- Before this run the engine passed the golden check: **all 480 rows of the phase-7 reference leaderboard (e01) reproduced exactly** (chosen config, edge flag, every metric within 1e-9): [`golden_e01.py`](golden_e01.py) → [`golden_e01.txt`](golden_e01.txt). CI keeps a 21-row subset (`tests/integration/test_golden_e01.py`).
- Noise settings: 1,024 shots, optimisation level 3, transpiler seed 0, sampler seed 11 (phase-7 values); part A = Qlassifier on every dataset (seeds 0–4), part B = Breast Cancer paired 60/40 subset (seeds 0–2) with FQK-bw linear, PQK and separable.
- Shipped cache: `src/qlassifier/_results/<dataset>.json` are these files marked `source: "shipped"`; a contract test blocks stale ones (fingerprint + version).
