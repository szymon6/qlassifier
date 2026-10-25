# Stage 2 research loop — experiments

Up: PRD (targets BT-1…BT-6, §6) · Log: [learning-log.md](learning-log.md) · Research: prior art, novelty · Data: [data/README.md](data/README.md) · Rules: chellange.md, principles.md, workflow · Stage 1: [protocol §2](../classifier/README.md#2-protocol-fixed-before-iteration-1), [learning log](../classifier/learning-log.md), TDD §7 · Next: phase 8 TDD (adopts §2 as the stage-2 benchmark protocol)

## 0. Final decision
Full entry: [learning-log.md → Final decision](learning-log.md#final-decision--stage-2-method-and-evidence-against-every-target). One iteration covered every target (every target has evidence; BT-5 is a recorded miss); the stop rule did not fire.

**Method unchanged from stage 1** (Bloch-kernel core + geometry mode, self-tuning on exact features). Stage 2 adds evidence and infrastructure: two new datasets, the noise-scale sweep, the QPU-time model, the real-device point, the shots rule.

| BT | Measured | Status |
|---|---|---|
| BT-1 | spambase 0.911 > ZZ 0.899 (+1.2 pp, 10/10), ≥ published 0.888; banknote 1.000 > ZZ 0.955 (+4.4, 10/10), ≥ 0.921; haberman 0.564 reported (< published 0.586) | met (2 of 3) |
| BT-2 | drop at 1× +0.12 pp (flat 0×–3×); AUC 0.900 vs FQK-bw 0.738 vs ZZ 0.625 | met |
| BT-3 | `ibm_fez` job `dbesrqnpc3tc739mb92g`, 82 QPU s, accuracy 0.900 vs predicted 0.880 ± 0.010 (+2.0 pp, inside ± 2 pp), BA 0.907 | met |
| BT-4 | counts = executed (1,707 circuits, 0 two-qubit gates); model brackets 0.271 / 0.319 / 0.691 s per circuit; hardware 0.273 s/circuit inside 0.264–0.365 | met |
| BT-5 | beijing −0.6 [−2.7, +1.4] vs RBF-SVM; solar top mean, +0.5 [−0.4, +1.4]; gate not met | missed (stretch) |
| BT-6 | 240 / 240 stage-1 rows reproduced; statuses unchanged | baseline holds |

## 1. Setup
- Machine: Apple M4, 10 cores, 16 GB. Repo `.venv`: Python 3.12.0, qiskit 2.6.1, qiskit-aer 0.18.0, qiskit-ibm-runtime 0.51.0, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0. No new runtime dependency.
- Method under test: the shipped `qlassifier` 0.1.0 (`src/qlassifier/`, commit 7b058c3): `Qlassifier`, `backends`, `bench` (protocol, models, noise, cost, stats). Experiments call the shipped engine; nothing in `src/` changes in this phase.
- Harness (this stage): [`common/extra_datasets.py`](common/extra_datasets.py) (registers unbundled datasets into the shipped engine), [`common/run_bench.py`](common/run_bench.py) (shipped protocol on any registered dataset: same grids, splits, statistics, output files as `qlassifier benchmark`), [`common/noise_sweep.py`](common/noise_sweep.py) (scaled FakeFez noise models and sweep cells), [`common/cost_model.py`](common/cost_model.py) (QPU-time model). Stage-1 harness (`../../01-winning-classifier/experiments/common/`) is reused only through its saved per-seed results.
- Memory rule: peak RSS of one task is measured before any parallel run ([e00](e00-setup-checks/README.md)); device-noise cells run with ≤ 3 workers, `OMP_NUM_THREADS=1`, under `.cache/run_guarded.sh`; logs in UTC in `.cache/`.
- No IBM token in this environment (`QISKIT_IBM_*` unset): BT-3 is recorded as not run (PRD R-1); §2.7 fixes the protocol a token holder would run.

## 2. Protocol (fixed before iteration 1)
Stage 1 §2 is adopted unchanged (splits, seed blocks, preprocessing recipes, 30-config equal-effort tuning, model set, metrics, corrected statistics, noise settings, cost counts). Only the additions below are new. Follows the fair-comparison rules in chellange.md.

### 2.1 Adopted from stage 1 (restated)
- Splits: `StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)`; grouped data: `GroupShuffleSplit` by source unit + inner `StratifiedGroupKFold(5)`; drift rule > 3 pp → `stratified_grouped`. Same split for every model.
- Seed blocks: **dev 100–109** (exploration, choices), **eval 0–9** (every reported number, every BT check; one run, no re-selection), **fresh 200–209 / 300–309** (confirmation of anything chosen with data).
- Recipes: `pca8` (StandardScaler → PCA(8) → StandardScaler, fit per fold; skipped when d ≤ 8) and `native` (raw columns, model-internal scaling fit on training rows). Same features for every model.
- Tuning: 5-fold stratified CV on balanced accuracy, 30 configs per tuned model (pinned ZZ: 4), ridge grid m ∈ {1/8…4} × C̃ ∈ {0.1…1,000}, C = C̃/m²; ties → first in grid order; grid-edge share recorded.
- Models: `trivial`, `logistic_regression`, `rbf_svm`, `random_forest`, `gradient_boosting`, `mlp` (every baseline in chellange.md), `zz_qsvc` (pinned standard QML baseline), `fqk_bandwidth_full`, `fqk_bandwidth_linear`, `pqk`, `separable` (published quantum), `qlassifier`, `classical_twin` (control, geometry datasets only). Definitions: TDD §7.4.
- Metrics: BA (headline), accuracy, ROC-AUC, F1; tune / fit / predict time; qubits, circuits, two-qubit gates, depth.
- Statistics: corrected resampled t (Nadeau & Bengio 2003), SE = sd·√(1/J + n_te/n_tr), 95% CI; best classical picked ex post; ties only with ≥ 5 seeds; one Δ convention everywhere: **Qlassifier − model**, in pp.
- Noise: `AerSimulator.from_backend(Fake…)`, opt level 3, `seed_transpiler=0`, sampler seed 11, 1,024 shots; hyperparameters from exact-mode tuning of the same split; no re-tuning on noisy data; per-qubit simulation of product circuits where exact (ADR 0006).

### 2.2 Dataset entry rule (stage 2)
A dataset enters only with: source URL, licence (CC BY 4.0 / CC0 / public domain), citation, SHA-256 of the downloaded file and of every derived table, loader rule (label, dropped rows / columns, groups), recipe and geometry. All of it is written in [data/README.md](data/README.md) **before any model run** on that dataset. Downloaded files are data: stored under `data/downloads/<name>/`, never executed; derived tables are built by `python -I` scripts kept in `common/`. Every screened dataset is reported, wins and losses.

| Id | Rows × features | Positive class (label 1) | Licence | Recipe / split | Role | Published quantum BA (cross-protocol) |
|---|---|---|---|---|---|---|
| `spambase` | 4,601 × 57 | spam (1,813) | CC BY 4.0 (UCI 94) | `pca8`, stratified | BT-1 candidate | 0.888 (Belis fixed map, 1,000-row subsample; RBF 0.903) arXiv:2604.18837 T5 |
| `banknote` | 1,372 × 4 | forged (610) | CC BY 4.0 (UCI 267) | `pca8` (d = 4 → no PCA), stratified | BT-1 candidate | 0.921 (Rot2DoF; QKT 0.871; RBF 1.000) arXiv:2604.18837 T5 |
| `haberman` | 306 × 3 | died within 5 years (81) | CC BY 4.0 (UCI 43) | `pca8` (d = 3 → no PCA), stratified | BT-1 candidate | 0.581 (Sakhnenko10; QKT 0.586; RBF 0.549: the one dataset where arXiv:2604.18837 reports a quantum lead) |
| `beijing_pm25` | 4,000 × 8 (label-free sample of 34,230 hourly rows, Dongsi station) | PM2.5 > 75 µg/m³ | CC BY 4.0 (UCI 501) | `native`; geometry {wind direction: 360°, hour: 24, month: 12} + 5 plain columns; **week-grouped** splits | BT-5 candidate | none found |
| `solar_boulder` | 4,000 × 2 (label-free sample of 26,266 hourly rows, USCRN Boulder CO 2021–2023) | solar radiation > 200 W/m² | U.S. public domain (NOAA NCEI) | `native`; geometry {hour: 24, day of year: 365.25}; **day-grouped** splits | BT-5 candidate | none found |
| `deep_quakes` (stage 1 file) | 3,729 × 2 | depth ≥ 70 km (795) | U.S. public domain (USGS) | `native`; geometry (lat, lon) = sphere; stratified | shots hypothesis (e05) | none |
| `breast_cancer`, `heart` (bundled) | stage 1 | stage 1 | stage 1 | stage 1 | BT-2 (BC), BT-4 (BC), BT-6 (BC, heart) | stage 1 |

- BT-1 candidates are all three UCI sets. Rule fixed now: each runs **once on eval seeds 0–9** with the full model set and cost study; BT-1 is met when ≥ 2 of the 3 satisfy both conditions (mean BA > `zz_qsvc` with paired win rate ≥ 0.8; mean BA ≥ published quantum BA). All three are reported whatever they show. Published numbers come from a different protocol (5-fold CV, no bandwidth, subsamples) and are context, as in stage 1 BT-3.
- BT-5 candidates: screened on **dev seeds 100–104** (full model set, 10 models + twin). Confirmation on **fresh seeds 200–209** only when the dev Δ vs the best classical model (mean BA) is ≥ +1.0 pp; otherwise the dataset is recorded as a tie / loss and not re-run. Confirmation pass = mean BA above every classical model and corrected 95% CI of Δ vs the best classical > 0 (one test per confirmed dataset; k candidates → α = 0.05/k). No changes after results.

### 2.3 Noise-scaling definition (BT-2)
- Base: Aer's device model of `FakeFez` (Heron r2): `NoiseModel.from_backend(FakeFez())`: per-qubit gate errors (depolarizing + thermal relaxation from T1, T2 and gate duration), per-qubit readout errors; 1,132 error entries; no crosstalk, no idle noise in our circuits.
- Scale s ∈ {0, 0.5, 1, 1.5, 2, 3} acts on the **device Target**, then Aer rebuilds the model: every instruction error e → min(s·e, 0.999) (gate errors and readout assignment errors), every T1 → T1/s and T2 → T2/s (s > 0). Effect on the Aer channels (checked in [e00](e00-setup-checks/README.md)): the non-identity probability of every depolarizing channel and both readout flip probabilities scale exactly by s; thermal relaxation probabilities scale by ≈ s for the short gates used (duration ≪ T1). s = 0 is an ideal sampler (shot noise only); s = 1 is the stage-1 device model; s = 3 is a device three times worse in every published error rate.
- Layout is fixed at s = 1: circuits are transpiled once on the unscaled `FakeFez` (opt level 3, seed 0; Qlassifier pinned layout as in `backends.transpile_pinned`), so every scale runs the same physical qubits and the same gates.
- Shots 1,024; sampler seed 11; Aer `density_matrix` method (the method Aer picks for these noise models; checked in e00). Qlassifier cells are simulated qubit by qubit (exact for product circuits and local noise, ADR 0006 amendment); fidelity-kernel cells simulate the whole 8-qubit register.
- Dataset: Breast Cancer, `pca8`. Part A: Qlassifier on all rows, eval seeds 0–4, every scale (exact-tuned (g, C) of the seed). Part B (paired): the stage-1 stratified 60 train / 40 test subset of eval seeds 0–2; Qlassifier and `fqk_bandwidth_linear` (the bandwidth-tuned fidelity kernel of the stage-1 noise study) at every scale; `zz_qsvc` (pinned: ZZ full, reps 2, MinMax [0, π], C ∈ {0.1, 1, 10, 100}) at s ∈ {0, 1, 3} on seeds 0–1 only (18 min of density-matrix simulation per cell: 433 CZ, depth 861; e00) — its curve is linearly interpolated between measured scales. Each model is tuned in exact mode on the subset's training rows (same `_subset_config` routine as the shipped part B).
- Reported per (part, model, s): accuracy and BA (mean over seeds), drop vs exact in pp (mean, worst seed), kernel Pearson r vs exact (training block), circuits, two-qubit gates, depth.
- **AUC** = trapezoid integral of the mean accuracy over s ∈ [0, 3] divided by 3 (= sweep-mean accuracy; range 0–1). For ZZ, the trapezoid uses its three measured scales. Also reported on BA.
- **BT-2 test (pre-registered):** part A mean accuracy drop at s = 1 ≤ 2 pp (shipped part-A convention, 569 rows), and part-B AUC(Qlassifier) > AUC(FQK-bw linear) and > AUC(ZZ QSVC) on the paired subset. Stretch report: drop at s = 3.

### 2.4 Cost-model definition (BT-4)
- Counts (exact): circuits = 3·(N_fit + N_predict) for Qlassifier and PQK; N_tr(N_tr − 1)/2 + N_te·N_tr for fidelity kernels (training diagonal skipped); on-device tuning circuits as in stage 1 §2.9; shots per circuit = `shots` (1,024 default); two-qubit gates and depth per circuit after transpilation to the chosen device (opt level 3; median over transpiler seeds 0–2 for the cost table; the executed circuits' own counts in noise records). Circuit durations: `QuantumCircuit.estimate_duration(backend.target)` of the transpiled circuit (seconds).
- QPU-time model (seconds), per IBM's published job-time model ([estimate-job-run-time](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time): usage ≈ per-sub-job overhead + (rep_delay + circuit length) × executions; default `rep_delay` = `backend.default_rep_delay` = 250 µs; ≈ 2 s overhead per sub-job; quick form 2 + 0.00035 × executions):
  - **low** = Σ_circuits shots × (rep_delay + duration) + 2 s × ⌈circuits / 300⌉ (300 circuits per job: arXiv:2604.18837 job hygiene),
  - **high** = Σ_circuits shots × 350 µs + 2 s × ⌈circuits / 300⌉ (IBM's quick per-execution figure, which includes typical circuit length and mitigation overheads).
  - Both use the device's `default_rep_delay` and `dt` read from the fake backend's configuration; nothing is fitted to data.
- **Published per-circuit timings (the BT-4 range):** 0.27 s/circuit at 1,024 shots (244 s QPU for 900 circuits, ibm_fez; arXiv:2607.20377), 0.32 s/circuit at 1,024 shots (584 s for 1,830 circuits, ibm_fez; arXiv:2604.18837 `qpu_budget_analysis.py`), 0.69 s/circuit at 2,048 shots (19.3 min for 1,675 circuits, ibm_pittsburgh; arXiv:2605.14672). Sources audited in `docs/references/rea/scalable-robust-kernels/05-technical-observations.md` and `…/kakavand-2026-qsvm-benchmark/05-technical-observations.md`.
- **BT-4 test (pre-registered):** (i) the model's circuit, shot and two-qubit-gate counts equal the counts of circuits actually executed on a fake device by the shipped `Qlassifier(backend="fake_fez")` (`resources_`) and by the noise cells; (ii) for each published measurement, the per-circuit time at its shot count lies inside [low, high] of the model (kernel-circuit durations from the published circuits' depth class: 4–8-qubit ZZ compute–uncompute).

### 2.5 Hardware point (BT-3)
- PRD amended 2026-10-25 (commit 69f4002): a token exists (IBM Open plan, 10 QPU min/month); BT-3 is required and the paper documents the run in full. Stage-2 hard cap: ≤ 3 QPU minutes billed. The run starts only on the main agent's explicit GO after the dry run is written up ([e07](e07-hardware-point/README.md)). Credentials: `QiskitRuntimeService()` defaults only; no token is read, printed or committed.
- Protocol (phase 9 ships it as `qlassifier benchmark --hardware`): Breast Cancer, `pca8`, eval seed 0, the part-B stratified 60 train / 40 test subset; Qlassifier tuned in exact mode on the 60 training rows (0 circuits); Bloch readout of all 100 rows (train + test) on one Heron device (`ibm_fez` preferred: twin of FakeFez; `ibm_kingston` if fez is down or has > 50 pending jobs) with `SamplerV2`, 1,024 shots, no DD or twirling, the shipped pinned transpilation (opt level 3, seed 0), 3 PUBs (X, Y, Z settings, 100 parameter sets each) = 300 circuits in one job; unit renormalisation; Gaussian kernel and SVM exactly as in simulation. Two analyses from one job: variant 1 (exact training features + hardware test features: the product's inference mode) and variant 2 (hardware features for train and test). Cost by §2.4: 300 × 1,024 × (250 + 1.6) µs + 2 s ≈ 79 s (low) … 110 s (high) of QPU time (inference-only readout: 120 circuits, 33–45 s).
- Pass rule: hardware accuracy within ± 2 pp of the FakeFez prediction for the same rows (pre-registered in e07: 0.880 ± 0.010 over 20 shot-noise draws → 35 or 36 of 40 test rows); kill: ≤ 34 or ≥ 37 rows. Also reported: BA, kernel r vs exact, mean Bloch error, job id, UTC timestamps, `job.usage()` seconds (compared with the §2.4 model), calibration snapshot of the used qubits, transpiled circuits, raw counts. Provenance files live in `e07-hardware-point/run-seed0/`; phase 9 copies them to `benchmarks/hardware/`.

### 2.6 Stage-1 regression check (BT-6)
- `qlassifier benchmark breast_cancer` and `qlassifier benchmark heart` (exact, defaults, seeds 0–9) on the current `main`, compared row by row with the stage-1 shipped `benchmarks/<dataset>/runs.csv`: chosen config, edge flag and every metric equal (|Δ| ≤ 1e-9; one ROC-AUC resolution step allowed off the reference platform, TDD §7.13). The stage-1 target table (`benchmarks/targets.json`) is re-read: all ten statuses unchanged. Phase 9 re-runs the full `--noise` command after the stage-2 changes.

### 2.7 Targets → measurements
| ID | Measurement under this protocol | Experiment |
|---|---|---|
| BT-1 | §2.2 rule on `spambase`, `banknote`, `haberman`, eval seeds 0–9 | [e01](e01-new-datasets/README.md) |
| BT-2 | §2.3 sweep, Breast Cancer | [e02](e02-noise-sweep/README.md) |
| BT-3 | §2.5: one job, ≤ 3 QPU min, pre-registered prediction | [e07](e07-hardware-point/README.md) |
| BT-4 | §2.4: counts exact; model brackets the 3 published timings | [e03](e03-cost-model/README.md) |
| BT-5 | §2.2 screen → confirm on `beijing_pm25`, `solar_boulder`; quake shots hypothesis | [e04](e04-geometry-screen/README.md), [e05](e05-quake-shots/README.md) |
| BT-6 | §2.6 | [e06](e06-regression/README.md) |

### 2.8 Time budget
≤ 5 h wall-clock of experiments, ≤ 60 min per experiment. Plan: e01 ≈ 35 min, e02 ≈ 60 min, e03 ≈ 5 min, e04 ≈ 25 (+ 20) min, e05 ≈ 10 min, e06 ≈ 5 min, e07 ≈ 5 min local + queue. Stop rule: every target has evidence, or 3 iterations without measurable gain, or the budget is spent.

## 3. Experiments
| Id | Question | Key result | Decision |
|---|---|---|---|
| [e00](e00-setup-checks/README.md) | Does the scaled noise model scale every channel by s? Cell costs and RSS? | readout and depolarizing probabilities ×s exactly (s = 0.5, 2); per-qubit BC cell 1–3 s; FQK-bw linear 60/40 cell 3.2 min; ZZ full cell 18 min; RSS ≤ 1.6 GB | sweep as §2.3 |
| [e01](e01-new-datasets/README.md) | BT-1 on three new UCI sets | spambase 0.911 (> ZZ 0.899, 10/10; > published 0.888; ties GB, below MLP −0.7 pp); banknote 1.000 (> ZZ 0.955; > published 0.921); haberman 0.564 (ZZ 0.546, 6/10; < published 0.586; everyone ties) | BT-1 met (2 of 3); bundle spambase + banknote |
| [e02](e02-noise-sweep/README.md) | BT-2 noise sweep 0×–3× | part A drop at s = 1: +0.12 pp (worst seed +0.58), identical from s = 0 to s = 3 (0.949 vs exact 0.950); part B AUC accuracy Qlassifier 0.900 vs FQK-bw linear 0.738 vs ZZ QSVC 0.625 (BA 0.889 / 0.700 / 0.500); FQK-bw already −19 pp at s = 0 (shot noise), −22 pp at s = 3; ZZ at the majority level throughout (kernel r 0.92 → 0.38 → 0.02) | **BT-2 met** (H2) |
| [e03](e03-cost-model/README.md) | BT-4 cost model | counts = executed circuits (1,707 = 3 × 569, 0 two-qubit gates); IBM job model with default rep_delay 250 µs brackets all three published timings: 0.271 ∈ [0.267, 0.365], 0.319 ∈ [0.268, 0.366], 0.691 ∈ [0.537, 0.724] s/circuit; BC fit + predict 452–624 QPU s vs 11–15 h for fidelity kernels | BT-4 met (H3) |
| [e04](e04-geometry-screen/README.md) | BT-5 geometry screen (dev seeds 100–104, grouped splits) | `beijing_pm25` 0.755 vs RBF-SVM 0.761 (−0.6 [−2.7, +1.4]); `solar_boulder` 0.917 top mean vs RBF-SVM 0.912 (+0.5 [−0.4, +1.4]), +2.3–2.6 pp over every quantum baseline, ZZ / LR at 0.500 | both below the +1.0 pp gate → no confirmation; BT-5 missed (ties) |
| [e05](e05-quake-shots/README.md) | Narrow kernels need more shots? | `deep_quakes` BA drop on FakeFez: 10.4 pp (1,024 shots) → 5.8 (4,096) → 1.8 (16,384) → 0.65 pp (65,536); exact 0.766 vs RF 0.831 on the same seeds | H6 confirmed: shot resolution, not device error; loss to trees unchanged |
| [e07](e07-hardware-point/README.md) | BT-3 hardware point | `ibm_fez`, job `dbesrqnpc3tc739mb92g`, 300 circuits × 1,024 shots, **82 QPU s** (model 79–110); hardware accuracy 0.900 (36/40) vs FakeFez prediction 0.880 ± 0.010 → +2.0 pp, inside the ± 2 pp rule and the shot-noise range; BA 0.907; kernel r 0.998; Bloch error 0.020 (sim 0.018) | **BT-3 met** (H8) |
| [e06](e06-regression/README.md) | BT-6 regression | 240 / 240 stage-1 rows (BC, heart) reproduced exactly (max |Δ| 0.0, configs and edge flags equal); stage-1 statuses unchanged | baseline holds (H7); phase 9 re-runs `--noise` |

## 4. Failed ideas
Details in [learning-log.md → Failed ideas](learning-log.md#failed-ideas-kept).

| Idea | Why it failed | Evidence |
|---|---|---|
| Geometry mode on Beijing PM2.5 (3 circles + 5 plain columns, week-grouped) | −0.6 pp vs RBF-SVM; 0.9 pp below its classical twin; pilot lead gone at N = 4,000 | [e04](e04-geometry-screen/README.md) |
| Geometry mode on the solar torus (hour × day of year) | top mean BA, beats every quantum baseline, but only +0.5 pp over tuned RBF-SVM (seams far from the boundary) | [e04](e04-geometry-screen/README.md) |
| Haberman as a BT-1 dataset | all models tie (0.546–0.583); published quantum lead not reproducible under equal tuning | [e01](e01-new-datasets/README.md) |
| More shots to rescue `deep_quakes` | device loss gone at 65,536 shots (0.65 pp) but the 6.5 pp loss to RF stays; ≈ 15 QPU h | [e05](e05-quake-shots/README.md) |
