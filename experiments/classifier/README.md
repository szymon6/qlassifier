# Stage 1 research loop — experiments

Up: PRD (targets BT-1…BT-10, §6) · Log: [learning-log.md](learning-log.md) · Research: prior art, theory · Rules: chellange.md, principles.md, workflow · Next: phase 8 TDD (adopts §2 as its benchmark protocol: TDD §7)

## 0. Final decision
Full entry: [learning-log.md → Final decision](learning-log.md#final-decision--the-qlassifier-method-stage-1). Loop closed after 5 iterations (stop rule: iterations 3–5 without gain on the open target). BT-6 is a recorded miss under the strict (chain-grouped) protocol; every other target is met.

**Chosen method: Qlassifier = Bloch-kernel core + opt-in geometry mode.**
- Core: 2 standardised features per qubit, RX(c₀x_a)·RY(c₀x_b) from |0⟩, c₀ = 2/√d fixed; 3 Pauli settings per sample; per-qubit unit renormalisation; Gaussian kernel on Bloch vectors, γ = m²/total variance; SVM.
- Geometry mode (`geometry=`): periodic column → own qubit RY(2πx/P); (lat, lon) → one qubit RY(90° − lat)·RZ(lon); other columns → core encoding.
- Self-tuning on training data only, 0 circuits: 5-fold CV on balanced accuracy over the 30-config ridge grid m ∈ {1/8…4} × C̃ ∈ {0.1…1,000}, C = C̃/m².
- Resources: ⌈d/2⌉ qubits (+1 per circle / sphere in geometry mode), 0 two-qubit gates, 3 circuits per sample.

| BT | Measured (eval seeds 0–9 unless noted) | Status |
|---|---|---|
| BT-1 | BC acc 0.956 vs pinned ZZ 0.677 (+27.9 pp) | met |
| BT-2 | +22.6 to +37.6 pp BA over ZZ on all 4 datasets, 10/10 splits each; +12.3 (`protein_strands`, chain-grouped) / +26.9 pp (`deep_quakes`), 10/10 | met |
| BT-3 | ≥ published − 1 pp on Iono, Sonar, Heart (BC −2.0) | met (3/4) |
| BT-4 | ≥ published on Iono (+1.4), Sonar (+1.1), Heart (+1.2) | met (3/4) |
| BT-5 | mean BA 0.878 vs separable 0.870, FQK-bw 0.864 / 0.856, PQK 0.862, ZZ 0.587 | met |
| BT-6 | Top mean BA on every protein variant, CI never clears 0 under chain-grouped splits: `protein_strands` +0.7 [−0.6, +2.1] (95%); `protein_div6` +0.25 [−1.13, +1.62], `protein_div10` +0.81 [−0.61, +2.23] (97.5%, fresh seeds 300–309). Leaky residue split: +1.4 [+0.3, +2.4]. `deep_quakes`: −6.6 vs RF | **missed** (final) |
| BT-7 | 1,707 vs 2,423,299 circuits (1,420×) | met |
| BT-8 | core: 4 qubits at d = 8, 0 two-qubit gates (3 devices) | met; geometry mode exceeds ⌈d/2⌉ (flagged) |
| BT-9 | drop 0.12 / 0.23 / 0.47 pp (Torino / Fez / Brisbane); FQK-bw 17.5–22.5 pp | met |
| BT-10 | 61.5 s, 11 models × 10 seeds, Breast Cancer (proxy) | met |

## 1. Setup
- Machine: Apple M4, 10 cores. Repo `.venv`: Python 3.12, qiskit 2.6.1, qiskit-aer 0.18.0, qiskit-ibm-runtime 0.51.0, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, joblib 1.6.0.
- Harness: [`common/lab.py`](common/lab.py) (datasets, preprocessing, exact kernels, model specs, protocol, statistics), [`common/run.py`](common/run.py) and [`common/run_x.py`](common/run_x.py) (parallel runners), [`common/cands.py`](common/cands.py) (candidate encodings, geometry mode), [`common/analyze.py`](common/analyze.py) (tables, corrected CIs). Copied from the archived stage `01-fair-verdict` (branch `archive/benchmark-library`, `docs/roadmap/01-fair-verdict/experiments/`); science reused, framing dropped.
- Noise path: [`common/noise_archive.py`](common/noise_archive.py) (archived e05 code, reference) and the experiment folders that use it.
- Data: [`data/`](data/README.md) (UCI files, CC BY 4.0, SHA-256 checked after copy: identical; structure-fit sets from USGS and RCSB PDB added in iteration 2).
- Kernel checks re-run after the copy: [e00](e00-kernel-checks/README.md), max error ≤ 3.1e-15 vs Qiskit `Statevector`.

## 2. Protocol (fixed before iteration 1)
Follows the fair-comparison rules in chellange.md. Phase 8 adopts it as the benchmark protocol.

### 2.1 Datasets
| Id | Rows × features | Positive class | Licence | Feature recipe |
|---|---|---|---|---|
| `breast_cancer` | 569 × 30 | malignant (212) | CC BY 4.0 (UCI 17) / BSD-3 (sklearn copy) | `pca8` |
| `ionosphere` | 351 × 33 (constant column dropped) | bad (126) | CC BY 4.0 (UCI 52) | `pca8` |
| `sonar` | 208 × 60 | rock (97) | CC BY 4.0 (UCI 151) | `pca8` |
| `heart` | 297 × 13 (6 rows with `?` dropped) | disease (137) | CC BY 4.0 (UCI 45) | `pca8` |
| `deep_quakes` (structure-fit, iteration 2) | 3,729 × 2 (latitude, longitude, degrees) | depth ≥ 70 km (795) | U.S. public domain (USGS ComCat) | `native`; geometry: (lat, lon) = one sphere point |
| `protein_strands` (structure-fit, iteration 2) | 3,000 × 6 (φ, ψ of residues i−1, i, i+1, degrees) from 150 chains | strand, PDB `SHEET` records (633) | CC0 1.0 (RCSB PDB) | `native`; geometry: 6 circles, period 360°; **chain-grouped splits** (§2.2) |
| `protein_div6`, `protein_div10` (structure-fit, iteration 5) | 4,000 × 6 / × 10 (φ, ψ of i−1 … i+1 / i−2 … i+2, degrees) from 1,773 chains, ≤ 3 rows per chain | strand, PDB `SHEET` records (996) | CC0 1.0 (RCSB PDB) | `native`; geometry: 6 / 10 circles; **chain-grouped splits**; `protein_div10` = proposed showcase (top mean BA, BT-6 not met) |
- Positive class = minority class, label 1.
- A new dataset enters only with: source, licence, citation, SHA-256, loader rule, recipe. Its recipe is declared before any model runs on it.

### 2.2 Splits and seeds
- `StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)`. Same split for every model.
- **Dev seeds 100–109**: exploration and method choice.
- **Eval seeds 0–9**: every reported number and every BT check. A method is chosen on dev seeds, then run once on eval seeds. No re-selection on eval seeds.
- Note: the archived loop chose its Bloch kernel on seeds 0–9. Those numbers are reused as context, not as fresh confirmation.
- **Fresh seeds 200–209** (added before iteration 2): datasets or models chosen with results on some seeds (e.g. the groundwork pilot used seeds 0–9) are confirmed on seeds never used for that choice. Every screened dataset is reported, wins and losses.
- **Grouped data** (added in iteration 4): when rows share a source unit, no unit may straddle a split. `protein_strands` (rows = residues of 150 PDB chains): outer `GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)` by chain; inner `StratifiedGroupKFold(5, shuffle=True, random_state=seed)` by chain. If a seed's test class share drifts > 3 pp, use a stratified-group variant instead (not needed on seeds 200–209: max drift 2.7 pp). The corrected t-test uses the mean realised n_te/n_tr. Residue-level runs (e07, e08) are context only.

### 2.3 Preprocessing and features (per split, training rows only)
- Recipe `pca8`: StandardScaler → PCA(8, `random_state=0`) → StandardScaler. d ≤ 8: no PCA. Fit on the training rows; re-fit inside every CV fold; test rows only transformed. All models get the same 8 features.
- Recipe `native`: raw columns, no shared transform. Every model applies its own scaler, fit on training rows (classical: StandardScaler; quantum: its encoding rule). All models get the same raw columns.
- Model-internal scaling is part of a model and is fit on training rows only (pinned ZZ: MinMax to [0, π], test clipped; separable: MinMax to [−π/2, π/2]).
- All-features runs (no PCA) are a scaling side study for models that scale (classical, product-state kernels). They never mix with `pca8` rows in one comparison.

### 2.4 Tuning (equal effort)
- 5-fold `StratifiedKFold(shuffle=True, random_state=seed)` inside the training split. Selection score: mean balanced accuracy (BA).
- Budget: **30 configs per tuned model**. Pinned ZZ QSVC: 4 configs (fixed by PRD FR-2). Ties → first config in grid order (grids ordered simple → flexible).
- Refit the chosen config on the whole training split; evaluate once on the test split.
- Kernel grids are ridge-following (archived e01, theory: C·bandwidth² = const near the linear limit): m ∈ {1/8, 1/4, 1/2, 1, 2, 4} × C̃ ∈ {0.1, 1, 10, 100, 1000}, C = C̃/m². Applies to RBF-SVM and every quantum kernel model.
- Classical grids: LR C ∈ logspace(−4, 4, 30); RF (200 trees) max_features ∈ {sqrt, 0.1, 0.2, 0.4, 0.7, 1.0} × min_samples_leaf ∈ {1, 2, 4, 8, 16}; GB (HistGradientBoosting, 200 iters) learning_rate ∈ {0.01, 0.03, 0.1, 0.3, 1} × max_leaf_nodes ∈ {2, 3, 7, 15, 31, 63}; MLP (lbfgs, 2,000 iters) hidden ∈ {(16), (64), (256), (16,16), (64,64)} × α ∈ {1e-3, 1e-2, 0.1, 1, 10, 100}.
- Hyperparameters come from training data only. No test labels, no test statistics (e.g. γ heuristics) anywhere.

### 2.5 Models
| Id | Role | Definition |
|---|---|---|
| `trivial` | trivial baseline | most frequent class |
| `logreg`, `rbf_svm`, `random_forest`, `grad_boost`, `mlp` | classical baselines (all five run; "best classical" = highest mean test BA among them) | grids in §2.4 |
| `zz_qsvc` | standard QML baseline (PRD FR-2) | `zz_feature_map(n, reps=2, entanglement="full")`, fidelity kernel, inputs MinMax to [0, π] fit on train, C ∈ {0.1, 1, 10, 100} |
| `fqk_bw_full`, `fqk_bw_linear` | published: bandwidth-tuned fidelity kernel (arXiv:2111.05451, 2206.06686) | same ZZ map (full / linear), x → c·x, c = m·c_zz, c_zz = 1/√(d + 2π²·E) |
| `pqk` | published: projected quantum kernel (arXiv:2011.01938; qml-benchmarks arXiv:2403.07059) | d+1 qubits, Haar-random product start (seed 1234), 3 Trotter steps of exp(−i(t/3)x_j(XX+YY+ZZ)), t = 0.1; Gaussian on 3(d+1) local Paulis |
| `separable` | published: separable kernel (arXiv:2403.07059) | per feature RX(π/4)·RY(s·x); K = Π_j (3/4 + cos(s·Δx_j)/4) |
| `bloch_gauss` | Qlassifier core (default encoding) | product map RX(c₀x_a)·RY(c₀x_b) per qubit, c₀ = 2/√d; Bloch vectors from 3 circuits per sample; K = exp(−γ‖F−F′‖²), γ = m²γ₀ |
| `qlass_geo` | Qlassifier geometry mode (added in iteration 2; datasets with declared geometry, recipe `native`) | periodic column → RY(2πx/P) on its own qubit; (lat, lon) → RY(90° − lat)·RZ(lon) on one qubit; other columns as `bloch_gauss`; same kernel and grid |
| twin (control, not a challenge baseline) | classical twin of geometry mode | RBF on sin/cos (circles), 3-D unit vectors (spheres), z-scores (rest); same Gram matrix as `qlass_geo` when all columns are angular (checked ≤ 1.8e-15), so it is reported, not re-run |
| candidates | this loop | defined in each experiment README |
- Exact mode uses verified closed forms / batched statevectors ([e00](e00-kernel-checks/README.md)).

### 2.6 Metrics
- Headline: balanced accuracy (BA). Also accuracy, ROC-AUC (from `decision_function` / `predict_proba`), F1 (positive = minority).
- Wall time per split: tuning, final fit, predict.
- Quantum resources: qubits; circuits per fit and predict (counted in exact mode too); two-qubit gates and depth per circuit after transpilation.

### 2.7 Statistics
- Per split j: d_j = BA_model − BA_reference.
- **Corrected resampled t-test** (Nadeau & Bengio 2003): SE = √((1/J + n_te/n_tr)·s²), t with J−1 df, 95% CI. Plain t-tests over re-splits are not used (pseudo-replication).
- "Best classical" is picked ex post (highest mean test BA), which is conservative for a quantum claim.
- Across datasets: sign counts; Wilcoxon over datasets where needed. No pooled p-value.
- Paired win rate vs `zz_qsvc` (share of splits with higher BA).

### 2.8 Noise settings
- Devices: `SamplerV2.from_backend(AerSimulator.from_backend(FakeTorino() | FakeFez() | FakeBrisbane()))` (Heron r1, Heron r2, Eagle r3). A plain `SamplerV2()` attaches no noise.
- Transpile: `generate_preset_pass_manager(optimization_level=3, backend, seed_transpiler=0)`. 1,024 shots. Fixed sampler seed.
- `shots` mode: ideal binomial sampling of each Pauli expectation (exact law for product states).
- Hyperparameters from exact-mode tuning of the same split; no re-tuning on noisy data.
- Qlassifier: Breast Cancer, all rows, eval seeds 0–4. Bandwidth-tuned FQK comparison: same stratified 60 train / 40 test subset for both models, eval seeds 0–2 (FQK needs O(N²) circuits).
- Reported: accuracy and BA per device, drop vs exact (pp), kernel Pearson r vs exact, circuits, two-qubit gates per circuit.
- Geometry mode (iteration 3): structure-fit showcase measured the same way on fresh seeds 200–204, configs from exact-mode CV ([e08](e08-geo-noise/README.md)).

### 2.9 Cost and speed
- BT-7: circuits to train + predict, Breast Cancer 70/30 (398 / 171). Qlassifier: settings × 569. FQK-bw pipeline: bandwidth search on device (6 bandwidths × 5 folds: N_a(N_a−1)/2 + N_b·N_a each) + final Gram (N_tr(N_tr−1)/2 + N_te·N_tr).
- BT-8: transpile to FakeTorino / FakeFez / FakeBrisbane, opt 3, transpiler seeds 0–2 (median).
- BT-10: harness proxy (same models, 10 eval seeds, Breast Cancer) here; phase 9 re-measures the shipped command.

### 2.10 Targets → measurements
| ID | Measurement under this protocol |
|---|---|
| BT-1 | Breast Cancer, eval seeds: mean test acc, Qlassifier − `zz_qsvc` ≥ 20 pp |
| BT-2 | every bundled dataset: mean BA > `zz_qsvc`; paired win on ≥ 80% of splits |
| BT-3 | mean BA ≥ published − 1 pp on ≥ 3 of 4 (BC 0.968, Iono 0.911, Sonar 0.813, Heart 0.804; arXiv:2604.18837; cross-protocol context) |
| BT-4 | mean BA ≥ published on ≥ 2 of 4 |
| BT-5 | mean BA over the 4 datasets highest among Qlassifier, `zz_qsvc`, `fqk_bw_*`, `pqk`, `separable` |
| BT-6 | ≥ 1 bundled real dataset: mean BA > every classical baseline and corrected 95% CI of Δ vs best classical > 0 |
| BT-7 | ≥ 100× fewer circuits than the FQK-bw pipeline incl. its bandwidth search (§2.9) |
| BT-8 | qubits ≤ ⌈d/2⌉; transpiled 2Q gates ≤ n − 1 on all three devices |
| BT-9 | mean acc drop exact → each device ≤ 2 pp (Breast Cancer, 1,024 shots); smaller than FQK-bw drop on the paired subset |
| BT-10 | Breast Cancer benchmark, defaults, exact, ≤ 2 min (harness proxy) |

## 3. Experiments
| Id | Question | Key result | Decision |
|---|---|---|---|
| [e00](e00-kernel-checks/README.md) | Harness still exact after the copy? New encodings exact? | max error ≤ 3.1e-15 (baselines), ≤ 6.7e-16 (candidates, geometry mode) | trust fast paths |
| [e01](e01-reproduce/README.md) | Reference leaderboard, eval seeds (archived protocol) | 480/480 rows identical to the archive; Bloch 0.878 mean BA, ties tuned classical, +23–38 pp vs ZZ | reference fixed |
| [e02](e02-dev-pca8/README.md) | H3 multi-scale, H4 re-uploading, H5 balanced loss (dev) | −0.3 / −5.5 / −0.3 pp mean BA vs Bloch | all dropped |
| [e03](e03-all-features/README.md) | H1 all features (dev) | everyone gains; Δ vs best classical −0.1 to −2.0 pp | dropped as BT-6 route; PCA-8 stays shared |
| [e04](e04-complex/README.md) | H2 complex values on Riemann-sphere qubits, Ionosphere (dev) | −0.6 / −0.8 pp vs RBF-SVM | dropped |
| [e05](e05-noise/README.md) | BT-9 noise, eval seeds | drop 0.12 / 0.23 / 0.47 pp (Torino / Fez / Brisbane); FQK-bw 17.5–22.5 pp | met |
| [e06](e06-resources/README.md) | BT-7 cost, BT-8 resources | 1,420× fewer circuits than FQK-bw pipeline; 4 qubits, 0 two-qubit gates | met |
| [e07](e07-structure-fit/README.md) | H7/H8: geometry mode vs every tuned classical model, structure-fit data (fresh seeds, residue-level splits) | `protein_strands` +1.4 [+0.3, +2.4] pp vs best classical; `deep_quakes` −6.6 pp vs RF | keep geometry mode; protein = showcase; re-tested in e11 |
| [e08](e08-geo-noise/README.md) | H10: protein win under device noise; noise-aware tuning (quakes) | drop 0.31 / 0.67 / 0.11 pp; noise-aware CV changes nothing on quakes (−8.2 pp) | hardware-ready showcase; noise-aware tuning dropped |
| [e09](e09-quake-grid/README.md) | H9 (exploratory): quake loss = grid limit? | ×8 grid: 0.814, still −2.1 pp vs RF; +5.2 pp vs RBF-SVM | trees win on very local labels |
| [e10](e10-speed/README.md) | BT-10 speed proxy | 61.5 s for 11 models × 10 seeds | met |
| [e11](e11-grouped-split/README.md) | H11: does the protein win survive chain-grouped splits? | geometry 0.879 vs MLP / GB 0.872: +0.7 [−0.6, +2.1] pp; still +1.6 / +1.9 pp (CI > 0) vs RBF-SVM / RF | BT-6 downgraded to missed; grouped splits adopted |
| [e12](e12-diverse-chains/README.md) | H12 / H13 (last BT-6 attempt): 1,773 chains; 6 vs 10 angles; 97.5% CIs | top mean BA both times (0.889 / 0.897); Δ vs best classical +0.25 [−1.13, +1.62] / +0.81 [−0.61, +2.23] | both fail → BT-6 missed, stop rule fires |
| [e13](e13-low-d-angle-scale/README.md) (phase 9) | Core angle scale below 8 plain columns (found by scikit-learn's estimator checks) | c₀ = 2/√max(d, 8): +0.7 to +17 pp at d < 8, identical at d ≥ 8 | keep; benchmark numbers unaffected (ADR 0001 amendment) |
| [e14](e14-per-qubit-noise/README.md) (phase 9) | Per-qubit vs whole-register simulation of product circuits on fake devices (the 10-qubit noise cells took ≈ 2 h each) | exact marginals equal to 3.3e-16; shot-level accuracy within the binomial SE | keep, behind an exactness guard (ADR 0006 amendment) |

## 4. Failed ideas
Details and numbers in [learning-log.md](learning-log.md). All are results, kept on purpose.

| Idea | Why it failed | Evidence |
|---|---|---|
| All features as a BT-6 route (H1) | every model gains; RBF-SVM as much as the Bloch kernel | [e03](e03-all-features/README.md) |
| Complex values on Riemann-sphere qubits, Ionosphere (H2) | faithful embedding of ℂ, but the label has no seam to exploit; −0.6 pp vs RBF-SVM | [e04](e04-complex/README.md) |
| Multi-scale Bloch features (H3) | still a Gaussian on smooth features; −0.3 pp at 3× circuits | [e02](e02-dev-pca8/README.md) |
| Data re-uploading ×2 (H4) | doubled frequencies, narrower kernel; −5.5 pp | [e02](e02-dev-pca8/README.md) |
| Class-balanced loss (H5) | shifts every kernel model alike; no relative edge | [e02](e02-dev-pca8/README.md) |
| Geometry mode on very local labels (`deep_quakes`) | trees adapt to uneven density; −6.6 pp (−2.1 pp with a ×8 grid); −8 pp under 1,024-shot noise | [e07](e07-structure-fit/README.md), [e09](e09-quake-grid/README.md), [e08](e08-geo-noise/README.md) |
| Noise-aware tuning for narrow kernels | CV on measured features picks the same config; loss is shot resolution | [e08](e08-geo-noise/README.md) |
| BT-6 win on `protein_strands` under chain-grouped splits (H11) | margin +0.7 pp, CI spans 0; residue splits had inflated it to +1.4 pp | [e11](e11-grouped-split/README.md) |
| More chains (H12) or wider context (H13) for BT-6 | top mean BA kept, margin 0.25–0.81 pp vs GB / MLP, CIs span 0; per-split sd barely shrank | [e12](e12-diverse-chains/README.md) |
| Archived: CZ-chain entanglement, ARD / KTA, fidelity readout under shots, calibration circuits, C re-tuning on noisy kernels, PQK t ≥ 1 | no gain or worse under noise | archived `01-fair-verdict/experiments/README.md` |
| Groundwork: entangled geometry variants (CZ chain, Bell readout) | −5.6 to +0.3 pp | `bloch-sphere-angular-data.md` |
