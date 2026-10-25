# Stage 1 learning log — research–experiment–learn loop

Up: [experiments/README.md](README.md) (protocol §2, index §3) · PRD targets: §6 · Research: prior art, theory · Workflow: loop rules
One entry per iteration: hypotheses (predicted effect, smallest test, kill criterion) → results → lessons → decision. Hypotheses are written before their results. Failed ideas stay here.

## Iteration 0 — setup and starting point (2026-10-24)
- **Reused science** (archived stage `01-fair-verdict`, read-only; framing rejected by the user):
  - Bloch kernel (`bloch_gauss`): product map RX(c₀x_a)·RY(c₀x_b) per qubit, c₀ = 2/√d, 3 circuits per sample, Gaussian on Bloch vectors, γ and C tuned classically.
  - PCA-8, eval seeds 0–9: BA 0.948 / 0.925 / 0.824 / 0.816 (BC / Iono / Sonar / Heart); tie with tuned classical on all 4; +23–38 pp over pinned ZZ; best mean BA among in-harness quantum models.
  - Noise drop ≤ 0.5 pp (FakeTorino / FakeFez / FakeBrisbane); 0 two-qubit gates; 1,707 circuits on Breast Cancer.
  - Failed there: entanglement (CZ chains), ARD, fidelity readout under shots, readout calibration, C re-tuning on noisy kernels.
- **Done now:** harness + data copied (SHA-256 identical); kernel checks re-run ([e00](e00-kernel-checks/README.md), max error ≤ 3.1e-15); protocol fixed ([README §2](README.md#2-protocol-fixed-before-iteration-1)); dev seeds 100–109 vs eval seeds 0–9 split introduced.
- **Gap analysis vs PRD §6 (archived evidence, mapped to the new BT ids):**

| BT | Status from archived evidence | Gap |
|---|---|---|
| BT-1, BT-2, BT-5, BT-7, BT-8, BT-9, BT-10 | met | re-confirm (e01, later noise/cost runs) |
| BT-3 | met 3/4 (BC 0.948 vs 0.968 − 1 pp = 0.958 ✗) | BC +1 pp would make 4/4 |
| BT-4 | met 3/4 (Iono, Sonar, Heart) | — |
| BT-6 | **missed**: ties only; best Δ vs best classical +0.5 pp (Sonar), CI ±9 pp | **biggest gap** |

- **What BT-6 needs (statistics, from archived spreads):** corrected CI half-width ≈ 1.64 × sd of per-split Δ at J = 10, 70/30. sd of Δ ≈ 1.1 (BC), 1.8 (Iono), 5.5 (Sonar), 1.3 (Heart) pp → need a consistent mean win of ≈ +1.8 / +2.9 / +9 / +2.2 pp. Sonar is out of reach by variance; Ionosphere and Heart are the realistic bundled candidates; structure-fit data (parallel groundwork) is the main route.
- Parallel groundwork (other agent): angular / directional / cyclic real data → `docs/references/ideas/bloch-sphere-angular-data.md`. Not duplicated here.

## Iteration 1 — cheap routes on bundled data (dev seeds 100–109)
### Research
- Public tabular benchmarks rarely let angle-encoded quantum models beat tuned classical ones; an edge needs target structure the classical inductive bias lacks ("Fourier wall", arXiv:2607.15815; equal-budget horse races arXiv:2607.20168, 2604.18837).
- Encoding structure can carry an inductive bias tuned RBF lacks (delay-window-aware kernel beats tuned RBF / Matérn, arXiv:2609.13360; task-aligned spectral bias, arXiv:2610.23834).
- Complex-valued sensor data is a natural fit for qubits: inverse stereographic projection maps ℂ onto the Bloch (Riemann) sphere (used for optical-fibre IQ clustering, arXiv:2308.03949). Ionosphere's 34 columns are 17 complex autocorrelation values (UCI 52 description); the bundled pipeline standardises Re / Im separately and pairs columns across pulses (constant column dropped).
- Multi-scale / local kernels mitigate concentration but gains are dataset-dependent (arXiv:2602.16097); data re-uploading raises the Fourier degree (arXiv:2008.08605).
### Hypotheses (written before results)
| Id | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|
| H1 | All features (no PCA) lift Qlassifier more than classical | +1–3 pp BA for every model; Δ vs best classical within ±1.5 pp | e03: 4 datasets, dev seeds, 5 classical + separable + Bloch at all features | no dataset with Δ(Q − best classical) ≥ +1.5 pp → drop as BT-6 route |
| H2 | One qubit per complex value (Riemann sphere, or aligned Re/Im pairs) fits Ionosphere's radar structure | +0.5–2 pp over the standard encoding; Δ vs best classical 0 to +2 pp | e04: Ionosphere, recipe `native`, dev seeds, 5 classical + separable + 3 encodings | Δ vs best classical < +1 pp → drop |
| H3/H4 | Richer per-qubit spectra: multi-scale features (`bloch_ms`, 9 circuits/sample) or re-uploading ×2 (`bloch_reup2`) | −0.5 to +0.5 pp mean BA vs `bloch_gauss` | e02: PCA-8, 4 datasets, dev seeds | mean gain < +0.5 pp → drop |
| H5 | Class-balanced SVM loss for BA (BT-3 Breast Cancer) | +0.5–1 pp BA on BC for Bloch; similar for classical controls | e02: `bloch_gauss_bal` + balanced LR / RBF / RF / GB | BC gain < +0.5 pp → drop |
### Experiments
- [e01](e01-reproduce/README.md): archived protocol re-run on eval seeds 0–9 (reference leaderboard; 480/480 rows identical to the archive).
- [e02](e02-dev-pca8/README.md) (H3, H4, H5), [e03](e03-all-features/README.md) (H1), [e04](e04-complex/README.md) (H2): dev seeds 100–109.
- Side work for the kept targets: [e05](e05-noise/README.md) noise (BT-9), [e06](e06-resources/README.md) resources and cost (BT-7, BT-8).
### Results (dev seeds; Δ = candidate − reference, BA, pp)
| Id | Predicted | Measured | Verdict |
|---|---|---|---|
| H1 all features | +1–3 pp all; Δ vs best classical within ±1.5 | Bloch +0.3 to +3.1 vs PCA-8, RBF-SVM +1.0 to +7.3; Δ vs best classical −0.1 / −2.0 / −0.1 / −0.7 | killed (no relative gain) |
| H2 complex qubits (Ionosphere) | +0.5–2 vs standard; Δ vs best classical 0 to +2 | Riemann −0.6, aligned pairs −0.8 vs RBF-SVM (= standard Bloch 0.941) | killed |
| H3 multi-scale | −0.5 to +0.5 | −0.3 mean (3× circuits) | killed |
| H4 re-uploading ×2 | −0.5 to +0.5 | −5.5 mean (−7.1 BC) | killed (worse than predicted) |
| H5 balanced loss | +0.5–1 BC | +0.3 BC; RBF-SVM control +0.5 | killed |
- Kept targets on eval seeds (e01, e05, e06): BT-1 +27.9 pp; BT-2 4/4, 10/10 wins; BT-5 0.878 > 0.870; noise drop 0.12 / 0.23 / 0.47 pp; 0 two-qubit gates; 1,420× fewer circuits than the FQK-bw pipeline.
### Lessons
- On generic tabular data the Bloch kernel and RBF-SVM are near twins (small-angle limit of a Gaussian on trigonometric features). Every tweak that keeps the kernel Gaussian-like keeps the tie; tweaks that raise frequencies (re-uploading) overfit.
- Matching the data's algebra (complex numbers ↔ qubits) is not enough: the label must depend on geometry that generic models miss.
- Split variance between seed blocks is large (BC BA 0.965 dev vs 0.948 eval for the same model): BT-3 for BC hinges on splits, not on the method.
### Decision
- Keep `bloch_gauss` as the generic Qlassifier core (no change). No measurable gain this iteration (1 of 3 for the stop rule).
- Next: structure-fit data from the groundwork pilot.

## Iteration 2 — geometry mode on structure-fit data (fresh seeds 200–209)
### Research (folded in from the groundwork, `docs/references/ideas/bloch-sphere-angular-data.md`)
- A qubit's state space is a sphere; RY(θ) walks a circle with exact 2π period. Angle-native data maps onto qubits with no seam: one circle per qubit (dihedral, hour, wind direction), one 3-D direction per qubit (lat/lon → RY(colatitude)·RZ(longitude)). Two full circles on one qubit alias (T² does not embed in S²).
- Prior art on the maths: Fourier view of angle encoding (arXiv:2008.08605), covariant / geometric QML (arXiv:2105.03406, 2205.06217, 2210.07980), periodic IQP regimes (arXiv:2610.23692). No applied QML on real directional / dihedral data found (groundwork §2).
- Groundwork pilot (their protocol: N = 1,200 subsample, seeds 0–9, rectangular grid): geometry mode vs best classical +1.9 pp on `protein_strand6` (GB), +1.7 pp on `quake` (RF), +1.2 Beijing, +0.5 protein coil, −1.4 bike; all corrected CIs span 0 (sd of Δ 2.5–3.4 pp at N = 1,200).
- Exact identity (checked, max diff ≤ 1.8e-15): with all columns angular, the geometry-mode Bloch kernel equals an RBF on hand-made sin/cos (circle) or 3-D unit-vector (sphere) features. The classical twin ties by construction; it is a transparency control, not a challenge baseline.
### Hypotheses (written before results)
| Id | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|
| H7 | Geometry mode (`qlass_geo`) beats every tuned classical model on raw angles | Δ vs best classical +1 to +3 pp on both datasets; corrected CI half-width ≈ 2.5–3 pp (quakes, N = 3,729) and ≈ 2 pp (protein, N = 3,000) → CI > 0 uncertain | e07: `deep_quakes`, `protein_strands`, recipe `native`, fresh seeds 200–209, 5 classical + pinned ZZ + separable + generic Bloch + geometry mode | Δ ≤ 0 on both → geometry route fails |
| H8 | The win comes from the geometry, not from the kernel | generic Bloch (geometry off) ≥ 2 pp below geometry mode | same run (ablation) | gap < 1 pp → geometry is not the cause |
- Datasets fixed before this run: `deep_quakes` (all 3,729 events), `protein_strands` (label-free thinned sample of 3,000 residues). Pilot datasets with smaller or negative pilot gaps (Beijing: geometry-off model as good; bike: trees win; protein coil: +0.5) are not re-run; their pilot numbers are reported as context.
- FQK-bw and PQK are skipped here (no BT needs them on these datasets; cost); phase 9 runs the full model set on every bundled dataset.
### Experiments
- [e07](e07-structure-fit/README.md): `deep_quakes` (3,729 events) and `protein_strands` (3,000 residues), recipe `native`, fresh seeds 200–209, 10 models.
### Results (fresh seeds; BA)
| Dataset | Geometry mode | Best classical | Δ vs best classical [corrected 95% CI] | Other |
|---|---|---|---|---|
| `protein_strands` | **0.884 ± 0.011** | MLP 0.870 (GB 0.870, RBF 0.868, RF 0.857, LR 0.815) | **+1.4 [+0.3, +2.4], p = 0.015, 10/10 wins** | vs GB +1.4 [+0.03, +2.7]; vs ZZ +11.7; geometry off (generic Bloch) 0.872 → +1.2 [−0.8, +3.1]; separable 0.882 |
| `deep_quakes` | 0.769 ± 0.016 | RF 0.835 (GB 0.835, MLP 0.764, RBF 0.713, LR 0.500) | −6.6 [−8.5, −4.8] | vs RBF-SVM +5.6 [+2.3, +9.0]; vs generic Bloch +4.6 [+0.1, +9.0]; ZZ, separable, LR at 0.500 |
- `protein_strands`: accuracy +0.4 pp vs MLP (CI spans 0), ROC-AUC −0.1 to −0.8 pp vs MLP / GB / RF (CI spans 0), F1 +1.3 [+0.2, +2.5] vs MLP. The edge is a better-balanced decision (more minority-class strands found) at equal ranking quality.
- `deep_quakes`: every kernel model (RBF, Bloch, geometry) picks the narrowest grid corner (m = 4, C̃ = 1,000) on 10/10 splits: the protocol grid cannot reach the few-degree kernel width that subduction zones need. Trees pick their most flexible configs too; they adapt to local density, kernels with one global width do not.
### Lessons
- H7 confirmed on 1 of 2 pre-registered datasets: **BT-6 met on `protein_strands`** on residue-level splits (superseded in iteration 4: missed under chain-grouped splits) (mean BA above all five tuned classical models; corrected CI vs the best classical > 0; survives a Bonferroni correction for the 2 confirmatory datasets, α = 0.025). Missed on `deep_quakes`.
- H8 (geometry is the cause): supported on quakes (+4.6 pp, CI > 0), weak on protein (+1.2 pp, CI spans 0): there the Bloch kernel family is already competitive and geometry turns a tie into a significant win.
- The classical twin computes the same Gram matrix (checked ≤ 1.8e-15): the win is the geometric prior, delivered natively by one qubit rotation per angle.
- Pilot-to-confirmation shift: quakes went from +1.7 pp (N = 1,200) to −6.6 pp (N = 3,729): trees gain most from more data on very local labels. Small-N pilots overstate kernel wins on such tasks.
### Decision
- **Keep geometry mode** in Qlassifier (`geometry=` column declaration; default off). Showcase dataset for BT-6: `protein_strands`. Report `deep_quakes` as a loss with its cause.
- Measurable gain this iteration (resets the stop-rule counter).

## Iteration 3 — robustness of the win and the quake loss (fresh seeds)
### Hypotheses (written before results)
| Id | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|
| H9 (exploratory, post hoc) | The quake loss is a grid limit: with every kernel model on the same ×8-shifted 30-config ridge grid, geometry mode closes the gap to trees | geometry mode 0.80–0.83, still below RF 0.835; RBF-SVM improves as well | e09: kernel models only, same splits as e07 (seeds 200–209); tree / MLP numbers reused | geometry mode < RF → trees win on very local labels (confirms the loss) |
| H10 | The protein win survives IBM device noise at 1,024 shots (6 qubits, 0 two-qubit gates) | BA drop ≤ 0.5 pp on FakeTorino / FakeFez / FakeBrisbane | e08: e07 configs, seeds 200–204, 3 circuits per sample, unit renormalisation | drop > 2 pp on any device |
- H9 uses splits already seen, so it can explain the loss but cannot add a confirmatory win.
### Experiments
- [e08](e08-geo-noise/README.md) (H10 + noise-aware check), [e09](e09-quake-grid/README.md) (H9), [e10](e10-speed/README.md) (BT-10 proxy).
- Folded in from the groundwork (not re-run): buoy and solar pilots (N = 1,200, seeds 0–9): geometry mode − best classical −0.6 (buoy 46026), −1.4 (buoy 51001), +0.7 (solar Boulder), −0.1 pp (solar Utqiaġvik); all corrected CIs span 0; geometry on − off +1.1 to +7.2 pp.
### Results
| Id | Predicted | Measured | Verdict |
|---|---|---|---|
| H9 quake grid ×8 (exploratory) | 0.80–0.83, below RF | 0.814 ± 0.020; − RF −2.1 [−5.2, +1.1]; − RBF-SVM ×8 +5.2 [+1.8, +8.6] | confirmed: the grid explains 4.6 pp, trees still win |
| H10 protein under device noise | drop ≤ 0.5 pp | 0.31 / 0.67 / 0.11 pp (Torino / Fez / Brisbane), worst seed 1.5 pp, kernel r 0.9998 | confirmed (≤ 2 pp) |
| Noise-aware tuning, quakes (coordinator) | recover most of the ~9 pp | CV on measured features picks the same corner on 3/3 seeds; BA 0.767 → 0.685 either way | killed: the loss is shot-noise resolution, not a tuning or fit/predict mismatch |
| BT-10 proxy | ≤ 2 min | 61.5 s wall, 11 models × 10 eval seeds, Breast Cancer | met |
### Lessons
- The protein win holds on all three IBM device models (noisy BA 0.878–0.883 > MLP exact 0.870): the showcase is both top of the leaderboard and hardware-ready.
- Very local labels (quakes) need narrow kernels; narrow kernels need fine angle resolution, i.e. many shots. A fixed-shot quantum kernel cannot buy that resolution with tuning.
- Kernel grids anchored at the total-variance bandwidth suit smooth labels; for very local labels every kernel model hits the edge. Log edge hits for every model.
### Decision
- Iteration 3 closed the robustness questions; BT-6 was re-tested in iterations 4–5 (see below).

## Iteration 4 — leakage check for the BT-6 claim (chain-grouped splits)
### Research
- Residue-level splits put residues of one protein chain in train and test. Residue rows share no dihedral values (thinning, e07), but residues of one chain share fold-level context, which can favour flexible, memorising models. Grouped splits by chain are the stricter test (reviewer request; e07 flag 5).
### Hypothesis (written before results)
| Id | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|
| H11 | The protein win survives chain-grouped splits | Δ vs best classical +0.5 to +1.5 pp | e11: `protein_strands`, recipe `native`, fresh seeds 200–209; outer `GroupShuffleSplit(test_size=0.3)` by `pdb` (class-share drift ≤ 2.7 pp on these seeds, under the 3 pp limit, so no stratified-group variant); inner `StratifiedGroupKFold(5)` by `pdb`; same 30-config grids; 5 classical + geometry mode + generic Bloch + separable + pinned ZZ + trivial | corrected 95% CI of Δ vs best classical ≤ 0 → BT-6 evidence downgraded; no rescue iterations |
- Corrected CI uses the mean realised n_te/n_tr of the grouped splits (n_te 812–973 of 3,000).

### Experiments
- [e11](e11-grouped-split/README.md): chain-grouped outer and inner splits, 10 fresh seeds, 10 models (510 s).
### Results
| Model | BA grouped | BA residue split (e07) |
|---|---|---|
| geometry mode | 0.879 ± 0.015 | 0.884 |
| MLP / GB (best classical, tied) | 0.872 / 0.872 | 0.870 / 0.870 |
| separable / generic Bloch / RBF-SVM / RF / LR / ZZ | 0.877 / 0.871 / 0.863 / 0.861 / 0.809 / 0.756 | 0.882 / 0.872 / 0.868 / 0.857 / 0.815 / 0.767 |
- Geometry − MLP: **+0.7 [−0.6, +2.1] pp, p = 0.25, 9/10 wins** (predicted +0.5 to +1.5). Accuracy +0.0, ROC-AUC −0.8, F1 +0.4 pp (all CIs span 0).
- Still significant: geometry − RBF-SVM +1.6 [+0.3, +2.9]; − RF +1.9 [+0.5, +3.2]; − LR +7.1; − ZZ +12.3 pp.
### Lessons
- Residue-level splits inflated the protein margin (+1.4 → +0.7 pp under chain grouping; different test sets, so the size of the inflation is an estimate). Grouped splits are the honest unit for structured biological data.
- The mean prediction was right; the corrected CI is what a stricter split takes away. Small real-data margins need the strictest split before they become headline claims.
### Decision
- **H11 killed by its pre-registered criterion → BT-6 downgraded to missed.** Evidence kept: top mean BA among all models on chain-grouped splits, significant wins over RBF-SVM, RF, LR and ZZ, tie with MLP and GB.
- Protocol: `protein_strands` uses chain-grouped splits (README §2.2). No rescue iterations (coordinator rule). Loop closed.

## Iteration 5 — last bounded BT-6 attempt (chain-grouped splits, fresh seeds 300–309; budget ≤ 90 min)
### Research
- e11 failed on power, not on sign: Δ vs MLP +0.72 pp, sd of per-split Δ 0.80 pp → 95% half-width 1.33 pp; 97.5% (Bonferroni, k = 2) half-width 1.58 pp. The independent unit is the chain (150 chains).
- Power analysis (e11 spread; corrected t, J = 10, n_te/n_tr 0.444, 97.5%: half-width = 1.98 × sd). A +0.7 pp effect needs sd ≤ 0.35 pp. With N = 4,000 rows over ≥ 1,334 chains (≤ 3 per chain), projected sd 0.69 pp if residue noise dominates (half-width 1.37 pp), 0.43 pp half-way (0.85 pp), 0.27 pp if chain variance dominates (0.53 pp). Only the chain-dominated case has power → N fixed at the cap, 4,000.
- Wider context: strands are β-ladders spanning several residues; φ/ψ of i ± 2 add evidence, and 10 raw-degree columns give trees and MLPs 10 seams (a circle qubit has none).
- Rejected: more seeds (the corrected test saturates on the n_te/n_tr term), new labels or metrics (fishing), new model families (out of scope).
### Hypotheses (written before any data download or run; confirmatory, k = 2, α = 0.025 each)
| Id | Hypothesis | Predicted effect | Test | Kill criterion |
|---|---|---|---|---|
| H12 | Chain diversity gives the grouped test power: geometry mode beats every tuned classical model with 6 angles (i−1 … i+1) | Δ vs best classical +0.5 to +1.5 pp; 97.5% half-width 0.5–1.4 pp | e12, dataset `protein_div6` | 97.5% corrected CI lower bound ≤ 0, or mean not above every classical → fail |
| H13 | Wider context helps more: same rows, 10 angles (i−2 … i+2), 10 circle qubits | Δ +0.5 to +2 pp | e12, dataset `protein_div10` (same rows, same splits) | same |
- **Data plan (fixed now):** RCSB search with the groundwork filters (X-ray, resolution ≤ 1.6 Å, protein-only entry, 1 polymer entity, length 80–400, one representative per 30%-identity cluster, release date ascending); first 1,800 representatives requested; legacy `.pdb` files (CC0); first chain; residues with complete backbone for i−2 … i+2 and no chain break; label = residue in a `SHEET` record. Per chain: up to 3 residues at sequence spacing ≥ 5 (no shared dihedral between rows for either feature set), chosen with numpy seed 0, label-free; then 4,000 rows sampled with seed 0 (≤ 3 per chain, ≥ 1,334 chains).
- **Data as built (before any model run):** 1,800 representatives downloaded; 1,799 chains parsed (1 skipped: no HELIX / SHEET annotation at all, a data-quality rule added at build time, label-free); 326,128 candidate residues; 4,000 rows from 1,773 chains (≤ 3 per chain); strand share 24.9%. Test class-share drift on seeds 300–309: ≤ 2.0 pp → plain `GroupShuffleSplit`, as planned.
- **Analysis plan:** recipe `native`; outer `GroupShuffleSplit(0.3)` by chain, inner `StratifiedGroupKFold(5)` by chain (stratified-group variant for all seeds if any seed's test class share drifts > 3 pp, checked before model runs); fresh seeds 300–309; 30-config grids (ZZ 4); models trivial, LR, RBF-SVM, RF, GB, MLP, pinned ZZ, separable, generic Bloch, geometry mode. Pass = geometry mean BA above all five classical models and corrected 97.5% CI of Δ vs the best classical (highest mean BA, ex post) > 0. Report BA, acc, ROC-AUC, F1, Δ vs every model; mean realised n_te/n_tr. No changes after results. A pass makes that dataset the bundled showcase; no pass → BT-6 missed and the stop rule fires (iterations 3–5 without gain).

### Experiments
- [e12](e12-diverse-chains/README.md): `protein_div6` (H12) and `protein_div10` (H13), same 4,000 rows and splits, 10 models, fresh seeds 300–309 (1,575 s and 1,610 s, in parallel).
### Results (corrected 97.5% CIs)
| Variant (all chain-grouped except the first) | Geometry BA | Best classical | Δ vs best classical | Verdict |
|---|---|---|---|---|
| `protein_strands`, residue split (e07, 95%) | 0.884 | MLP 0.870 | +1.4 [+0.3, +2.4] | passed, but leaky split |
| `protein_strands`, chain-grouped (e11, 95%) | 0.879 | MLP 0.872 | +0.7 [−0.6, +2.1] | fail |
| `protein_div6`, 1,773 chains (H12) | 0.889 | MLP 0.887 | +0.25 [−1.13, +1.62], p = 0.64 | **fail** |
| `protein_div10`, 1,773 chains, i±2 (H13) | 0.897 | GB 0.889 | +0.81 [−0.61, +2.23], p = 0.16 | **fail** |
- H13 detail: − MLP +1.35 [−0.59, +3.29] (10/10), − RF +1.46 [−0.09, +3.00] (10/10), − RBF-SVM +2.29 [+0.16, +4.43]; geometry on − off +2.04 [+0.65, +3.42]; ZZ collapses at 10 qubits (0.543).
- H12 detail: − RF +1.11 [+0.22, +2.01]; geometry on − off +0.17 (n.s.).
- Accuracy and ROC-AUC tie the best classical model in both runs (|Δ| ≤ 0.5 pp, CIs span 0).
- Realised sd of per-split Δ: 0.70 / 0.73 pp (e11: 0.80) → the power analysis's pessimistic scenario; more chains did not shrink the variance enough.
### Lessons
- The sign is stable across all four protein variants (geometry mode has the top mean BA every time), but the margin over the best tuned tree / MLP is 0.25–0.8 pp under leakage-safe splits: below what 10 corrected splits can resolve.
- Geometry pays off as context grows (10 angles: +2.0 pp over the generic encoding, CI > 0), yet gradient boosting gains almost as much from the same columns.
- Per-split variance came mostly from residue-level test noise, not from chain heterogeneity: more chains alone does not buy power.
### Decision
- **H12 and H13 fail their pre-registered rule → BT-6 missed.** The stop rule fires: iterations 3, 4 and 5 brought no measurable gain on the open target. Loop closed; no further attempts.
- Structure-fit showcase for phase 8 (chosen after these results; claim limited accordingly): `protein_div10` — 1,773 chains, leakage-safe grouped split, top mean BA of all models (0.897), significant over RBF-SVM, LR, ZZ and the generic encoding; tie with gradient boosting / MLP. Its device-noise robustness is not yet measured (e08 measured the 6-circle `protein_strands`).

## Final decision — the Qlassifier method (stage 1)
### Method
- **Core (default):** product-state encoding, 2 standardised features per qubit, RX(c₀x_a)·RY(c₀x_b) from |0⟩, c₀ = 2/√d fixed (not tuned). Readout: all qubits in X, Y, Z (3 circuits per sample); per-qubit unit renormalisation of measured Bloch vectors. Kernel K = exp(−m²γ₀‖F − F′‖²), γ₀ = 1/total variance of the training Bloch features. SVM on the precomputed kernel.
- **Geometry mode (opt-in, `geometry=` column declaration):** each periodic column gets its own qubit, RY(2πx/P) (P = 360°, 24 h, …); each (latitude, longitude) pair gets one qubit, RY(90° − lat)·RZ(lon) (Bloch vector = the direction); remaining columns use the core encoding. Same readout, kernel and tuning. Never two circles on one qubit (aliasing, theory §2.3).
- **Self-tuning (training data only, 0 circuits):** 5-fold stratified CV, balanced accuracy, 30-config ridge grid m ∈ {1/8, 1/4, 1/2, 1, 2, 4} × C̃ ∈ {0.1, 1, 10, 100, 1,000}, C = C̃/m²; ties → simplest config. Hyperparameters are chosen on exact (closed-form) features; devices only measure the final features.
- **Defaults:** `geometry=None`; `backend="exact"`; 1,024 shots on `"shots"` / `"fake_torino"` / `"fake_fez"` / `"fake_brisbane"`; transpile at optimisation level 3.
- **Resources:** ⌈d/2⌉ qubits (+1 per declared circle, 1 per sphere pair), 0 two-qubit gates, depth 6–8 after transpilation, 3 circuits per sample (2 for circle-only data in exact mode).
### Evidence against every target (eval seeds 0–9 unless noted)
| BT | Target | Measured | Status |
|---|---|---|---|
| BT-1 | BC acc ≥ ZZ + 20 pp | 0.956 vs 0.677 → +27.9 pp | met |
| BT-2 | BA > ZZ on every bundled dataset, ≥ 80% paired wins | +37.6 / +22.6 / +27.9 / +28.3 pp (BC / Iono / Sonar / Heart), 10/10 each; +12.3 (`protein_strands`, chain-grouped), +26.9 (`deep_quakes`), 10/10 (fresh seeds) | met |
| BT-3 | ≥ published − 1 pp on ≥ 3 of 4 | Iono 0.925 (+1.4), Sonar 0.824 (+1.1), Heart 0.816 (+1.2), BC 0.948 (−2.0) → 3/4 | met |
| BT-4 | ≥ published on ≥ 2 of 4 | 3/4 (Iono, Sonar, Heart) | met |
| BT-5 | best mean BA among in-harness quantum models | 0.878 vs separable 0.870, FQK-bw full 0.864, PQK 0.862, FQK-bw linear 0.856, ZZ 0.587 | met |
| BT-6 | beat every tuned classical model on ≥ 1 real dataset, paired CI > 0 | Geometry mode has the top mean BA on every protein variant but never clears the CI under leakage-safe splits: chain-grouped `protein_strands` +0.7 [−0.6, +2.1] (95%); `protein_div6` +0.25 [−1.13, +1.62] and `protein_div10` +0.81 [−0.61, +2.23] (97.5%, Bonferroni k = 2, fresh seeds 300–309). Context: residue-level split +1.4 [+0.3, +2.4]; `deep_quakes` −6.6 vs RF | **missed** (stop rule fired after iteration 5) |
| BT-7 | ≥ 100× fewer circuits than FQK-bw pipeline | 1,707 vs 2,423,299 → 1,420× | met |
| BT-8 | qubits ≤ ⌈d/2⌉, 2Q gates ≤ n − 1 | core: 4 qubits at d = 8, 0 two-qubit gates on all 3 devices | met (core); geometry mode on 6 angles uses 6 qubits (flag) |
| BT-9 | acc drop ≤ 2 pp per device; < FQK-bw drop | 0.12 / 0.23 / 0.47 pp; FQK-bw 17.5–22.5 pp (paired subset) | met |
| BT-10 | BC benchmark ≤ 2 min | 61.5 s (harness proxy, 11 models × 10 seeds, 10 cores) | met (proxy) |
### Failed ideas (all kept above)
- This loop: all features as a BT-6 route (H1); complex values on Riemann-sphere qubits (H2); multi-scale Bloch features (H3); data re-uploading ×2 (H4, −5.5 pp); class-balanced loss (H5); geometry mode on very local labels (`deep_quakes`, trees win even with a ×8 grid, H9); noise-aware tuning for narrow kernels.
- Archived loop: entangled CZ-chain encodings, ARD / kernel-target alignment, fidelity readout under shots, readout calibration circuits, C re-tuning on noisy kernels, rectangular grids, PQK with t ≥ 1.
- Groundwork pilot: entangled geometry variants (CZ chain, Bell readout) −5.6 to +0.3 pp.
- Iteration 4: the BT-6 win does not survive chain-grouped splits at the 95% level (H11).
- Iteration 5: more chains (H12) and wider context (H13) keep the top mean BA but not the CI.
### Flags for the user
1. **BT-6 is missed (final).** Under leakage-safe chain-grouped splits, geometry mode is top of the leaderboard on mean BA in all three grouped runs (0.879 / 0.889 / 0.897) but ties the best tuned tree or MLP (+0.25 to +0.81 pp, CIs span 0). Only the leaky residue split cleared the CI. Claims must stay at "top mean BA; significant vs RBF-SVM, LR, ZZ; tie with GB / MLP", always with the exact classical twin. Never quantum advantage. This changes an acceptance criterion: needs your decision (accept the miss, or carry BT-6 into stage 2).
2. **BT-8 vs geometry mode (questioned requirement):** a full circle needs its own qubit (T² does not embed in S²), so 6 angles use 6 qubits, above ⌈d/2⌉ = 3. Still 0 two-qubit gates. Proposal: apply ⌈d/2⌉ to the default encoding and "1 qubit per circle or direction" to geometry mode. Changing the criterion needs your approval. **Resolved 2026-10-25:** approved at the stage-1 merge (ADR 0005); PRD BT-8 amended.
3. `deep_quakes` is a reported loss (trees win; also −8 pp under 1,024-shot noise). Keep it as a simulator teaching example or drop it; it is not the showcase.
4. BC stays 2.0 pp below the published 0.968 (QKT with leaky feature selection); BT-3/BT-4 are met on the other three datasets.
5. Resolved: `protein_strands` now uses chain-grouped splits in the protocol (e11). Residue-level numbers (e07, e08) stay as context; the e08 noise drops do not depend on the split type.
6. Licences: PDB data CC0 1.0 (fits the PRD list); USGS ComCat is U.S. public domain (not literally in the PRD list; equivalent in practice).
7. Protocol additions made during the loop (before the runs that used them): `native` recipe, dev / eval / fresh seed blocks, structure-fit dataset rows. Phase 8 adopts README §2 as the benchmark protocol.

## Phase-9 addendum — angle scale below 8 plain columns (e13)
- **Found by:** scikit-learn's estimator checks during implementation (2-feature blobs: training accuracy 0.82).
- **Hypothesis (written before the run):** c₀ = 2/√d over-rotates standardised values at small d (folding on the Bloch sphere); capping it at 2/√8 removes the fold and leaves d ≥ 8 untouched. Kill: no gain at d < 8 or any change at d = 8.
- **Result ([e13](e13-low-d-angle-scale/README.md)):** +17 pp (blobs), +3.0 (moons), +7.0 (circles), +3.3 / +0.7 / +0.7 pp (Heart PCA-2 / 4 / 6); Heart PCA-8 identical.
- **Decision:** keep `c₀ = 2/√max(d_plain, 8)` (ADR 0001 amendment). No benchmark target moves: the protocol uses d = 8 (UCI) and geometry mode (protein).

## Phase-9 addendum — per-qubit device simulation (e14)
- **Found by:** the shipped noise study stalled on `protein_angles` part A (10 qubits: ≈ 2 h per fake-device cell; Aer cost grows 4ⁿ / 2ⁿ per shot), and the TDD quickstart `Qlassifier(backend="fake_fez")` on raw Breast Cancer (15 qubits) did not finish in 10 min.
- **Hypothesis (before the run):** for product circuits on local noise models (single-qubit gate and readout errors, no idle noise, no crosstalk) and per-qubit readout, simulating each qubit alone on its pinned physical qubit gives the same marginals exactly. Kill: any exact marginal off by > 1e-12, or a shot-level gap outside the binomial SE.
- **Result ([e14](e14-per-qubit-noise/README.md)):** 450 exact marginals (4 and 6 qubits, 3 devices, 3 settings) agree to 3.3e-16 after trace normalisation (Aer's channels are trace preserving to ~1e-10); shot-level part A accuracy within 0.35 pp of the whole-register path (binomial SE 1.08 pp).
- **Decision:** keep it, behind an exactness guard with a whole-register fallback (ADR 0006 amendment). The phase-9 benchmark run uses it; e05 / e08 stay as recorded.
