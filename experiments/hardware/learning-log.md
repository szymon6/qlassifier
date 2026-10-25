# Stage 2 learning log — research–experiment–learn loop

Up: [experiments/README.md](README.md) (protocol §2, index §3) · PRD targets: §6 · Research: prior art, novelty · Stage 1: [learning log](../classifier/learning-log.md) (iteration 5, final decision, flags) · Workflow: loop rules
One entry per iteration: hypotheses (predicted effect, smallest test, kill criterion) → results → lessons → decision. Hypotheses are written before their results. Failed ideas stay here.

## Iteration 0 — setup and gap analysis (2026-10-25)
- **Starting point (stage 1, shipped 0.1.0):** Qlassifier = Bloch-kernel core + geometry mode; 9 of 10 stage-1 targets met, BT-6 (beat tuned classical, CI > 0) missed under leakage-safe splits (protein: +1.0 [−0.2, +2.2] pp vs GB on seeds 0–9). Device drop ≤ 0.5 pp at 1× FakeTorino / FakeFez / FakeBrisbane; FQK-bw linear 17.5–22.5 pp on the paired subset; 1,420× fewer circuits than the FQK-bw pipeline.
- **Environment:** no IBM token (`QISKIT_IBM_*` unset) → BT-3 not run (R-1), protocol fixed in README §2.5.
- **Protocol fixed** (README §2): stage 1 §2 adopted; stage-2 additions: dataset entry rule + the five new datasets, noise-scaling definition (Target scaling, checked exact in [e00](e00-setup-checks/README.md)), cost-model definition with the three published per-circuit timings, hardware protocol (not run), BT-6 regression check, budget.
- **Gap analysis vs PRD §6:**

| BT | Evidence before stage 2 | Gap |
|---|---|---|
| BT-1 | none on new data; stage 1: +22.6 to +37.6 pp over ZZ on 4 UCI sets, ≥ published on 3 of 4 | run 3 new UCI sets once on eval seeds |
| BT-2 | only s = 1 measured (drop 0.12 pp part A; 0.8 pp part B); FQK-bw 20 pp at s = 1 | sweep 0×–3×, AUC |
| BT-3 | — | not run; protocol |
| BT-4 | counts exact in stage 1 (`resources_`, cost study); no QPU-time model | model + bracket the published timings |
| BT-5 | stage-1 BT-6 missed; pilots on beijing / solar were ties at N = 1,200 | screen 2 geometry sets at N = 4,000 with grouped splits; quake shots hypothesis |
| BT-6 | stage-1 shipped results | re-run BC + heart, compare rows |

- **Research (summary; details in prior-art.md):** no published accuracy-vs-noise-scale sweep of a quantum kernel classifier on real tabular data was found (searches: arXiv "quantum kernel" × noise strength / scaled noise / error rate sweep; closest: exponential concentration under noise, arXiv:2208.11060; hardware–ideal kernel fidelity at one noise level, arXiv:2604.18837, 2607.20377; noise-scaling as a mitigation primitive, arXiv:2005.10921). Published per-circuit QPU timings: 0.27 s (arXiv:2607.20377), 0.32 s (arXiv:2604.18837) at 1,024 shots; 0.69 s at 2,048 shots (arXiv:2605.14672); IBM's job-time model (rep_delay 250 µs + circuit length per execution, ≈ 2 s per sub-job). Published quantum BA on the BT-1 candidates: spambase 0.888, banknote 0.921, haberman 0.586 (arXiv:2604.18837). No QML work on the BT-5 candidates.

### Hypotheses for iteration 1 (written before any result)
| Id | Target | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|---|
| H1 | BT-1 | Qlassifier beats pinned ZZ and reaches the published quantum BA on the new UCI sets | spambase BA ≈ 0.90–0.93 (ZZ ≈ 0.5–0.7), banknote ≈ 0.99–1.00 (ZZ ≈ 0.7–0.9 at 4 qubits), haberman ≈ 0.55–0.62 (ZZ ≈ 0.5); ≥ published on ≥ 2 of 3 | [e01](e01-new-datasets/README.md): eval seeds 0–9, full model set, cost study | < 2 datasets pass both conditions → BT-1 missed (no rescue runs) |
| H2 | BT-2 | Qlassifier's accuracy is flat in s because its circuits have 0 two-qubit gates, depth ≤ 10 and per-qubit renormalisation; fidelity kernels collapse with s (depth × error) | part A drop at s = 1: 0.1–0.5 pp; at s = 3: ≤ 2 pp. Part B: FQK-bw linear drop ≈ 3–5 pp at s = 0 (shots), ≈ 20 pp at s = 1, > 25 pp at s ≥ 2; ZZ at majority level throughout. AUC Qlassifier > FQK-bw > ZZ | [e02](e02-noise-sweep/README.md): §2.3 | part-A drop at s = 1 > 2 pp, or AUC(Qlassifier) ≤ AUC(FQK-bw) |
| H3 | BT-4 | IBM's job-time model with device defaults brackets every published per-circuit timing | low–high at 1,024 shots ≈ 0.26–0.37 s/circuit, at 2,048 ≈ 0.52–0.72 s; published 0.27 / 0.32 / 0.69 inside | [e03](e03-cost-model/README.md) | any published point outside [low, high], or a count that differs from executed circuits |
| H4 | BT-5 | `beijing_pm25` (wind direction, hour, month as circles; week-grouped) — geometry mode ties the best tuned classical model | Δ vs best classical −1.5 to +1.0 pp (pilot +0.9 at N = 1,200, trees gain with N) | [e04](e04-geometry-screen/README.md): dev seeds 100–104 | dev Δ < +1.0 pp → no confirmation run; recorded as tie / loss |
| H5 | BT-5 | `solar_boulder` (hour × day-of-year torus) — label is a smooth function of the torus; geometry mode ties | Δ −1.0 to +0.5 pp (pilot −0.1 pp) | same | same |
| H6 | BT-5 (quakes) | The `deep_quakes` device loss (−8 pp at 1,024 shots) is shot resolution: BA recovers toward exact as shots grow; the loss to RF (−6.6 pp exact) does not change | FakeFez BA: 1,024 → ≈ exact − 8 pp; 4,096 → −5; 16,384 → −2.5; 65,536 → within 1 pp of exact; all below RF 0.835 | [e05](e05-quake-shots/README.md): fresh seeds 200–204, shipped `Qlassifier(backend="fake_fez", shots=S)` | BA at 65,536 shots < exact − 2 pp → shots do not explain the loss |
| H7 | BT-6 | The shipped engine on current `main` reproduces the stage-1 shipped rows exactly | 0 differing rows (BC, heart) | [e06](e06-regression/README.md) | any row differs beyond 1e-9 |

Ranking by expected gain: H2 (new evidence on the judging axis "hardware awareness", originality: first sweep of this kind), H1 (usefulness: wins on new real data), H3 (cost transparency), H6 (teaching value), H4/H5 (stretch; low prior).

### Added during iteration 1 (PRD amendment 2026-10-25, commit 69f4002): BT-3 is required
- The user has an IBM Open-plan token; BT-3 must run (≤ 3 QPU min for the stage) and the paper must document it in full. Protocol §2.5 rewritten; [e07](e07-hardware-point/README.md) designed with a dry run and a **pre-registered prediction before submission**:

| Id | Target | Hypothesis | Predicted effect | Smallest test | Kill criterion |
|---|---|---|---|---|---|
| H8 | BT-3 | Real Heron r2 readout of the 4-qubit, 0-two-qubit-gate circuits reproduces the FakeFez prediction on the seed-0 60/40 subset | variant 1 (exact train features + hardware test features) accuracy 0.880 ± 0.010 (35–36 of 40 rows), BA ≈ 0.88; variant 2 (hardware train + test) 0.880 ± 0.013; kernel r vs exact ≈ 0.998; QPU 79–110 s | one `SamplerV2` job, 3 PUBs × 100 rows × 1,024 shots on `ibm_fez` (`ibm_kingston` fallback) | variant-1 accuracy outside 0.880 ± 2 pp (≤ 34 or ≥ 37 rows) → BT-3 missed; job failure after one retry → recorded as not run |

## Iteration 1 — all targets, one pass (2026-10-25)
### Experiments
- [e01](e01-new-datasets/README.md) BT-1 · [e02](e02-noise-sweep/README.md) BT-2 · [e03](e03-cost-model/README.md) BT-4 · [e04](e04-geometry-screen/README.md) BT-5 screen · [e05](e05-quake-shots/README.md) H6 · [e06](e06-regression/README.md) BT-6 · [e07](e07-hardware-point/README.md) BT-3.
### Results (eval seeds 0–9 unless stated; Δ = Qlassifier − model, pp)
| Id | Predicted | Measured | Verdict |
|---|---|---|---|
| H1 spambase | BA 0.90–0.93, > ZZ, ≥ 0.888 | **0.911 ± 0.006**; ZZ 0.899 (+1.2 [+0.8, +1.6], 10/10); +2.3 pp vs published; ties GB 0.918 (−0.8 [−2.0, +0.4]); below MLP 0.918 (−0.7 [−1.2, −0.2]) | pass |
| H1 banknote | 0.99–1.00, > ZZ, ≥ 0.921 | **1.000 ± 0.001** (2 qubits); ZZ 0.955 (+4.4 [+2.7, +6.2], 10/10); +7.9 pp vs published; ties MLP / FQK-bw / separable at 1.000 | pass |
| H1 haberman | 0.55–0.62, > ZZ, ≥ 0.586 | 0.564 ± 0.031; ZZ 0.546 (+1.9 [−4.1, +7.9], 6/10 wins); −2.2 pp vs published 0.586; MLP 0.583; all 11 tuned models tie (0.546–0.583) | fail (reported) |
| H1 → BT-1 | ≥ 2 of 3 | 2 of 3 | **met** |
| H3 cost model | brackets 0.27 / 0.32 / 0.69 s | 0.271 ∈ [0.267, 0.365], 0.319 ∈ [0.268, 0.366], 0.691 ∈ [0.537, 0.724] s/circuit; counts = executed (1,707 = 3 × 569; 0 two-qubit gates) | confirmed → **BT-4 met** |
| H7 regression | 0 differing rows | 240 / 240 rows identical (BC, heart), configs and edges equal; stage-1 statuses unchanged | confirmed → BT-6 baseline holds |
| H2 noise sweep | part A drop 0.1–0.5 pp at s = 1, ≤ 2 pp at s = 3; FQK-bw −3 to −5 pp at s = 0, −20 at s = 1, worse beyond; ZZ at majority; AUC Q > FQK > ZZ | part A **+0.12 pp at every s** (worst +0.58; acc 0.949 vs 0.950 exact; kernel r 0.9984 → 0.9983); part B Qlassifier 0.900 at every s (exact 0.908); FQK-bw linear **−19.2 pp already at s = 0** (shot noise on a 60-point narrow-bandwidth Gram), −20.0 at s = 1, −21.7 at s = 3 (kernel r 0.995 → 0.872); ZZ 0.625 = majority at s = 0, 1, 3 (r 0.92 → 0.02). AUC acc 0.900 / 0.738 / 0.625 | confirmed → **BT-2 met**; FQK-bw's drop is shot-noise-dominated, not gate-noise-dominated (correction to the stage-1 reading) |
| H4 beijing_pm25 | Δ −1.5 to +1.0 vs best classical | 0.755 vs RBF-SVM 0.761: **−0.6 [−2.7, +1.4]**; twin 0.764; ZZ +18.8 | tie; gate failed → no confirmation |
| H5 solar_boulder | Δ −1.0 to +0.5 | **0.917 top mean**; vs RBF-SVM 0.912 **+0.5 [−0.4, +1.4]**; MLP +1.1, RF +1.7, GB +2.1; every quantum baseline −2.3 to −2.6 (CI > 0); ZZ, LR 0.500 | tie; gate failed → no confirmation; BT-5 missed |
| H6 quake shots | 1,024 → −8; 4,096 → −5; 16,384 → −2.5; 65,536 → within 1 pp; all below RF | BA drop 10.35 / 5.76 / 1.84 / **0.65 pp** (seeds 200–204); exact 0.766 vs RF 0.831 (−6.5 pp) | confirmed: shot resolution explains the device loss, not the loss to trees |
| H8 hardware point | 35–36 of 40 rows; QPU 79–110 s | `ibm_fez` job `dbesrqnpc3tc739mb92g`: **36 of 40 (0.900)**, BA 0.907, both variants; +2.0 pp vs the FakeFez prediction 0.880 (inside ± 2 pp and the shot-noise range); kernel r 0.998–0.999; Bloch error 0.020 (sim 0.018); **82 QPU s billed** (0.273 s/circuit) | confirmed → **BT-3 met** |
### Lessons
- The standard ZZ baseline is not always collapsed: on spambase (word frequencies squeezed to [0, π], mostly near 0) it scores 0.899, so the stage-1 "+23–38 pp over ZZ" is dataset-specific. Report ZZ per dataset; the fair claim is "best quantum model in-harness" plus the measured margin.
- Published quantum "leads" (haberman) vanish under equal tuning: every model ties within the seed spread; the honest result is a tie table.
- IBM's job-time model with device defaults is enough to bracket real billed times; repetition delay (250 µs × shots) dominates, circuit depth adds ≤ 12%. The real `ibm_fez` job landed at 0.273 s per circuit, 4% above the low bound.
- Scaling every FakeFez error rate up to 3× does not move Qlassifier at all (same 0.949 at every scale): with 0 two-qubit gates and depth ≤ 10 the only effect is a near-uniform shrink of each Bloch vector, which per-qubit renormalisation removes; what remains is the 1,024-shot binomial floor (|ΔR| ≈ 0.018). The fidelity-kernel baselines lose most of their accuracy to shot noise before any gate error is applied (FQK-bw −19 pp at s = 0), and ZZ's 433-CZ circuits lose the kernel itself (r → 0.02 at 3×).
- On real hardware the picture held: the device bias (|ΔR| 0.020) was at the simulated shot-noise floor and accuracy moved by one test row.
- Every geometry candidate screened at N = 4,000 with grouped splits ends in a tie with tuned RBF-SVM; the geometry prior beats the other quantum kernels and trees / MLP on the solar torus but not the plain RBF on raw angles unless the label straddles a seam.
### Decision
- Keep the shipped method unchanged (no method change was needed for any target). Loop closed after one iteration: BT-1, BT-2, BT-3, BT-4 met; BT-5 missed (stretch, pre-registered gate); BT-6 baseline holds. Phase 8: bundle `spambase` and `banknote`; add `PUBLISHED` entries; Target-scaled noise sweep; QPU-time bracket in the cost view; hardware path as in e07; report ZZ per dataset.

## Final decision — stage 2 method and evidence against every target
### Method
- **Unchanged from stage 1** (ADR 0001): Bloch-kernel core (2 standardised features per qubit, RX·RY, c₀ = 2/√max(d, 8)) + opt-in geometry mode; 3 Pauli settings per sample; per-qubit unit renormalisation; Gaussian kernel on Bloch vectors; SVM; self-tuning on exact features (0 circuits). No stage-2 experiment found a method change that pays; the stage-2 work is evidence, protocol and infrastructure: new datasets, a noise-scale sweep, a QPU-time model, a real-device point, and the shot-resolution rule.
- **Stage-2 additions for phase 8:** bundle `spambase` and `banknote` with their published numbers; noise sweep = Target-scaled FakeFez (§2.3) with the per-qubit path; cost view = exact counts + the [low, high] QPU-time bracket (§2.4); hardware path = dry run (plan, prediction, cost) → explicit confirmation → one `SamplerV2` job with full provenance (e07); shots guidance from the tuned kernel width (e05).

### Evidence against every target
| BT | Target | Measured | Status |
|---|---|---|---|
| BT-1 | 2 extra real datasets: > pinned ZZ, ≥ published quantum, one command | `spambase` 0.911 (ZZ 0.899, +1.2 pp, 10/10; published 0.888); `banknote` 1.000 (ZZ 0.955, +4.4 pp, 10/10; published 0.921); `haberman` 0.564 (ZZ 0.546, 6/10; published 0.586) reported as a miss; one command per dataset via the shipped engine ([e01](e01-new-datasets/README.md)) | **met** (2 of 3) |
| BT-2 | drop ≤ 2 pp at 1×; AUC over 0×–3× > FQK-bw and > ZZ | part A drop +0.12 pp at s = 1 (worst seed +0.58), flat from 0× to 3×; part B AUC accuracy 0.900 vs FQK-bw linear 0.738 vs ZZ 0.625 ([e02](e02-noise-sweep/README.md)) | **met** |
| BT-3 | hardware point, ≤ 3 QPU min, within ± 2 pp of the noisy-sim prediction, fully documented | `ibm_fez`, job `dbesrqnpc3tc739mb92g`, 2026-10-25 09:37–09:39 UTC, 300 circuits × 1,024 shots, **82 QPU s**; accuracy 0.900 (36/40) vs predicted 0.880 ± 0.010 → +2.0 pp; BA 0.907; kernel r 0.998; calibration snapshot, circuits, counts saved ([e07](e07-hardware-point/README.md)) | **met** |
| BT-4 | counts exact; QPU-time model within the published 0.27–0.69 s per circuit | counts = executed circuits (1,707 = 3 × 569; 0 two-qubit gates); model brackets 0.271 / 0.319 / 0.691 s; hardware check: 82 s billed vs 79–110 s predicted (0.273 s/circuit) ([e03](e03-cost-model/README.md), e07) | **met** |
| BT-5 (stretch) | beat every tuned classical model on ≥ 1 more real dataset, CI > 0 | `beijing_pm25` −0.6 [−2.7, +1.4] vs RBF-SVM; `solar_boulder` top mean BA but +0.5 [−0.4, +1.4] vs RBF-SVM; both below the pre-registered +1.0 pp gate, no confirmation run ([e04](e04-geometry-screen/README.md)); `deep_quakes` shots sweep does not change its loss to RF ([e05](e05-quake-shots/README.md)) | **missed** (stretch; ties) |
| BT-6 | stage-1 targets still pass | 240 / 240 stage-1 rows reproduced exactly on current `main` (BC, heart); ten statuses unchanged ([e06](e06-regression/README.md)); phase 9 re-runs `--noise` after the stage-2 changes | baseline holds |

### Failed ideas (kept)
- Geometry mode on Beijing PM2.5 (wind direction + hour + month circles with 5 plain columns, week-grouped): −0.6 pp vs RBF-SVM, 0.9 pp below its own classical twin; the pilot's +0.9 pp lead (N = 1,200, random splits) did not survive.
- Geometry mode on the solar torus: top mean BA and +2.3–2.6 pp over every quantum baseline, but only +0.5 pp over tuned RBF-SVM (seams far from the label boundary); below the gate.
- Haberman as a BT-1 dataset: every model ties (0.546–0.583); the published quantum lead is inside the seed spread.
- More shots for `deep_quakes`: removes the device loss (0.65 pp at 65,536 shots) but not the 6.5 pp loss to random forest; and 16k+ shots × 3 × 3,729 rows is ≈ 15 QPU hours.
- Stage-1 failed ideas stand (re-uploading, multi-scale, complex qubits, entanglement, class weights, noise-aware tuning, protein BT-6 variants).

### Flags for the user
1. **BT-5 missed (stretch, final for stage 2).** Pre-registered gate not met on either candidate; no re-selection. Claims stay at "top mean BA on solar; ties on Beijing; never beats every tuned classical model with CI > 0". Same situation as stage-1 BT-6; it changes no acceptance criterion unless BT-5 is read as required.
2. **BT-3 ran on real hardware on the main agent's GO** (PRD amendment 69f4002): 82 of the Open plan's 600 monthly QPU seconds used; one job, no retry. The ± 2 pp rule is a one-test-row tolerance at n = 40 (2.5 pp per row): the result passes it (+2.0 pp), but any single extra flipped row would have failed it. Phase 8 should state the rule as "within one test row and inside the simulator's shot-noise range" or raise n_test.
3. **Haberman** is reported below the published quantum number (0.564 vs 0.586); the published lead is not reproducible under equal tuning. The paper should say so plainly.
4. **The standard ZZ baseline does not collapse on every dataset** (spambase 0.899, banknote 0.955): the stage-1 "+23–38 pp" margin is dataset-specific; stage-2 margins are +1.2 and +4.4 pp. Report per dataset.
5. **Spambase: MLP beats Qlassifier significantly** (−0.7 [−1.2, −0.2] pp) although Qlassifier ties the best classical model (GB) by the leaderboard's tie rule. The dashboard's "tied with the best" band hides a significant loss to the second-best model; consider showing the pair CI against every model, not only the best.
6. Raw downloads were committed once by mistake (commit 1929d02, 8.6 MB) and untracked in 3dd2d30; they remain in git history. Rewriting `main` was not done (another process shares the tree).
7. Shipped-engine limit found: `bench.protocol.arrays` resolves circle geometry only; a sphere dataset (`deep_quakes`) cannot be registered for the engine without a small change (phase 8 note).

### Addendum (2026-10-25, after the loop closed): four more hardware seeds
- Seeds 1–4 ran on `ibm_fez` on the user's request (410 QPU s in all for 5 seeds); predictions computed after the runs by the same procedure. All 5 within 2 pp of the FakeFez prediction (mean deviation +0.3 pp, largest 2.0 pp), kernel r ≥ 0.998. Details: [e07 README](e07-hardware-point/README.md#seeds-14-added-2026-10-25-after-the-pre-registered-run-user-request). BT-3's one-seed caveat is now a one-device, one-day caveat.

### Recommendations for phase 8 (TDD)
- Datasets: bundle `spambase`, `banknote` (CC BY 4.0; `PUBLISHED` 0.888 / 0.921, arXiv:2604.18837); keep `haberman`, `beijing_pm25`, `solar_boulder` as reported-only results (paper), optionally `solar_boulder` as a 2-qubit teaching tile.
- Noise sweep (FR-4): `backends.scaled_device(name, s)` = Target scaling of §2.3 (errors × s, T1/T2 ÷ s), density-matrix Aer, per-qubit path for product circuits; sweep API `noise_sweep(dataset, scales=(0, 0.5, 1, 1.5, 2, 3))`; the dashboard chart = accuracy vs s with the exact line; AUC in the API contract.
- Cost view (FR-3): `Resources` + `qpu_seconds = (low, high)` from §2.4 with `default_rep_delay` and `estimate_duration`; label "estimate"; show the shots rule from the tuned width (e05) when the configuration sits on the narrow grid corner.
- Hardware (FR-2): `Qlassifier(backend="ibm:<name>")` / `qlassifier benchmark --hardware`: dry run (plan + prediction + cost) → explicit confirmation → one job → provenance JSON exactly as e07's `record.json`; `QiskitRuntimeService()` defaults only; results under `benchmarks/hardware/`.
- Statistics: keep one pre-registered test per target; print pair CIs against every model (flag 5).
