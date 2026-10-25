# e05 — noise robustness on IBM fake backends (BT-9), eval seeds

Up: [../README.md](../README.md) (protocol §2.8) · Code: [`../common/noise_archive.py`](../common/noise_archive.py) (archived e05 code; outputs redirected here) · Raw: [`partA.csv`](partA.csv), [`partA_stats.json`](partA_stats.json), [`partB_runs/`](partB_runs/) · Theory: §6 · Log: [learning-log](../learning-log.md)

- **Question:** does the Qlassifier core (`bloch_gauss`) keep its accuracy under IBM device noise (≤ 2 pp on each device) and lose less than the bandwidth-tuned fidelity kernel (FQK-bw)?
- **Setup:** Breast Cancer, PCA-8 → 4 qubits. `SamplerV2.from_backend(AerSimulator.from_backend(Fake…()))` for FakeTorino (Heron r1), FakeFez (Heron r2), FakeBrisbane (Eagle r3); `optimization_level=3`, `seed_transpiler=0`; 1,024 shots; sampler seed 11. Hyperparameters from exact-mode CV on the same training split; no re-tuning.
  - Part A: all 569 rows, eval seeds 0–4; 3 circuits per sample (+2 calibration circuits, ablation only). Shipped post-processing `norm` = unit Bloch vectors per qubit.
  - Part B: same stratified 60 train / 40 test subset for `bloch_gauss` and FQK-bw (linear ZZ, compute–uncompute, symmetric eigen-clip PSD repair), eval seeds 0–2, one process per (seed, device).
  - Run: `E05_DIR=… E05_MODELS_A=bloch_gauss noise_archive.py partA` (91 s); Part B: `E05_SEEDS=s E05_MODELS=bloch_gauss,fqk_bw_linear E05_OUT=partB_runs/s{s}_{device}.csv noise_archive.py partB {device}`.

## Part A result (569 rows, 5 eval seeds; exact acc 0.950, BA 0.942)
| Mode (post-processing) | Acc | Mean drop vs exact (pp) | Worst seed (pp) | BA | Kernel r vs exact |
|---|---|---|---|---|---|
| shots only (`norm`) | 0.946 | 0.35 | 1.17 | 0.941 | 0.998 |
| **FakeTorino (`norm`)** | **0.949** | **0.12** | 0.58 | 0.940 | 0.998 |
| **FakeFez (`norm`)** | **0.947** | **0.23** | 1.17 | 0.939 | 0.998 |
| **FakeBrisbane (`norm`)** | **0.945** | **0.47** | 1.17 | 0.937 | 0.998 |
| FakeTorino / Fez / Brisbane (`raw`) | 0.945 / 0.946 / 0.944 | 0.47 / 0.35 / 0.58 | 1.17 | — | 0.998 |
- Reproduces the archived run (0.1 / 0.2 / 0.5 pp). `mit+norm` = `norm` (readout offsets on the chosen qubits ≤ 0.3%).
- **BT-9 first clause met:** ≤ 2 pp on every device (max 0.47 pp mean, 1.17 pp worst seed).

## Part B result (paired 60/40 subset, 3 eval seeds; merged in [`partB.csv`](partB.csv))
| Model | Circuits (fit + predict) | 2Q gates / circuit | Exact acc | Drop Torino / Fez / Brisbane (pp) | Kernel r |
|---|---|---|---|---|---|
| **`bloch_gauss`** | **302** | **0** | **0.908** | **−0.8 / 0.8 / 0.0** | **0.998** |
| FQK-bw (linear ZZ) | 4,170 | 54 | 0.917 | 22.5 / 20.0 / 17.5 | 0.91–0.96 |
- 40 test points: 1 point = 2.5 pp. Identical to the archived Part B (same seeds, same code).
- FQK-bw keeps kernel r ≥ 0.91 but loses 17.5–22.5 pp: its tuned small bandwidth puts kernel differences below device noise. 1,067 s per device per seed on shared cores vs 3.4 s for `bloch_gauss`.
- **BT-9 second clause met:** Qlassifier drop (≤ 0.8 pp) is far below the FQK-bw drop (≥ 17.5 pp) on every device.

## Decision
- BT-9 met. Shipped noise handling unchanged: unit renormalisation only; no calibration circuits, no PSD repair, no C re-tuning.
