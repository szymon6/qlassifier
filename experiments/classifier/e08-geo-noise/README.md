# e08 — geometry mode under IBM device noise (H10) + noise-aware tuning check, fresh seeds

Up: [../README.md](../README.md) (protocol §2.8) · Code: [`geo_noise.py`](geo_noise.py), [`geo_noise_aware.py`](geo_noise_aware.py) · Raw: `protein_strands_fake_*.csv`, [`aware_deep_quakes_fake_fez.csv`](aware_deep_quakes_fake_fez.csv) · Exact results and configs: [e07](../e07-structure-fit/README.md) · Groundwork noise probe: `bloch-sphere-angular-data.md` §6 · Log: [learning-log](../learning-log.md) (iteration 3)

- **Hypothesis H10:** the BT-6 win on `protein_strands` survives IBM device noise at 1,024 shots (6 qubits, 0 two-qubit gates): BA drop ≤ 0.5 pp; kill if > 2 pp on any device.
- **Check (coordinator's suggestion):** on `deep_quakes` the groundwork saw ~9 pp loss under shot noise. Does noise-aware tuning (CV on the *measured* training features; 0 extra circuits, since geometry features need no per-fold preprocessing) recover it? Is the loss an exact-fit / noisy-predict mismatch?
- **Setup:** circuits per qubit: circle RY(2πx/360°); sphere RY(90° − lat)·RZ(lon); all qubits measured in X, Y, Z (3 circuits per sample); unit renormalisation per qubit. `SamplerV2.from_backend(AerSimulator.from_backend(Fake…()))`, opt level 3, 1,024 shots, sampler seed 11. Train and test features are both measured on the device. Configs (m, C) from exact-mode CV in e07 (training data only).
  - Protein: fresh seeds 200–204, FakeTorino / FakeFez / FakeBrisbane, 9,000 circuits per run. Quakes: seeds 200–202, FakeFez.

## H10 result: `protein_strands` (5 fresh seeds; exact BA 0.885, reproduces e07 exactly)
| Device | BA | Mean BA drop (pp) | Worst seed (pp) | Acc drop (pp) | Kernel r vs exact | 2Q gates / depth |
|---|---|---|---|---|---|---|
| FakeTorino (Heron r1) | 0.881 | 0.31 | 1.26 | 0.31 | 0.9998 | 0 / 8 |
| FakeFez (Heron r2) | 0.878 | 0.67 | 1.53 | 0.51 | 0.9998 | 0 / 8 |
| FakeBrisbane (Eagle r3) | 0.883 | 0.11 | 1.32 | 0.04 | 0.9998 | 0 / 8 |
- Noisy geometry mode (0.878–0.883) stays above the best classical model's exact score on the same splits (MLP 0.870). These are residue-level splits (e07); the noise drop itself does not depend on the split type, but the BT-6 margin does (see [e11](../e11-grouped-split/README.md)).
- **H10 confirmed** (≤ 2 pp everywhere; FakeFez 0.67 pp slightly above the 0.5 pp guess). The BT-6 showcase is hardware-ready at 1,024 shots.

## Noise-aware tuning check: `deep_quakes`, FakeFez (3 fresh seeds)
| Variant | BA | Acc | Chosen (m, C̃) |
|---|---|---|---|
| exact | 0.767 | 0.880 | (4, 1,000) |
| measured features, exact-mode config | 0.685 | 0.834 | (4, 1,000) |
| measured features, config tuned on measured features | 0.685 | 0.834 | (4, 1,000) on 3/3 seeds |
- Noise-aware CV picks the same narrowest corner: the label needs a few-degree kernel even under noise. No fit / predict mismatch either (both sides measured, same config).
- The 8.2 pp loss is resolution-limited: one Bloch component read with 1,024 shots has error ≈ 1/√1024 ≈ 1.8°, comparable to the needed width. Groundwork shot sweep: 65,536 shots bring it back to within 1.6 pp.
- **Decision:** noise-aware tuning dropped (no effect, 0 extra circuits but extra complexity). Quakes stay a simulator showcase; the hardware showcase is `protein_strands`.
- Product rule for phase 8: warn when the tuned kernel width is below the shot-noise resolution (≈ 1/√shots per Bloch component) and suggest more shots.
