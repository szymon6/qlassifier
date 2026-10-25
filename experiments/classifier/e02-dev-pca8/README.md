# e02 — richer spectra and class-balanced loss (H3, H4, H5), PCA-8, dev seeds

Up: [../README.md](../README.md) · Candidates: [`../common/cands.py`](../common/cands.py) (`spectrum_specs`, `balanced_specs`) · Runner: [`../common/run_x.py`](../common/run_x.py) · Raw: [`results.csv`](results.csv) · Summary: [`summary.md`](summary.md) · Log: [learning-log](../learning-log.md) (iteration 1)

- **Hypotheses:** H3 multi-scale Bloch features (`bloch_ms`: angle scales c₀/2, c₀, 2c₀, 9 circuits/sample) and H4 data re-uploading ×2 (`bloch_reup2`) lift mean BA ≥ +0.5 pp over `bloch_gauss`; H5 class-balanced SVM loss (`bloch_gauss_bal`) lifts Breast Cancer BA ≥ +0.5 pp, with balanced classical controls run alongside.
- **Setup:** protocol §2, recipe `pca8`, 4 datasets × dev seeds 100–109, 17 models (5 classical + 4 balanced classical controls, separable, PQK, FQK-bw linear, `bloch_gauss`, 3 candidates). Encodings verified vs Qiskit ([e00](../e00-kernel-checks/README.md), `check_candidates.py`). Run: `run_x.py results.csv --seeds 100-109 --k 8 --models … --extra cands:spectrum_specs cands:balanced_specs` (1,018 s on shared cores).

## Result (BA, mean over 10 dev seeds)
| Model | BC | Heart | Iono | Sonar | Mean | Δ mean vs `bloch_gauss` |
|---|---|---|---|---|---|---|
| **`bloch_gauss`** | 0.965 | 0.798 | 0.910 | 0.831 | **0.876** | — |
| `bloch_ms` (H3) | 0.958 | 0.811 | 0.912 | 0.810 | 0.873 | −0.3 pp |
| `bloch_gauss_bal` (H5) | 0.968 | 0.803 | 0.919 | 0.802 | 0.873 | −0.3 pp |
| `bloch_reup2` (H4) | 0.894 | 0.776 | 0.896 | 0.719 | 0.821 | −5.5 pp |
| MLP (best classical mean) | 0.968 | 0.813 | 0.900 | 0.815 | 0.874 | — |
| RBF-SVM / balanced | 0.966 / 0.971 | 0.795 / 0.809 | 0.905 / 0.919 | 0.796 / 0.792 | 0.865 / 0.873 | — |
| PQK / separable | 0.971 / 0.971 | 0.816 / 0.807 | 0.900 / 0.901 | 0.801 / 0.775 | 0.872 / 0.863 | — |

- Δ `bloch_gauss` vs best classical (corrected 95% CI): BC −0.7 [−3.2, +1.8] (LR), Heart −1.5 [−7.1, +4.1] (MLP), Iono −0.1 [−4.4, +4.2] (RF), Sonar +1.7 [−6.8, +10.1] (MLP, wins 9/10 splits). Ties everywhere.
- Balanced loss: Bloch +0.3 / +0.4 / +1.0 / −3.0 pp (BC / Heart / Iono / Sonar); RBF-SVM control +0.5 / +1.4 / +1.4 / −0.3 pp. No relative edge for the quantum model.
- Re-uploading doubles every frequency at the same angle scale: kernel too narrow in input space; selection cannot recover it (−7.1 pp on BC).
- Split variance is large: `bloch_gauss` BC BA 0.965 here vs 0.948 on eval seeds (same model, other splits).

## Decision
- H3 dropped (−0.3 pp, 3× circuits). H4 dropped (−5.5 pp). H5 dropped (BC +0.3 pp < +0.5; classical controls gain as much or more).
- Keep `bloch_gauss` as the generic Qlassifier core.
