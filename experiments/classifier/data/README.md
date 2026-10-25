# Data files

UCI files: raw, unmodified. Structure-fit files: derived tables built by the groundwork scripts (section below).

Up: [../README.md](../README.md) · Loader: [`../common/lab.py`](../common/lab.py) (`load`) · Protocol: [README §2](../README.md#2-protocol-fixed-before-iteration-1)

| File | Dataset | Source | Licence | Rows × cols | SHA-256 |
|---|---|---|---|---|---|
| `ionosphere.data` | Ionosphere | UCI id 52, https://archive.ics.uci.edu/static/public/52/ionosphere.zip | CC BY 4.0 | 351 × 34 + label | `46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd` |
| `sonar.all-data` | Connectionist Bench (Sonar, Mines vs Rocks) | UCI id 151, https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip | CC BY 4.0 | 208 × 60 + label | `e90434cdbf00fcf93ffa911fe447ae25606979658e60f1d32e155c3b5240234d` |
| `processed.cleveland.data` | Heart Disease (Cleveland) | UCI id 45, https://archive.ics.uci.edu/static/public/45/heart+disease.zip | CC BY 4.0 | 303 × 13 + target | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| — | Breast Cancer Wisconsin (Diagnostic) | `sklearn.datasets.load_breast_cancer` (UCI id 17) | CC BY 4.0 (UCI); BSD-3 (scikit-learn copy) | 569 × 30 | — |

Downloaded 2026-10-24. Loader rules (adopted by the phase 8 TDD):
- Ionosphere: label `b` = 1 (126), `g` = 0. Column 2 is constant (all 0): dropped. d = 33.
- Sonar: label `R` (rock) = 1 (97), `M` = 0. d = 60.
- Heart: `?` = missing; 6 rows with missing `ca` / `thal` dropped → 297 rows. Target `num > 0` = 1 (137). Categorical codes kept as numbers. d = 13.
- Breast Cancer: malignant = 1 (212). d = 30.
- Citations: Sigillito et al. 1989 (Ionosphere); Gorman & Sejnowski 1988 (Sonar); Detrano et al. 1989 (Heart); Street, Wolberg, Mangasarian 1993 (Breast Cancer).

## Structure-fit datasets (added in iteration 2; groundwork: `docs/references/ideas/bloch-sphere-angular-data.md`)
| File | Dataset | Source | Licence | Rows × cols | SHA-256 |
|---|---|---|---|---|---|
| `deep_quakes.csv` | Global M ≥ 6 earthquakes 2000–2024; label `deep` = depth ≥ 70 km | USGS ANSS Comprehensive Earthquake Catalog (ComCat), https://earthquake.usgs.gov/fdsnws/event/1/ (type = earthquake) | U.S. public domain (USGS-authored data) | 3,729 × 5 (latitude, longitude, depth, mag, deep) | `1ead2a1b7119781adadd6cc51a11a5bb156e6dfcce9a1223b2cee7e013482f65` |
| `protein_strands.csv` | Backbone dihedrals φ/ψ of residues i−1, i, i+1; label = residue i in a `SHEET` record (strand) | RCSB PDB, 150 non-redundant chains (X-ray ≤ 1.6 Å, 1 entity, length 80–400, 30% identity clusters), https://files.rcsb.org/download/<ID>.pdb | CC0 1.0 (wwPDB data policy) | 3,000 × 10 (pdb, seq, resn, 6 angles, ss) | `b2bc77bf87364b9fc86e5a31e7b7b95263c531a7e220ef9a92f80a047dd54696` |

- Built by the groundwork scripts (`.cache/ideas/bloch-sphere-angular-data/`: `fetch_pdb.py`, `prep_data.py`; dihedrals from N, Cα, C coordinates, chain breaks skipped). Copied 2026-10-24.
- `deep_quakes`: all 3,729 events kept. Features used: latitude, longitude only (depth is the label source).
- `protein_strands`: label-free thinning of 28,406 residues: keep residues with `seq % 3 == 0` (no two kept residues of one chain share a dihedral value), then a fixed random sample of 3,000 (`numpy` seed 0). Strand share 21.1%. Splits are grouped by chain (`pdb` column, 150 chains; protocol §2.2, from iteration 4), so no chain is on both sides of a split.
- Citations: U.S. Geological Survey (2017), ANSS ComCat, doi:10.5066/F7MS3QZH; Berman et al. (2000), The Protein Data Bank, Nucleic Acids Res. 28:235–242.

## Diverse-chain protein set (added in iteration 5)
| File | Dataset | Source | Licence | Rows × cols | SHA-256 |
|---|---|---|---|---|---|
| `protein_div.csv` | Backbone dihedrals φ/ψ of residues i−2 … i+2; label = residue i in a `SHEET` record (strand, 24.9%) | RCSB PDB, 1,800 representatives of 30%-identity clusters (X-ray ≤ 1.6 Å, protein-only, 1 entity, length 80–400, release date ascending), legacy `.pdb.gz` | CC0 1.0 (wwPDB data policy) | 4,000 × 14 (pdb, seq, resn, 10 angles, ss); 1,773 chains, ≤ 3 rows per chain | `b1d01c7566dcc3ce19745107a037946dfd11c26e57d52a3a9aa99af2c6e44555` |

- Built by [`../e12-diverse-chains/fetch_pdb.py`](../e12-diverse-chains/fetch_pdb.py) and [`../e12-diverse-chains/build_dataset.py`](../e12-diverse-chains/build_dataset.py) (run with `python -I`; raw files kept outside the repo in `.cache/phase7/pdb_div/`). 1,799 chains parsed (1 skipped: no HELIX / SHEET annotation); 326,128 candidate residues.
- Label-free sampling: per chain up to 3 residues at sequence spacing ≥ 5 (no shared dihedral between rows for either feature set; numpy seed 0), then 4,000 rows (seed 0).
- Loaders: `protein_div6` (φ/ψ of i−1 … i+1, 6 circles) and `protein_div10` (i−2 … i+2, 10 circles); both use chain-grouped splits (`load_groups`).
