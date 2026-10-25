# Bundled datasets: sources, licences, citations

Loader: [`../datasets.py`](../datasets.py) · Provenance: [phase-7 data README](../../../experiments/classifier/data/README.md)

Every file is a byte-identical copy of the phase-7 file and is checked against its SHA-256 hash at load.
Breast Cancer is not bundled here: it loads from scikit-learn's own bundled copy (offline).

| Dataset id | File | Source | Licence | Rows × features | Positive class (label 1) | SHA-256 |
|---|---|---|---|---|---|---|
| `breast_cancer` | — (`sklearn.datasets.load_breast_cancer`) | UCI 17, https://archive.ics.uci.edu/dataset/17 | CC BY 4.0 (UCI); BSD-3-Clause (scikit-learn copy) | 569 × 30 | malignant (212) | — |
| `ionosphere` | `ionosphere.csv` (UCI `ionosphere.data`) | UCI 52, https://archive.ics.uci.edu/dataset/52 | CC BY 4.0 | 351 × 33 (constant column `a02` dropped) | bad (126) | `46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd` |
| `sonar` | `sonar.csv` (UCI `sonar.all-data`) | UCI 151, https://archive.ics.uci.edu/dataset/151 | CC BY 4.0 | 208 × 60 | rock (97) | `e90434cdbf00fcf93ffa911fe447ae25606979658e60f1d32e155c3b5240234d` |
| `heart` | `heart.csv` (UCI `processed.cleveland.data`) | UCI 45, https://archive.ics.uci.edu/dataset/45 | CC BY 4.0 | 297 × 13 (6 rows with `?` dropped) | disease, `num > 0` (137) | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| `protein_angles` | `protein_angles.csv` (phase-7 `protein_div.csv`) | RCSB PDB, https://www.rcsb.org (1,773 chains, 30% identity clusters) | CC0 1.0 (wwPDB data policy) | 4,000 × 10 (φ, ψ of residues i−2 … i+2, degrees) | strand: residue i in a PDB `SHEET` record (996) | `b1d01c7566dcc3ce19745107a037946dfd11c26e57d52a3a9aa99af2c6e44555` |

## Loader rules
- Label 1 = the positive (minority) class; zero-variance columns are dropped (only Ionosphere `a02`).
- Heart: `?` marks missing values; the 6 affected rows are dropped. Categorical codes stay numeric.
- Protein: features = backbone dihedrals in degrees; `groups` = PDB chain id (no chain on both sides of a split);
  `geometry` = every column is a circle with period 360°.

## Citations
- **Breast Cancer Wisconsin (Diagnostic).** W. Wolberg, O. Mangasarian, N. Street, W. Street (1993), UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B. Paper: W. N. Street, W. H. Wolberg, O. L. Mangasarian, "Nuclear feature extraction for breast tumor diagnosis", Proc. SPIE 1905, 861–870 (1993).
- **Ionosphere.** V. Sigillito, S. Wing, L. Hutton, K. Baker (1989), UCI Machine Learning Repository, https://doi.org/10.24432/C5W01B. Paper: "Classification of radar returns from the ionosphere using neural networks", Johns Hopkins APL Technical Digest 10, 262–266 (1989).
- **Sonar.** T. Sejnowski, R. Gorman, UCI Machine Learning Repository, https://doi.org/10.24432/C5T01Q. Paper: R. P. Gorman, T. J. Sejnowski, "Analysis of hidden units in a layered network trained to classify sonar targets", Neural Networks 1, 75–89 (1988).
- **Heart Disease (Cleveland).** A. Janosi, W. Steinbrunn, M. Pfisterer, R. Detrano (1989), UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X. Paper: R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease", American Journal of Cardiology 64, 304–310 (1989).
- **Protein Data Bank.** H. M. Berman et al., "The Protein Data Bank", Nucleic Acids Research 28, 235–242 (2000). Data under CC0 1.0 (wwPDB policy).
