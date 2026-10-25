| Dataset | Qlassifier BA ± sd | ZZ QSVC BA | Δ vs ZZ [CI] (wins) | Published quantum BA | Qlassifier − published | Best classical (BA) | Δ vs best classical [CI] | Tied with best? | BT-1 |
|---|---|---|---|---|---|---|---|---|---|
| `spambase` | 0.911 ± 0.006 | 0.899 | +1.2 [+0.8, +1.6] (10/10) | 0.888 | +2.3 pp | gradient_boosting (0.918) | -0.8 [-2.0, +0.4] | yes | pass |
| `banknote` | 1.000 ± 0.001 | 0.955 | +4.4 [+2.7, +6.2] (10/10) | 0.921 | +7.9 pp | mlp (1.000) | +0.0 [-0.2, +0.2] | yes | pass |
| `haberman` | 0.564 ± 0.031 | 0.546 | +1.9 [-4.1, +7.9] (6/10) | 0.586 | -2.2 pp | mlp (0.583) | -1.8 [-8.1, +4.4] | yes | fail |

BT-1: 2 of 3 datasets pass both conditions (rule: ≥ 2) → **met**.
