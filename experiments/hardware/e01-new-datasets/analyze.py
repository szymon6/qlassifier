"""e01 aggregation: BT-1 table over the three new UCI sets from the shipped result.json files.

Up: README.md · Protocol: ../README.md §2.2 · Run: .venv/bin/python experiments/hardware/e01-new-datasets/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = {"spambase": 0.888, "banknote": 0.921, "haberman": 0.586}
CLASSICAL = {"logistic_regression", "rbf_svm", "random_forest", "gradient_boosting", "mlp"}
lines = ["| Dataset | Qlassifier BA ± sd | ZZ QSVC BA | Δ vs ZZ [CI] (wins) | Published quantum BA | Qlassifier − published | Best classical (BA) | Δ vs best classical [CI] | Tied with best? | BT-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
verdicts = {}
for ds in ("spambase", "banknote", "haberman"):
    p = HERE / ds / "result.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    models = {m["id"]: m for m in r["models"]}
    comps = {c["model"]: c["metrics"]["balanced_accuracy"] for c in r["comparisons"]}
    q = models["qlassifier"]["metrics"]["balanced_accuracy"]
    zz = models["zz_qsvc"]["metrics"]["balanced_accuracy"]
    dz = comps["zz_qsvc"]
    best = max((m for m in r["models"] if m["id"] in CLASSICAL), key=lambda m: m["metrics"]["balanced_accuracy"]["mean"])
    db = comps[best["id"]]
    pub = PUBLISHED[ds]
    cond_zz = q["mean"] > zz["mean"] and dz["win_rate"] >= 0.8
    cond_pub = q["mean"] >= pub
    verdicts[ds] = cond_zz and cond_pub
    lines.append(
        f"| `{ds}` | {q['mean']:.3f} ± {q['std']:.3f} | {zz['mean']:.3f} | {100*dz['mean_diff']:+.1f} [{100*dz['ci_low']:+.1f}, {100*dz['ci_high']:+.1f}] ({round(dz['win_rate']*dz['n'])}/{dz['n']}) | {pub:.3f} | {100*(q['mean']-pub):+.1f} pp | {best['id']} ({best['metrics']['balanced_accuracy']['mean']:.3f}) | {100*db['mean_diff']:+.1f} [{100*db['ci_low']:+.1f}, {100*db['ci_high']:+.1f}] | {'yes' if q['tied_with_best'] else 'no'} | {'pass' if verdicts[ds] else 'fail'} |"
    )
n = sum(verdicts.values())
lines.append("")
lines.append(f"BT-1: {n} of {len(verdicts)} datasets pass both conditions (rule: ≥ 2) → **{'met' if n >= 2 else 'missed'}**.")
out = "\n".join(lines)
(HERE / "bt1_table.md").write_text(out + "\n")
print(out)
for ds in ("spambase", "banknote", "haberman"):
    s = HERE / ds / "summary.txt"
    if s.exists():
        print(f"\n--- {ds}\n" + s.read_text().split("\n\n")[0])
