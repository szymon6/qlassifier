"""Compare the e06 rerun rows with the stage-1 shipped benchmark rows (chosen config, edge flag, metrics)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SHIPPED = HERE.parents[2] / "benchmarks"
NEW = Path(".cache/stage2-e06")
METRICS = ["balanced_accuracy", "accuracy", "roc_auc", "f1"]
report = {}
for ds in ("breast_cancer", "heart"):
    a = pd.read_csv(SHIPPED / ds / "runs.csv").set_index(["seed", "model"]).sort_index()
    b = pd.read_csv(NEW / ds / "runs.csv").set_index(["seed", "model"]).sort_index()
    common = a.index.intersection(b.index)
    diff = (a.loc[common, METRICS] - b.loc[common, METRICS]).abs()
    cfg_same = int((a.loc[common, "config"] == b.loc[common, "config"]).sum())
    edge_same = int((a.loc[common, "edge"] == b.loc[common, "edge"]).sum())
    report[ds] = {"rows": len(common), "rows_shipped": len(a), "rows_new": len(b), "max_abs_diff": diff.max().to_dict(),
                  "rows_within_1e-9": int((diff.max(axis=1) <= 1e-9).sum()), "config_equal": cfg_same, "edge_equal": edge_same}
    print(ds, json.dumps(report[ds]))
targets = json.loads((SHIPPED / "targets.json").read_text())
report["stage1_targets"] = {t["id"]: t["status"] for t in (targets if isinstance(targets, list) else targets.get("targets", []))}
(HERE / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
print(report["stage1_targets"])
