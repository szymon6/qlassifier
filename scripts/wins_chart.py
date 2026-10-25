"""The headline chart: Qlassifier vs the standard quantum baseline (and, in the paper, the best tuned classical model).

Reads the saved results (stage-1 ``benchmarks/`` and the stage-2 ``e01`` runs) and writes
``docs/paper/figures/wins.pdf`` (paper, Fig. 1, with the classical bars) and ``docs/assets/wins.png`` (README hero,
quantum only). Datasets are sorted by Qlassifier's gain over the standard quantum baseline, largest first.
Numbers come from the same files as ``paper_tables.py``; rerun after a benchmark:
.venv/bin/python scripts/wins_chart.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as pt

SERIES = (  # fixed categorical order and hues (validated with the dataviz palette checker)
    ("zz_qsvc", "Standard quantum baseline (ZZ QSVC)", "#eb6834"),
    ("qlassifier", "Qlassifier", "#2a78d6"),
    ("best_classical", "Best tuned classical model", "#1baf7a"),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"


def margin(row: dict[str, Any]) -> float:
    """Qlassifier minus the standard quantum baseline (ZZ QSVC), in BA."""
    return float(row["qlassifier"][0] - row["zz_qsvc"][0])


def rows(bench: Path | None = None, stage2: Path | None = None) -> list[dict[str, Any]]:
    """One row per dataset, largest ``margin`` first: label and the three BA means with CIs."""
    results = pt.load_results(bench or pt.BENCH)
    results2 = pt.load_stage2_results(stage2 or pt.STAGE2)
    out = []
    for ds, r in [*results.items(), *results2.items()]:
        h = r["highlights"]
        best = h["best_classical"]
        label = pt.LABEL.get(ds) or pt.STAGE2_DATASETS[ds]
        if ds in results2:
            label += "†"
        row: dict[str, Any] = {"label": label}
        for key, model_id in (("zz_qsvc", "zz_qsvc"), ("qlassifier", "qlassifier"), ("best_classical", best)):
            m = pt.metric(r, model_id)
            row[key] = (m["mean"], m.get("ci_low"), m.get("ci_high"))
        out.append(row)
    return sorted(out, key=margin, reverse=True)


def draw(data: list[dict[str, Any]], classical: bool = True) -> Any:
    """The figure: horizontal grouped bars, thin whiskers, direct labels on the two quantum bars.

    ``classical=False`` drops the best tuned classical model (the README hero); the paper keeps it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED})
    n = len(data)
    series = SERIES if classical else SERIES[:2]
    fig, ax = plt.subplots(figsize=(7.4, (0.62 if classical else 0.5) * n + 1.5))
    height, offsets = (0.24, (0.27, 0.0, -0.27)) if classical else (0.3, (0.16, -0.16))
    for (key, _, color), off in zip(series, offsets, strict=True):
        for i, row in enumerate(data):
            mean, lo, hi = row[key]
            y = n - 1 - i + off
            ax.barh(y, mean, height=height, color=color, edgecolor="white", linewidth=1.0)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color=INK, linewidth=0.8, solid_capstyle="butt", alpha=0.75)
            if key != "best_classical":
                ax.annotate(f"{mean:.3f}", (max(mean, hi or mean), y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK, annotation_clip=False)  # fmt: skip
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in reversed(data)], fontsize=9, color=INK)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Balanced accuracy (mean over the protocol seeds; whiskers: corrected 95% CI)", color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    handles = [Patch(facecolor=c, label=name) for _, name, c in series]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.6, borderaxespad=0.0)  # fmt: skip
    fig.text(0.01, 0.005, "† stage-2 research run, not yet bundled; same protocol, 10 seeds.",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main(argv: list[str] | None = None) -> int:
    """Write ``figures/wins.pdf`` (with the classical bars) and ``assets/wins.png`` (without), or both into --out."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=None, help="write both files into this folder instead")
    args = ap.parse_args(argv)
    data = rows()
    pdf = (args.out / "wins.pdf") if args.out else pt.PAPER / "figures" / "wins.pdf"
    png = (args.out / "wins.png") if args.out else pt.ROOT / "docs" / "assets" / "wins.png"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    draw(data).savefig(pdf, metadata={"CreationDate": None})
    draw(data, classical=False).savefig(png, dpi=200)
    print(f"wins_chart: wrote {pdf} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
