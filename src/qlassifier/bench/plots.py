"""Matplotlib figures of a ``BenchmarkResult``: leaderboard, noise, cost (TDD §5.5, §7.12).

Colour (Okabe-Ito, colour-blind safe) + shape by family, never colour alone: Qlassifier vermillion circle, other
quantum reddish purple diamond, classical blue square, majority class / twin grey. Figures are built with the
object API (no pyplot state), so saving works headless and ``plot()`` returns a figure for notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from qlassifier.bench.stats import clip_unit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qlassifier.bench.result import BenchmarkResult

__all__ = ["plot", "save_all"]

OURS, QUANTUM, CLASSICAL, GREY = "#D55E00", "#CC79A7", "#0072B2", "#8A8A8A"
DEVICE_LABELS = {
    "exact": "exact",
    "shots": "shots",
    "fake_torino": "Torino",
    "fake_fez": "Fez",
    "fake_brisbane": "Brisbane",
}
METRIC_LABELS = {"balanced_accuracy": "balanced accuracy", "accuracy": "accuracy", "roc_auc": "ROC-AUC", "f1": "F1"}


def _style(model_id: str, family: str) -> tuple[str, str]:
    if model_id == "qlassifier":
        return OURS, "o"
    if family == "quantum":
        return QUANTUM, "D"
    if family == "classical":
        return CLASSICAL, "s"
    return GREY, "o"


def _figure(ax: Axes | None, size: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax  # type: ignore[return-value]
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, layout="constrained")
    return fig, fig.add_subplot()


def _clean(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def leaderboard(result: BenchmarkResult, metric: str = "balanced_accuracy", ax: Axes | None = None) -> Figure:
    """One row per model, sorted by mean: per-seed dots, corrected 95% CI bar, mean mark; majority class as a rule."""
    rows = [m for m in result.models if m.id != "trivial"]
    rows.sort(key=lambda m: getattr(m.metrics, metric).mean)
    fig, ax = _figure(ax, (7.5, 0.42 * len(rows) + 1.4))
    if not rows:  # a run stopped before any model finished
        ax.text(0.5, 0.5, "no finished runs", ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()
        return fig
    per_seed: dict[str, list[float]] = {}
    for r in result.runs:
        v = getattr(r, metric)
        if v is not None:
            per_seed.setdefault(r.model, []).append(float(v))
    for y, m in enumerate(rows):
        s = getattr(m.metrics, metric)
        color, marker = _style(m.id, m.family)
        if s.tied_with_best:
            ax.axhspan(y - 0.45, y + 0.45, color=OURS if m.id == "qlassifier" else "#000000", alpha=0.05, lw=0)
        pts = per_seed.get(m.id, [])
        ax.scatter(pts, np.full(len(pts), y), s=16, color=color, alpha=0.35, lw=0)
        if s.ci_low is not None and s.ci_high is not None:  # clipped to the metric's range (TDD §7.7)
            ax.plot([clip_unit(s.ci_low), clip_unit(s.ci_high)], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.scatter([s.mean], [y], s=48 if m.id == "qlassifier" else 34, color=color, marker=marker, zorder=3)
    ax.set_yticks(range(len(rows)), [m.name for m in rows])
    for label, m in zip(ax.get_yticklabels(), rows, strict=True):
        if m.id == "qlassifier":
            label.set_fontweight("bold")
            label.set_color(OURS)
    trivial = next((m for m in result.models if m.id == "trivial"), None)
    if trivial is not None:
        ax.axvline(getattr(trivial.metrics, metric).mean, color=GREY, ls="--", lw=1)
    ax.set_ylim(-1.2, len(rows) - 0.4)
    lows = [
        clip_unit(v)
        for m in rows
        for v in [getattr(m.metrics, metric).ci_low, getattr(m.metrics, metric).mean]
        if v is not None
    ]
    ax.set_xlim(max(0.0, min(lows) - 0.03), 1.0 + 0.005)
    ax.set_xlabel(f"{METRIC_LABELS[metric]} (mean, corrected 95% CI, {len(result.protocol.seeds)} seeds)", fontsize=9)
    ax.set_title(result.dataset.name, fontsize=11, loc="left")
    _clean(ax)
    return fig


def noise(result: BenchmarkResult, ax: Axes | None = None) -> Figure:
    """Accuracy lost vs exact (pp) on shots and three IBM device noise models; one line per (model, part)."""
    fig, ax = _figure(ax, (6.5, 3.6))
    study = result.noise
    order = ["exact", "shots", "fake_torino", "fake_fez", "fake_brisbane"]
    if study is None:
        ax.text(0.5, 0.5, "no noise study", ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()
        return fig
    lines: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for s in study.summary:
        lines.setdefault((s.part, s.model), []).append((order.index(s.device), s.drop_mean_pp))
    family = {m.id: m.family for m in result.models}
    for (part, model), pts in sorted(lines.items()):
        pts.sort()
        color, marker = _style(model, family.get(model, "quantum"))
        lw = 2.4 if part == "A" else 1.2
        name = next((m.name for m in result.models if m.id == model), model)
        label = f"{name} ({'all rows' if part == 'A' else '60/40'})"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, marker=marker, ms=5, label=label)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(len(order)), [DEVICE_LABELS[o] for o in order])
    ax.invert_yaxis()
    ax.set_ylabel("accuracy lost vs exact (pp)", fontsize=9)
    ax.set_title(f"{result.dataset.name}: noise (1,024 shots)", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    return fig


def cost(result: BenchmarkResult, ax: Axes | None = None) -> Figure:
    """Circuits to train + predict (log; hatched = on-device tuning), qubits, two-qubit gates (FakeTorino)."""
    from matplotlib.figure import Figure

    study = result.cost
    if ax is not None:
        fig = ax.figure
        axes = [ax]
    else:
        fig = Figure(figsize=(9, 3.2), layout="constrained")
        axes = list(fig.subplots(1, 3))
    if study is None:
        axes[0].text(0.5, 0.5, "no cost study", ha="center", va="center", transform=axes[0].transAxes, color=GREY)
        for a in axes:
            a.set_axis_off()
        return fig  # type: ignore[return-value]
    names = {m.id: m.name for m in result.models}
    models = study.models
    labels = [names.get(m.model, m.model) for m in models]
    colors = [_style(m.model, "quantum")[0] for m in models]
    y = np.arange(len(models))
    a0 = axes[0]
    fit = [m.circuits_fit_predict for m in models]
    tune = [m.tuning_circuits_on_device for m in models]
    a0.barh(y, fit, color=colors)
    a0.barh(y, tune, left=fit, color="none", edgecolor=colors, hatch="///")
    a0.set_xscale("log")
    a0.set_xlim(10 ** np.floor(np.log10(max(min(fit), 1))) / 2, max(f + t for f, t in zip(fit, tune, strict=True)) * 8)
    for yi, f, t in zip(y, fit, tune, strict=True):
        a0.annotate(f"{f + t:,}", (f + t, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    a0.set_yticks(y, labels)
    a0.set_xlabel("circuits (train + predict; hatched: tuning)", fontsize=9)
    _clean(a0)
    if len(axes) == 3:
        qubits = [m.qubits for m in models]
        gates = [m.two_qubit_gates.get("fake_torino", 0) for m in models]
        for a, vals, label in ((axes[1], qubits, "qubits"), (axes[2], gates, "two-qubit gates / circuit (Torino)")):
            a.barh(y, vals, color=colors)
            a.set_yticks(y, [])
            a.set_xlabel(label, fontsize=9)
            a.set_xlim(0, max(max(vals), 1) * 1.25)
            for yi, v in zip(y, vals, strict=True):
                a.annotate(f"{v:,}", (v, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
            _clean(a)
    fig.suptitle(f"{result.dataset.name}: cost", fontsize=11, x=0.02, ha="left")
    return fig  # type: ignore[return-value]


def plot(
    result: BenchmarkResult, kind: str = "leaderboard", metric: str = "balanced_accuracy", ax: Any = None
) -> Figure:
    """Dispatch to :func:`leaderboard`, :func:`noise` or :func:`cost`."""
    if kind == "leaderboard":
        return leaderboard(result, metric, ax)
    if kind == "noise":
        return noise(result, ax)
    if kind == "cost":
        return cost(result, ax)
    from qlassifier.errors import InvalidParameterError

    raise InvalidParameterError(f"Unknown plot kind '{kind}'.", "Use 'leaderboard', 'noise' or 'cost'.")


def save_all(result: BenchmarkResult, directory: Path) -> list[Path]:
    """Write every available figure as PNG (160 dpi) and PDF."""
    figures = {"leaderboard": leaderboard(result)}
    if result.noise is not None:
        figures["noise"] = noise(result)
    if result.cost is not None:
        figures["cost"] = cost(result)
    written = []
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            path = directory / f"{name}.{ext}"
            fig.savefig(path, dpi=160, metadata={"CreationDate": None} if ext == "pdf" else None)
            written.append(path)
    return written
