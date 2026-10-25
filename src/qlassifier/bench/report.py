"""Benchmark targets BT-1...BT-10 -> ``REPORT.md``, ``targets.json``, ``summary.md``, ``environment.json`` (TDD §7.10).

Rules are the PRD §6 targets as measured by TDD §7.10. A target whose inputs were not run reads ``not-measured``;
BT-6 is ``report-only`` (ADR 0004). BT-8 judges the core encoding by the PRD's ceil(d/2) qubits and geometry mode
by the ADR 0005 rule (one qubit per circle or direction; approved 2026-10-25). BT-10 is measured on a default Breast
Cancer run (this run, or ``bt10/``).
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from qlassifier.bench.protocol import PUBLISHED, arrays
from qlassifier.bench.result import BenchmarkResult
from qlassifier.bench.stats import MIN_TIE_SEEDS, clip_unit
from qlassifier.encoding import Encoding, Geometry

__all__ = ["STATUSES", "Target", "evaluate", "geometry_qubit_limit", "write_report"]

Status = Literal["met", "missed", "report-only", "not-measured"]
STATUSES: tuple[Status, ...] = ("met", "missed", "report-only", "not-measured")
UCI = ("breast_cancer", "ionosphere", "sonar", "heart")
ALL = (*UCI, "protein_angles")
DEVICES = ("fake_torino", "fake_fez", "fake_brisbane")
QUANTUM = ("qlassifier", "zz_qsvc", "fqk_bandwidth_full", "fqk_bandwidth_linear", "pqk", "separable")
CLASSICAL = ("logistic_regression", "rbf_svm", "random_forest", "gradient_boosting", "mlp")
BT10_LIMIT_S = 120.0
# What a run must contain for each target to be measured (REPORT.md says so instead of a bare "not-measured").
NEEDS = {
    "BT-1": "`breast_cancer`",
    "BT-2": "all 5 datasets",
    "BT-3": "the 4 UCI datasets",
    "BT-4": "the 4 UCI datasets",
    "BT-5": "the 4 UCI datasets",
    "BT-6": "—",
    "BT-7": "`breast_cancer` (cost study)",
    "BT-8": "any dataset (cost study)",
    "BT-9": "`breast_cancer --noise`",
    "BT-10": "`qlassifier benchmark breast_cancer` with defaults (10 seeds, exact), or a `bt10/` folder",
}


@dataclass(frozen=True)
class Target:
    """One benchmark target: what was measured, the pass rule, and the verdict."""

    id: str
    measured: str
    rule: str
    status: Status

    def to_dict(self) -> dict[str, str]:
        """JSON-ready dict."""
        return {"id": self.id, "measured": self.measured, "rule": self.rule, "status": self.status}


def _mean(r: BenchmarkResult, model: str, metric: str = "balanced_accuracy") -> float | None:
    try:
        return float(getattr(r.model(model).metrics, metric).mean)
    except KeyError:
        return None


def _complete(results: Mapping[str, BenchmarkResult], ids: tuple[str, ...]) -> bool:
    return all(i in results and results[i].status == "complete" for i in ids)


def _bt1(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "Breast Cancer: mean accuracy Qlassifier - ZZ QSVC >= 20 pp"
    r = results.get("breast_cancer")
    q, z = (_mean(r, "qlassifier", "accuracy"), _mean(r, "zz_qsvc", "accuracy")) if r else (None, None)
    if r is None or q is None or z is None:
        return Target("BT-1", "—", rule, "not-measured")
    d = 100 * (q - z)
    return Target("BT-1", f"{q:.3f} vs {z:.3f} → {d:+.1f} pp", rule, "met" if d >= 20 else "missed")


def _bt2(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "mean BA Qlassifier > ZZ QSVC on all 5 datasets; paired win rate >= 0.8 on each"
    parts, ok, missing = [], True, False
    for ds in ALL:
        r = results.get(ds)
        comp = next((c for c in r.comparisons if c.model == "zz_qsvc"), None) if r else None
        if comp is None:
            missing = True
            continue
        d = comp.metrics.balanced_accuracy
        parts.append(f"{ds} {100 * d.mean_diff:+.1f} pp ({round(d.win_rate * d.n)}/{d.n})")
        ok &= d.mean_diff > 0 and d.win_rate >= 0.8
    if missing:
        return Target("BT-2", "; ".join(parts) or "—", rule, "not-measured")
    return Target("BT-2", "; ".join(parts), rule, "met" if ok else "missed")


def _vs_published(results: Mapping[str, BenchmarkResult]) -> tuple[list[tuple[str, float, float]], bool]:
    rows: list[tuple[str, float, float]] = []
    for ds in UCI:
        r = results.get(ds)
        q = _mean(r, "qlassifier") if r else None
        if q is None:
            return rows, False
        rows.append((ds, q, PUBLISHED[ds].value))
    return rows, True


def _bt3_bt4(results: Mapping[str, BenchmarkResult]) -> tuple[Target, Target]:
    rule3 = "mean BA >= published - 1 pp on >= 3 of 4 UCI sets (0.968 / 0.911 / 0.813 / 0.804)"
    rule4 = "mean BA >= published on >= 2 of 4 UCI sets"
    rows, full = _vs_published(results)
    text = "; ".join(f"{ds} {q:.3f} ({100 * (q - p):+.1f} pp)" for ds, q, p in rows) or "—"
    if not full:
        return Target("BT-3", text, rule3, "not-measured"), Target("BT-4", text, rule4, "not-measured")
    n3 = sum(q >= p - 0.01 for _, q, p in rows)
    n4 = sum(q >= p for _, q, p in rows)
    return (
        Target("BT-3", f"{text} → {n3}/4", rule3, "met" if n3 >= 3 else "missed"),
        Target("BT-4", f"{text} → {n4}/4", rule4, "met" if n4 >= 2 else "missed"),
    )


def _bt5(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "mean BA over the 4 UCI sets: Qlassifier highest among the in-harness quantum models"
    if not all(ds in results for ds in UCI):
        return Target("BT-5", "—", rule, "not-measured")
    means = {}
    for m in QUANTUM:
        vals = [_mean(results[ds], m) for ds in UCI]
        if all(v is not None for v in vals):
            means[m] = sum(v for v in vals if v is not None) / len(UCI)
    if "qlassifier" not in means or len(means) < len(QUANTUM):
        return Target("BT-5", "—", rule, "not-measured")
    ordered = sorted(means.items(), key=lambda kv: -kv[1])
    text = ", ".join(f"{m} {v:.3f}" for m, v in ordered)
    return Target("BT-5", text, rule, "met" if ordered[0][0] == "qlassifier" else "missed")


def _bt6(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "≥ 1 dataset: mean BA above every tuned classical model and corrected 95% CI of Δ vs best classical > 0"
    parts = []
    for ds in ALL:
        r = results.get(ds)
        if r is None or r.highlights.vs_best_classical is None or r.highlights.best_classical is None:
            continue
        d = r.highlights.vs_best_classical
        q = _mean(r, "qlassifier")
        above = q is not None and all((v := _mean(r, c)) is None or q > v for c in CLASSICAL)
        lo = f"{100 * d.ci_low:+.1f}" if d.ci_low is not None else "?"
        hi = f"{100 * d.ci_high:+.1f}" if d.ci_high is not None else "?"
        verdict = "CI > 0" if (above and d.ci_low is not None and d.ci_low > 0) else "tie"
        parts.append(f"{ds} vs {r.highlights.best_classical}: {100 * d.mean_diff:+.1f} [{lo}, {hi}] pp ({verdict})")
    return Target("BT-6", "; ".join(parts) or "—", rule + " (stretch; report-only, ADR 0004)", "report-only")


def _bt7(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "Breast Cancer: >= 100× fewer circuits than the bandwidth-tuned fidelity-kernel pipeline (incl. its search)"
    r = results.get("breast_cancer")
    ratio = r.highlights.circuit_ratio_vs_fqk_pipeline if r else None
    if r is None or r.cost is None or ratio is None:
        return Target("BT-7", "—", rule, "not-measured")
    by = {m.model: m for m in r.cost.models}
    q, f = by["qlassifier"], by.get("fqk_bandwidth_linear") or by["fqk_bandwidth_full"]
    text = f"{q.circuits_fit_predict:,} vs {f.circuits_fit_predict + f.tuning_circuits_on_device:,} → {ratio:,.0f}×"
    return Target("BT-7", text, rule, "met" if ratio >= 100 else "missed")


def geometry_qubit_limit(plan: Encoding) -> int:
    """ADR 0005 qubit limit of an encoding plan: 1 per circle, 1 per (lat, lon) pair, ceil(d_plain / 2) for the rest."""
    return sum(kind in ("circle", "sphere") for kind in plan.kinds) + math.ceil(len(plan.plain) / 2)


def _bt8(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = (
        "per circuit on FakeTorino / FakeFez / FakeBrisbane: core encoding qubits <= ceil(d/2); geometry mode: "
        "1 qubit per circle or (lat, lon) pair + ceil(d_plain/2) (ADR 0005); 2Q gates <= n - 1"
    )
    parts, statuses = [], []
    for ds, r in results.items():
        if r.cost is None:
            continue
        q = next((m for m in r.cost.models if m.model == "qlassifier"), None)
        if q is None:
            continue
        d = r.protocol.n_features_used
        two_q = max(q.two_qubit_gates.values())
        gates_ok = two_q <= q.qubits - 1
        core_limit = math.ceil(d / 2)
        geometry = arrays(ds).geometry
        if geometry is None:
            statuses.append("met" if q.qubits <= core_limit and gates_ok else "missed")
            parts.append(f"{ds}: {q.qubits} qubits (core, ceil({d}/2) = {core_limit}), {two_q} two-qubit gates")
            continue
        adr_limit = geometry_qubit_limit(Encoding.plan(d, cast(Geometry, geometry), None))
        statuses.append("met" if q.qubits <= adr_limit and gates_ok else "missed")
        parts.append(f"{ds}: {q.qubits} qubits (geometry, ADR 0005 limit {adr_limit}), {two_q} two-qubit gates")
    if not parts:
        return Target("BT-8", "—", rule, "not-measured")
    return Target("BT-8", "; ".join(parts), rule, "missed" if "missed" in statuses else "met")


def _bt9(results: Mapping[str, BenchmarkResult]) -> Target:
    rule = "Breast Cancer, 1,024 shots: mean accuracy drop <= 2 pp on each device (part A); < FQK-bw drop (part B)"
    r = results.get("breast_cancer")
    if r is None or r.noise is None:
        return Target("BT-9", "—", rule, "not-measured")
    s = {(x.part, x.model, x.device): x for x in r.noise.summary}
    a = [s.get(("A", "qlassifier", dv)) for dv in DEVICES]
    b_q = [s.get(("B", "qlassifier", dv)) for dv in DEVICES]
    b_f = [s.get(("B", "fqk_bandwidth_linear", dv)) for dv in DEVICES]
    if any(x is None for x in a + b_q + b_f):
        return Target("BT-9", "—", rule, "not-measured")
    da = [x.drop_mean_pp for x in a if x is not None]
    dq = [x.drop_mean_pp for x in b_q if x is not None]
    df = [x.drop_mean_pp for x in b_f if x is not None]
    text = (
        f"part A drop {' / '.join(f'{v:.2f}' for v in da)} pp (Torino / Fez / Brisbane); part B Qlassifier "
        f"{' / '.join(f'{v:.1f}' for v in dq)} vs FQK-bw linear {' / '.join(f'{v:.1f}' for v in df)} pp"
    )
    ok = all(v <= 2.0 for v in da) and all(q < f for q, f in zip(dq, df, strict=True))
    return Target("BT-9", text, rule, "met" if ok else "missed")


def _bt10(bt10: Mapping[str, Any] | None) -> Target:
    rule = "`qlassifier benchmark breast_cancer`, defaults, exact, Apple M-class laptop: wall <= 120 s"
    if bt10 is None:
        return Target("BT-10", "—", rule, "not-measured")
    wall = float(bt10["wall_s"])
    text = f"{wall:.1f} s on {bt10.get('cpu', '?')} ({bt10.get('cores', '?')} cores, {bt10.get('platform', '?')}"
    load = bt10.get("load_avg_at_start")
    text += f"; 1-min load {load[0]:.2f} at start)" if load else ")"
    return Target("BT-10", text, rule, "met" if wall <= BT10_LIMIT_S else "missed")


def evaluate(results: Mapping[str, BenchmarkResult], bt10: Mapping[str, Any] | None = None) -> list[Target]:
    """Every target from the results of one benchmark output folder."""
    bt3, bt4 = _bt3_bt4(results)
    return [
        _bt1(results), _bt2(results), bt3, bt4, _bt5(results), _bt6(results), _bt7(results), _bt8(results),
        _bt9(results), _bt10(bt10),
    ]  # fmt: skip


def _cpu() -> str:
    if platform.system() == "Darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _summary_md(results: Mapping[str, BenchmarkResult]) -> str:
    lines = ["# Benchmark summary", "", "Generated by `qlassifier benchmark` (TDD §7.12). BA = balanced accuracy.", ""]
    for ds, r in results.items():
        lines += [f"## {r.dataset.name} (`{ds}`)", ""]
        lines += [
            f"{len(r.protocol.seeds)} seeds · {r.protocol.recipe} · split {r.protocol.split} · "
            f"{r.protocol.cv_folds}-fold CV · corrected 95% CIs (n_te/n_tr {r.protocol.test_train_ratio:.3f}) · "
            f"status {r.status}",
            "",
            "| Model | BA mean ± sd | 95% CI | Qlassifier − model (pp) [CI] | Qlassifier wins | Acc | ROC-AUC | F1 "
            "| Grid edge |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        comps = {c.model: c.metrics.balanced_accuracy for c in r.comparisons}
        for m in sorted(r.models, key=lambda m: -m.metrics.balanced_accuracy.mean):
            s = m.metrics.balanced_accuracy
            sd = f"{s.std:.3f}" if s.std is not None else "—"
            ci = (
                f"[{clip_unit(s.ci_low):.3f}, {clip_unit(s.ci_high):.3f}]"
                if s.ci_low is not None and s.ci_high is not None
                else "—"
            )
            d = comps.get(m.id)
            if d is None:
                delta, wins = "—", "—"
            else:
                lo = f"{100 * d.ci_low:+.1f}" if d.ci_low is not None else "?"
                hi = f"{100 * d.ci_high:+.1f}" if d.ci_high is not None else "?"
                delta, wins = f"{100 * d.mean_diff:+.1f} [{lo}, {hi}]", f"{round(d.win_rate * d.n)}/{d.n}"
            name = f"**{m.name}**" if m.id == "qlassifier" else m.name
            tie = " ≈" if s.tied_with_best else ""
            edge = f"{m.edge_share:.1f}" if r.protocol.configs_per_model.get(m.id, 0) > 1 else "—"
            lines.append(
                f"| {name}{tie} | {s.mean:.3f} ± {sd} | {ci} | {delta} | {wins} | {m.metrics.accuracy.mean:.3f} | "
                f"{m.metrics.roc_auc.mean:.3f} | {m.metrics.f1.mean:.3f} | {edge} |"
            )
        ties = (
            "≈ statistically tied with the best (corrected CI of the paired difference includes 0)."
            if len(r.protocol.seeds) >= MIN_TIE_SEEDS
            else f"Ties need ≥ {MIN_TIE_SEEDS} seeds (none marked at {len(r.protocol.seeds)})."
        )
        lines += [
            "",
            f"{ties} CI ends are clipped to [0, 1] (raw values in `result.json`). Grid edge: share of seeds whose "
            "chosen configuration sits on an edge of the model's grid (a wider grid could move it).",
            "",
        ]
        h = r.highlights
        if h.published is not None:
            delta = f"Qlassifier {h.vs_published_pp:+.1f} pp" if h.vs_published_pp is not None else "Qlassifier not run"
            lines.append(
                f"- Published best quantum: {h.published.value:.3f} BA ({h.published.method}, {h.published.source}); "
                f"{delta} (cross-protocol context)."
            )
        if r.noise is not None:
            lines += [
                "",
                "| Noise part | Model | Device | Acc | Drop vs exact (pp, mean / worst) | Kernel r |",
                "|---|---|---|---|---|---|",
            ]
            for x in r.noise.summary:
                kr = f"{x.kernel_r_mean:.4f}" if x.kernel_r_mean is not None else "—"
                lines.append(
                    f"| {x.part} | {x.model} | {x.device} | {x.accuracy_mean:.3f} | {x.drop_mean_pp:.2f} / "
                    f"{x.drop_worst_pp:.2f} | {kr} |"
                )
        if r.cost is not None:
            lines += [
                "",
                f"| Cost (N_tr {r.cost.n_train}, N_te {r.cost.n_test}) | Qubits | 2Q gates Torino / Fez / Brisbane | "
                "Depth (Torino) | Circuits train + predict | On-device tuning circuits |",
                "|---|---|---|---|---|---|",
            ]
            for c in r.cost.models:
                g = " / ".join(str(c.two_qubit_gates[dv]) for dv in DEVICES)
                lines.append(
                    f"| {c.model} | {c.qubits} | {g} | {c.depth['fake_torino']} | {c.circuits_fit_predict:,} | "
                    f"{c.tuning_circuits_on_device:,} |"
                )
        lines.append("")
    return "\n".join(lines)


Provenance = dict[str, dict[str, Any]]


def saved_results(
    out: Path, exclude: Iterable[str] = ()
) -> tuple[dict[str, BenchmarkResult], dict[str, float], Provenance]:
    """Dataset results already saved in an output folder by earlier runs, with their wall times and provenance.

    ``qlassifier benchmark`` reports every ``<out>/<dataset>/result.json``, so the datasets can be run one at a time
    (or one of them rerun) and the report still covers all of them. Returns ``({dataset: result}, {dataset: wall_s},
    {dataset: {"wall_s", "n_jobs", "command"}})`` for the datasets not in ``exclude``, in protocol order.
    """
    from qlassifier.errors import DATASET_IDS

    results: dict[str, BenchmarkResult] = {}
    walls: dict[str, float] = {}
    provenance: Provenance = {}
    env_path = out / "environment.json"
    previous = json.loads(env_path.read_text()) if env_path.is_file() else {}
    for ds in DATASET_IDS:
        path = out / ds / "result.json"
        if ds in exclude or not path.is_file():
            continue
        result = BenchmarkResult.from_json(path)
        results[ds] = result
        entry = previous.get("per_dataset", {}).get(ds) or {
            "wall_s": previous.get("wall_s", {}).get(ds, result.environment.wall_s),
            "n_jobs": result.environment.n_jobs,
            "command": previous.get("command") if ds in previous.get("wall_s", {}) else None,
        }
        walls[ds] = float(entry["wall_s"])
        provenance[ds] = dict(entry)
    return results, walls, provenance


def _status_cell(t: Target) -> str:
    if t.status == "not-measured":
        return f"**not measured** in this run: needs {NEEDS[t.id]}"
    return f"**{t.status}**"


def _headline(results: Mapping[str, BenchmarkResult]) -> str:
    """One line a newcomer can read: Qlassifier's BA per dataset vs the standard quantum baseline and best classical."""
    parts = []
    for ds, r in results.items():
        q = _mean(r, "qlassifier")
        if q is None:
            parts.append(f"{ds}: Qlassifier not run")
            continue
        text = f"{ds}: Qlassifier BA {q:.3f}"
        z, b = r.highlights.vs_standard_qml, r.highlights.vs_best_classical
        if z is not None:
            text += f", {100 * z.mean_diff:+.1f} pp vs the standard quantum baseline (ZZ QSVC)"
        if b is not None and r.highlights.best_classical is not None:
            verdict = (
                "a statistical tie"
                if b.ci_low is None or b.ci_high is None or b.ci_low <= 0 <= b.ci_high
                else ("a significant win" if b.ci_low > 0 else "a significant loss")
            )
            text += (
                f", {100 * b.mean_diff:+.1f} pp vs the best classical model ({r.highlights.best_classical}: {verdict})"
            )
        parts.append(text)
    return "; ".join(parts) + "."


def _rerun(per_dataset: Mapping[str, Mapping[str, Any]], command: str) -> str:
    """The command(s) that produced the folder: one, or each distinct one with its datasets (two-run provenance)."""
    by_command: dict[str, list[str]] = {}
    for ds, entry in per_dataset.items():
        by_command.setdefault(str(entry.get("command") or command), []).append(ds)
    if len(by_command) <= 1:
        return f"`{command}`"
    return " and ".join(f"`{cmd}` ({', '.join(ds)})" for cmd, ds in by_command.items())


def write_report(
    out: Path,
    results: Mapping[str, BenchmarkResult],
    command: str,
    walls: Mapping[str, float],
    n_jobs: int,
    bt10: Mapping[str, Any] | None,
    load_avg: Sequence[float] | None = None,
    saved: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write ``REPORT.md``, ``targets.json``, ``summary.md`` and ``environment.json``; return the report path.

    ``load_avg``: the machine's 1/5/15-min load average when the run started (timing context, e.g. BT-10).
    ``saved``: provenance of datasets taken from earlier runs in the folder (``saved_results``): their wall time,
    worker count and command are kept in ``environment.json["per_dataset"]`` next to this run's.
    """
    import os

    first = next(iter(results.values()))
    saved = saved or {}
    per_dataset = {
        ds: dict(saved[ds]) if ds in saved else {"wall_s": round(walls[ds], 1), "n_jobs": n_jobs, "command": command}
        for ds in results
    }
    env: dict[str, Any] = {
        "command": command,
        "versions": first.versions.model_dump(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu(),
        "cores": os.cpu_count(),
        "n_jobs": n_jobs,
        "seeds": first.protocol.seeds,
        "datasets": list(results),
        "noise": any(r.noise is not None for r in results.values()),
        "wall_s": {ds: round(w, 1) for ds, w in walls.items()},
        "per_dataset": per_dataset,
    }
    if load_avg is not None:
        env["load_avg_at_start"] = [round(float(v), 2) for v in load_avg]
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    targets = evaluate(results, bt10)
    (out / "targets.json").write_text(json.dumps([t.to_dict() for t in targets], indent=2, ensure_ascii=False) + "\n")
    (out / "summary.md").write_text(_summary_md(results))
    lines = [
        "# Benchmark report — targets BT-1…BT-10",
        "",
        "## Read this first",
        f"- {_headline(results)}",
        "- Leaderboards with confidence intervals, noise and cost, per dataset: [summary.md](summary.md). Per-seed "
        "rows: `<dataset>/runs.csv`; plots: `<dataset>/plots/`.",
        "- Below: the stage's ten benchmark targets (BT-1…BT-10), each with its measured value, pass rule and "
        "verdict. A target this run could not measure says what it needs.",
        "",
        f"Rerun: {_rerun(per_dataset, command)} · Protocol: TDD §7 · Rules: PRD §6 / TDD §7.10 · Details: "
        "[summary.md](summary.md), [environment.json](environment.json), per-dataset `result.json`, `runs.csv`, "
        "`plots/`.",
        "",
        "| BT | Measured | Rule | Status |",
        "|---|---|---|---|",
    ]
    lines += [f"| {t.id} | {t.measured} | {t.rule} | {_status_cell(t)} |" for t in targets]
    lines += [
        "",
        f"- Datasets: {', '.join(results)}; seeds {first.protocol.seeds}; versions qlassifier "
        f"{first.versions.qlassifier}, qiskit {first.versions.qiskit}, scikit-learn {first.versions.scikit_learn}.",
        f"- Machine: {env['cpu']}, {env['cores']} cores; wall per dataset (workers): "
        + ", ".join(f"{ds} {per_dataset[ds]['wall_s']:.0f} s ({per_dataset[ds]['n_jobs']})" for ds in results)
        + ".",
        "- BT-6 is report-only in stage 1 (ADR 0004): the phase-7 pre-registered tests failed; nothing is upgraded "
        "here whatever these seeds show.",
        "- BT-8 judges the core datasets by the PRD's ceil(d/2) qubits and the geometry-mode dataset (protein_angles) "
        "by the ADR 0005 rule: one qubit per circle or direction (approved 2026-10-25). Two-qubit gates: <= n - 1 "
        "everywhere.",
        "",
    ]
    report = out / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


def bt10_from(
    out: Path,
    results: Mapping[str, BenchmarkResult],
    walls: Mapping[str, float],
    defaults: bool,
    load_avg: Sequence[float] | None = None,
) -> dict[str, Any] | None:
    """BT-10 input: this run when it is a default exact Breast Cancer run, else ``<out>/bt10/environment.json``."""
    import os

    if defaults and list(results) == ["breast_cancer"]:
        return {
            "wall_s": walls["breast_cancer"],
            "cpu": _cpu(),
            "cores": os.cpu_count(),
            "platform": platform.platform(),
            "load_avg_at_start": list(load_avg) if load_avg is not None else None,
        }
    env = out / "bt10" / "environment.json"
    if env.is_file():
        data = json.loads(env.read_text())
        if "breast_cancer" in data.get("wall_s", {}):
            return {
                "wall_s": data["wall_s"]["breast_cancer"],
                "cpu": data.get("cpu"),
                "cores": data.get("cores"),
                "platform": data.get("platform"),
                "load_avg_at_start": data.get("load_avg_at_start"),
            }
    return None
