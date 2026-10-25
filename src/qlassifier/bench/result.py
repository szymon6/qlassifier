"""``BenchmarkResult``: the frozen, JSON-first result of a benchmark run (TDD §5.5, §7.12).

JSON is the source of truth (``schema_version`` "1.0"); the dashboard, the paper tables and the shipped cache all
read it. pydantic models here generate the API contract (``src/qlassifier/schemas/*.json``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qlassifier.datasets import DatasetInfo
from qlassifier.errors import DATASET_IDS, NoResultError, UnknownDatasetError

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "METRICS",
    "SCHEMA_VERSION",
    "BenchmarkResult",
    "Comparison",
    "CostModel",
    "CostStudy",
    "Environment",
    "Highlights",
    "MetricSummary",
    "ModelMetrics",
    "ModelSummary",
    "NoisePart",
    "NoiseRecord",
    "NoiseStudy",
    "NoiseSummary",
    "PairedDiff",
    "PairedMetrics",
    "Protocol",
    "Published",
    "RunRecord",
    "Times",
    "Versions",
]

SCHEMA_VERSION = "1.0"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "_results"
MetricName = Literal["balanced_accuracy", "accuracy", "roc_auc", "f1"]
METRICS: tuple[MetricName, ...] = ("balanced_accuracy", "accuracy", "roc_auc", "f1")
METRIC_LABELS = {"balanced_accuracy": "BA", "accuracy": "accuracy", "roc_auc": "ROC-AUC", "f1": "F1"}
ConfigValue = bool | int | float | str | list[int] | None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Versions(_Frozen):
    """Package versions that produced the result."""

    qlassifier: str
    python: str
    numpy: str
    scipy: str
    scikit_learn: str
    pandas: str
    joblib: str
    qiskit: str
    qiskit_aer: str
    qiskit_ibm_runtime: str


class Environment(_Frozen):
    """Machine and run settings."""

    platform: str
    machine: str
    cpu_count: int
    n_jobs: int
    wall_s: float = Field(description="Wall-clock seconds of the whole run.")


class Protocol(_Frozen):
    """The fair-comparison protocol the numbers were produced under (TDD §7)."""

    fingerprint: str = Field(description="SHA-256 of grids, models, seeds, recipe and split rule.")
    seeds: list[int]
    test_size: float
    split: Literal["stratified", "grouped", "stratified_grouped"]
    test_train_ratio: float = Field(description="Mean realised n_test / n_train (corrected CIs use it).")
    cv_folds: int
    recipe: Literal["pca8", "native"]
    n_features_used: int
    configs_per_model: dict[str, int]
    selection_metric: Literal["balanced_accuracy"]
    ci_level: float
    statistics: Literal["nadeau_bengio_corrected_t"]
    models: list[str]


class MetricSummary(_Frozen):
    """One metric of one model over the finished seeds."""

    mean: float
    std: float | None = Field(description="Sample sd over seeds (ddof 1); null below 2 seeds.")
    ci_low: float | None = Field(
        description="Corrected 95% CI of the mean (Nadeau-Bengio); null below 2 seeds. Raw value: with few seeds "
        "it can leave [0, 1]; displays clip it to the metric's range."
    )
    ci_high: float | None
    n: int
    tied_with_best: bool = Field(
        description="Corrected 95% CI of the paired difference vs the top-mean model includes 0 (top model: true). "
        "Evaluated only with >= 5 seeds; false below (TDD 7.7)."
    )


class ModelMetrics(_Frozen):
    """Every metric of one model."""

    balanced_accuracy: MetricSummary
    accuracy: MetricSummary
    roc_auc: MetricSummary
    f1: MetricSummary


class Times(_Frozen):
    """Median seconds per seed (in-worker)."""

    tune: float
    fit: float
    predict: float


class ModelSummary(_Frozen):
    """One leaderboard row."""

    id: str
    name: str
    family: Literal["quantum", "classical", "trivial", "control"]
    role: Literal["ours", "standard_qml", "published_quantum", "classical", "trivial", "twin"]
    metrics: ModelMetrics
    time_s: Times
    n_seeds_done: int
    rank: int = Field(description="Rank by mean balanced accuracy (1 = best).")
    edge_share: float = Field(description="Share of seeds whose chosen config sits on a grid edge.")


class PairedDiff(_Frozen):
    """Qlassifier minus another model, paired by seed (metric units, not pp)."""

    mean_diff: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    win_rate: float = Field(description="Share of seeds where Qlassifier is higher.")
    n: int


class PairedMetrics(_Frozen):
    """Paired differences for every metric."""

    balanced_accuracy: PairedDiff
    accuracy: PairedDiff
    roc_auc: PairedDiff
    f1: PairedDiff


class Comparison(_Frozen):
    """Qlassifier - ``model``."""

    model: str
    metrics: PairedMetrics


class Published(_Frozen):
    """Best published quantum result on this dataset (cross-protocol context)."""

    value: float
    metric: Literal["balanced_accuracy"]
    source: str
    method: str


class Highlights(_Frozen):
    """The numbers behind the dashboard's proof chips."""

    best_classical: str | None = Field(description="Classical model with the top mean balanced accuracy (ex post).")
    vs_standard_qml: PairedDiff | None = Field(description="Qlassifier - pinned ZZ QSVC, balanced accuracy.")
    vs_best_classical: PairedDiff | None = Field(description="Qlassifier - best classical, balanced accuracy.")
    published: Published | None
    vs_published_pp: float | None = Field(description="Qlassifier mean BA - published BA, in pp.")
    noise_drop_max_pp: float | None = Field(description="Max mean accuracy drop over the 3 devices (part A), pp.")
    circuit_ratio_vs_fqk_pipeline: float | None = Field(
        description="Circuits of the bandwidth-tuned fidelity-kernel pipeline / Qlassifier circuits."
    )


class RunRecord(_Frozen):
    """One (seed, model) result."""

    seed: int
    model: str
    config: dict[str, ConfigValue]
    config_index: int = Field(description="Position of the chosen config in the model's grid.")
    cv_score: float | None = Field(description="Mean CV balanced accuracy of the chosen config.")
    edge: bool
    balanced_accuracy: float
    accuracy: float
    roc_auc: float | None
    f1: float
    t_tune: float
    t_fit: float
    t_predict: float
    n_train: int
    n_test: int


class NoiseRecord(_Frozen):
    """One (part, model, device, seed) noisy run next to its exact twin."""

    part: Literal["A", "B"]
    model: str
    device: str
    seed: int
    accuracy: float
    balanced_accuracy: float
    exact_accuracy: float
    exact_balanced_accuracy: float
    drop_pp: float = Field(description="Accuracy lost vs exact, pp.")
    kernel_r: float | None = Field(description="Pearson r of noisy vs exact training kernel entries.")
    circuits: int
    two_qubit_gates: int
    depth: int
    n_train: int
    n_test: int
    seconds: float


class NoiseSummary(_Frozen):
    """Mean over seeds per (part, model, device); ``device == "exact"`` rows have drop 0."""

    part: Literal["A", "B"]
    model: str
    device: str
    accuracy_mean: float
    balanced_accuracy_mean: float
    drop_mean_pp: float
    drop_worst_pp: float
    kernel_r_mean: float | None
    n_seeds: int


class NoisePart(_Frozen):
    """Scope of one part of the noise study (TDD §7.8)."""

    id: Literal["A", "B"]
    models: list[str]
    seeds: list[int]
    devices: list[str]
    rows: Literal["all", "subset"]
    n_train: int | None
    n_test: int | None


class NoiseStudy(_Frozen):
    """Exact vs shot noise vs IBM device noise models."""

    shots: int
    devices: list[str]
    parts: list[NoisePart]
    records: list[NoiseRecord]
    summary: list[NoiseSummary]
    source: Literal["run", "shipped"] = Field(
        default="run",
        description=(
            'Provenance: "run" = measured by the run that produced this result; "shipped" = the shipped study '
            "(e.g. attached by GET /api/v1/results/{dataset} to a session run, which has no noise study)."
        ),
    )
    created_at: str | None = Field(default=None, description="ISO 8601 UTC time of the run that measured it.")


class CostModel(_Frozen):
    """Circuit resources of one quantum model."""

    model: str
    qubits: int
    two_qubit_gates: dict[str, int] = Field(description="Per device, median over 3 transpiler seeds.")
    depth: dict[str, int]
    circuits_fit_predict: int
    tuning_circuits_on_device: int


class CostStudy(_Frozen):
    """Transpiled resources and circuit counts (TDD §7.9)."""

    n_train: int
    n_test: int
    devices: list[str]
    models: list[CostModel]


class BenchmarkResult(_Frozen):
    """Leaderboard of Qlassifier vs every baseline on one dataset, with statistics, noise and cost.

    Build it with :func:`qlassifier.benchmark`, load it with :meth:`from_json` or :meth:`cached`.
    """

    schema_version: str = SCHEMA_VERSION
    status: Literal["partial", "complete", "cancelled"]
    source: Literal["run", "shipped"]
    created_at: str = Field(description="ISO 8601 UTC.")
    versions: Versions
    environment: Environment
    dataset: DatasetInfo
    protocol: Protocol
    models: list[ModelSummary]
    comparisons: list[Comparison]
    highlights: Highlights
    runs: list[RunRecord]
    noise: NoiseStudy | None
    cost: CostStudy | None

    # ------------------------------------------------------------------------- views
    def model(self, model_id: str) -> ModelSummary:
        """The leaderboard row of one model."""
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def summary(self, metric: MetricName = "balanced_accuracy") -> str:
        """Plain-text leaderboard: mean ± sd, 95% CI, Qlassifier − model [CI] in pp, Qlassifier's paired wins.

        CI ends are clipped to [0, 1] for display (the JSON keeps the raw values); ties are marked only with
        ≥ 5 seeds (TDD §7.7).
        """
        from qlassifier.bench.stats import MIN_TIE_SEEDS, clip_unit

        label = METRIC_LABELS[metric]
        n_seeds = len(self.protocol.seeds)
        head = (
            f"{self.dataset.name} · {len(self.models)} models × {n_seeds} seeds · {self.protocol.recipe} · "
            f"{label} (corrected 95% CI) · {self.status}"
        )
        comps = {c.model: getattr(c.metrics, metric) for c in self.comparisons}
        rows = sorted(self.models, key=lambda m: -getattr(m.metrics, metric).mean)
        lines = [
            head,
            f"{'model':<22} {'mean ± sd':>15} {'95% CI':>16} {'Qlassifier − model':>24} {'Qlassifier wins':>16}",
        ]
        for m in rows:
            s: MetricSummary = getattr(m.metrics, metric)
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
                delta = f"{100 * d.mean_diff:+.1f} pp [{lo}, {hi}]"
                wins = f"{round(d.win_rate * d.n)}/{d.n}"
            tie = " *" if s.tied_with_best else ""
            lines.append(f"{m.name + tie:<22} {f'{s.mean:.3f} ± {sd}':>15} {ci:>16} {delta:>24} {wins:>16}")
        if n_seeds < MIN_TIE_SEEDS:
            lines.append(f"ties need ≥ {MIN_TIE_SEEDS} seeds (none marked at {n_seeds})")
        else:
            lines.append("* statistically tied with the best (CI of the paired difference includes 0)")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per (seed, model) run."""
        import pandas as pd

        return pd.DataFrame([r.model_dump() for r in self.runs])

    def plot(
        self,
        kind: Literal["leaderboard", "noise", "cost"] = "leaderboard",
        metric: MetricName = "balanced_accuracy",
        ax: Axes | None = None,
    ) -> Figure:
        """Matplotlib figure: leaderboard (CIs + per-seed dots), noise (accuracy lost per device) or cost."""
        from qlassifier.bench.plots import plot

        return plot(self, kind=kind, metric=metric, ax=ax)

    # ------------------------------------------------------------------------- I/O
    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise to JSON (and write it when ``path`` is given)."""
        text = self.model_dump_json(indent=1)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, src: str | Path) -> BenchmarkResult:
        """Load from a JSON string or a file path."""
        if isinstance(src, Path) or (isinstance(src, str) and not src.lstrip().startswith("{")):
            src = Path(src).read_text(encoding="utf-8")
        return cls.model_validate_json(src)

    @classmethod
    def cached(cls, dataset: str) -> BenchmarkResult:
        """The shipped result for a bundled dataset (instant; used by notebooks and the dashboard)."""
        if dataset not in DATASET_IDS:
            raise UnknownDatasetError(dataset)
        path = RESULTS_DIR / f"{dataset}.json"
        if not path.is_file():
            raise NoResultError(dataset)
        return cls.from_json(path)

    def as_shipped(self) -> BenchmarkResult:
        """This result marked as the shipped cache (``source="shipped"``, its noise study too)."""
        noise = self.noise
        if noise is not None:
            noise = noise.model_copy(update={"source": "shipped", "created_at": noise.created_at or self.created_at})
        return self.model_copy(update={"source": "shipped", "noise": noise})

    def save(self, directory: str | Path) -> Path:
        """Write ``result.json``, ``runs.csv``, ``noise.csv`` (with a noise study) and ``plots/`` (TDD §7.12)."""
        out = Path(directory)
        (out / "plots").mkdir(parents=True, exist_ok=True)
        self.to_json(out / "result.json")
        frame = self.to_frame()
        if not frame.empty:
            frame = frame.assign(config=[json.dumps(c, sort_keys=True) for c in frame["config"]])
        frame.to_csv(out / "runs.csv", index=False)
        if self.noise is not None:
            import pandas as pd

            pd.DataFrame([r.model_dump() for r in self.noise.records]).to_csv(out / "noise.csv", index=False)
        from qlassifier.bench.plots import save_all

        save_all(self, out / "plots")
        return out

    def __repr__(self) -> str:
        seeds = len({r.seed for r in self.runs})
        text = f"BenchmarkResult({self.dataset.id}: {len(self.models)} models × {seeds} seeds, {self.status}"
        try:
            q = self.model("qlassifier").metrics.balanced_accuracy
        except KeyError:
            return text + ")"
        from qlassifier.bench.stats import clip_unit

        ci = (
            f" [{clip_unit(q.ci_low):.3f}, {clip_unit(q.ci_high):.3f}]"
            if q.ci_low is not None and q.ci_high is not None
            else ""
        )
        text += f", Qlassifier BA {q.mean:.3f}{ci}"
        if self.highlights.vs_standard_qml is not None:
            text += f", Δ vs ZZ {100 * self.highlights.vs_standard_qml.mean_diff:+.1f} pp"
        return text + ")"

    __str__ = __repr__


def dataset_info_dict(info: DatasetInfo) -> dict[str, Any]:
    """``DatasetInfo`` as a JSON-ready dict."""
    return asdict(info)
