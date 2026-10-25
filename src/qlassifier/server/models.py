"""API contract models of the local server (TDD §6.3-6.4). Generated JSON Schemas: ``src/qlassifier/schemas``.

Every 4xx / 5xx response and every ``failed`` event carries :class:`ErrorResponse` / :class:`ErrorBody`.
``BenchmarkResult`` lives in :mod:`qlassifier.bench.result`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qlassifier.bench.result import BenchmarkResult, Published

__all__ = [
    "API_VERSION",
    "DatasetCard",
    "DatasetList",
    "ErrorBody",
    "ErrorResponse",
    "FailedData",
    "Glyph",
    "Meta",
    "MetaVersions",
    "PartialData",
    "ProgressData",
    "Run",
    "RunRequest",
    "TerminalData",
]

API_VERSION: Literal["1"] = "1"
RunStatus = Literal["running", "cancelling", "complete", "cancelled", "failed"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorBody(_Frozen):
    """A typed error: ``message`` + ``hint`` are exactly the library exception's text."""

    code: str
    message: str
    hint: str
    status: int


class ErrorResponse(_Frozen):
    """Body of every 4xx / 5xx response."""

    error: ErrorBody


class MetaVersions(_Frozen):
    """Versions shown by the dashboard."""

    python: str
    qiskit: str
    qiskit_aer: str
    qiskit_ibm_runtime: str
    scikit_learn: str
    numpy: str


class Run(_Frozen):
    """A dashboard benchmark run (one at a time)."""

    id: str
    dataset: str
    seeds: list[int]
    status: RunStatus
    done: int
    total: int = Field(description="seeds × models + 1 cost task.")
    eta_s: float | None
    started_at: str = Field(description="ISO 8601 UTC.")
    finished_at: str | None
    error: ErrorBody | None


class Meta(_Frozen):
    """``GET /api/v1/meta``: versions, defaults and the active run (for reattach after reload)."""

    api_version: Literal["1"]
    schema_version: str
    qlassifier: str
    versions: MetaVersions
    default_dataset: str
    run_seeds: int
    active_run: Run | None


class Glyph(_Frozen):
    """Picture-tile glyph: 120 rows as ``[x, y, class]`` in [0, 1]²."""

    kind: Literal["pca", "torus"]
    points: list[list[float]]


class DatasetCard(_Frozen):
    """One dataset tile."""

    id: str
    name: str
    n_samples: int
    n_features: int
    class_counts: dict[str, int]
    positive_class: str
    licence: str
    source: str
    citation: str
    recipe: Literal["pca8", "native"]
    grouped: bool
    geometry: dict[str, float] | None
    glyph: Glyph
    published: Published | None


class DatasetList(_Frozen):
    """``GET /api/v1/datasets``."""

    datasets: list[DatasetCard]


class RunRequest(BaseModel):
    """``POST /api/v1/runs`` body."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    seeds: int | None = Field(default=None, ge=2, le=20, description="Defaults to `qlassifier ui --seeds`.")


class ProgressData(_Frozen):
    """SSE ``progress`` event data: sent after each finished task."""

    run_id: str
    done: int
    total: int
    seed: int | None = Field(description="Seed of the finished task (null for the cost task).")
    model: str
    eta_s: float | None


class PartialData(_Frozen):
    """SSE ``partial`` event data: the partial result (all statistics computed server-side)."""

    run_id: str
    result: BenchmarkResult


class TerminalData(_Frozen):
    """SSE ``complete`` / ``cancelled`` event data."""

    run_id: str
    result_url: str


class FailedData(_Frozen):
    """SSE ``failed`` event data."""

    run_id: str
    error: ErrorBody
