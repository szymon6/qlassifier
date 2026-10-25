"""Qlassifier: the quantum classifier built to win, with its proof inside.

Public API (TDD §5): ``Qlassifier``, ``Resources``, ``load_dataset``, ``list_datasets``, ``Dataset``,
``DatasetInfo``, ``benchmark``, ``BenchmarkResult``, ``ProgressEvent`` and the typed errors.
Heavy dependencies (Qiskit, matplotlib, FastAPI) load only when a feature needs them.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from qlassifier.datasets import Dataset, DatasetInfo, list_datasets, load_dataset
from qlassifier.encoding import QubitMap
from qlassifier.errors import (
    DatasetIntegrityError,
    GeometryError,
    InvalidParameterError,
    MissingValueError,
    MultiClassError,
    NonNumericError,
    NoResultError,
    QlassifierError,
    RunInProgressError,
    RunNotFoundError,
    ShotNoiseWarning,
    SingleClassError,
    TooFewSamplesError,
    TooManyQubitsError,
    TooManySamplesError,
    UnknownBackendError,
    UnknownDatasetError,
    UnknownModelError,
)
from qlassifier.model import Qlassifier, Resources

if TYPE_CHECKING:
    from qlassifier.bench import ProgressEvent, benchmark
    from qlassifier.bench.result import BenchmarkResult

__version__ = version("qlassifier")

__all__ = [
    "BenchmarkResult",
    "Dataset",
    "DatasetInfo",
    "DatasetIntegrityError",
    "GeometryError",
    "InvalidParameterError",
    "MissingValueError",
    "MultiClassError",
    "NoResultError",
    "NonNumericError",
    "ProgressEvent",
    "Qlassifier",
    "QlassifierError",
    "QubitMap",
    "Resources",
    "RunInProgressError",
    "RunNotFoundError",
    "ShotNoiseWarning",
    "SingleClassError",
    "TooFewSamplesError",
    "TooManyQubitsError",
    "TooManySamplesError",
    "UnknownBackendError",
    "UnknownDatasetError",
    "UnknownModelError",
    "__version__",
    "benchmark",
    "list_datasets",
    "load_dataset",
]


def __getattr__(name: str) -> Any:
    """Load the benchmark API lazily (it pulls in joblib and pydantic models, not needed for ``fit``)."""
    if name in ("benchmark", "ProgressEvent"):
        from qlassifier import bench

        return getattr(bench, name)
    if name == "BenchmarkResult":
        from qlassifier.bench.result import BenchmarkResult

        return BenchmarkResult
    raise AttributeError(f"module 'qlassifier' has no attribute {name!r}")
