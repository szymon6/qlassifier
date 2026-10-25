"""Typed errors and warnings. Every message says what happened; every hint says how to fix it.

Design: TDD §5.6 (templates) and §6.5 (HTTP mapping). ``str(error) == f"{message} {hint}"``; the local API
returns ``{"error": {"code", "message", "hint", "status"}}`` with the same text, so the dashboard toast and the
Python traceback read the same.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

__all__ = [
    "BACKENDS",
    "DATASET_IDS",
    "DatasetIntegrityError",
    "GeometryError",
    "InvalidParameterError",
    "MissingValueError",
    "MultiClassError",
    "NoResultError",
    "NonNumericError",
    "QlassifierError",
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
    "error_body",
]

DATASET_IDS: tuple[str, ...] = ("breast_cancer", "ionosphere", "sonar", "heart", "protein_angles")
BACKENDS: tuple[str, ...] = ("exact", "shots", "fake_torino", "fake_fez", "fake_brisbane")
MIN_ROWS = 20
MIN_PER_CLASS = 5
MAX_TRAIN_ROWS = 10_000


def _plain(value: Any) -> Any:
    """Python scalar for numpy scalars, so messages read ``1`` and not ``np.int64(1)``."""
    return value.item() if hasattr(value, "item") and getattr(value, "shape", None) == () else value


def _join(items: Iterable[Any], limit: int = 3) -> str:
    """Join the first ``limit`` items with commas; add an ellipsis when more exist."""
    values = [str(_plain(v)) for v in items]
    shown = ", ".join(values[:limit])
    return shown + (", …" if len(values) > limit else "")


class QlassifierError(Exception):
    """Base class of every Qlassifier error.

    Attributes:
        code: stable machine-readable code (the same in the HTTP API).
        status: HTTP status the local API uses for this error.
        message: what happened.
        hint: how to fix it.
    """

    code: ClassVar[str] = "internal"
    status: ClassVar[int] = 500

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(f"{message} {hint}")
        self.message = message
        self.hint = hint

    def __reduce__(self) -> tuple[Any, ...]:
        # Keep message and hint apart when the error crosses a process boundary (joblib workers).
        return (_rebuild, (type(self), self.message, self.hint))

    def to_dict(self) -> dict[str, Any]:
        """Return the API error body ``{code, message, hint, status}``."""
        return {"code": self.code, "message": self.message, "hint": self.hint, "status": self.status}


def _rebuild(cls: type[QlassifierError], message: str, hint: str) -> QlassifierError:
    err = QlassifierError.__new__(cls)
    QlassifierError.__init__(err, message, hint)
    return err


class _InputError(QlassifierError, ValueError):
    """Bad input data or arguments (also a ``ValueError``, as in scikit-learn)."""

    status: ClassVar[int] = 422


class MissingValueError(_InputError):
    """X or y contains NaN or infinite values."""

    code = "missing_values"

    @classmethod
    def in_x(cls, n: int, columns: Sequence[Any], method: str = "fit") -> MissingValueError:
        """Build the error for ``n`` bad values spread over ``columns`` of X."""
        k = len(columns)
        return cls(
            f"X has {n} missing or infinite value{'s' if n != 1 else ''} in {k} column{'s' if k != 1 else ''} "
            f"({_join(columns)}).",
            f"Drop or impute those rows, then call {method} again.",
        )

    @classmethod
    def in_y(cls, n: int) -> MissingValueError:
        """Build the error for ``n`` missing labels."""
        return cls(
            f"y has {n} missing value{'s' if n != 1 else ''}.", "Drop or impute those rows, then call fit again."
        )


class NonNumericError(_InputError):
    """A feature column is not numeric."""

    code = "non_numeric"

    def __init__(self, column: Any, value: Any) -> None:
        super().__init__(
            f"Column '{_plain(column)}' is not numeric (e.g. '{_plain(value)}').",
            "Encode it as numbers (e.g. one-hot) or drop it.",
        )


class SingleClassError(_InputError):
    """y contains only one class."""

    code = "single_class"

    def __init__(self, label: Any) -> None:
        super().__init__(f"y has only one class ({_plain(label)!r}).", "Qlassifier needs two classes.")


class MultiClassError(_InputError):
    """y contains more than two classes (stage 1 is binary)."""

    code = "multi_class"

    def __init__(self, labels: Sequence[Any]) -> None:
        super().__init__(
            f"y has {len(labels)} classes ({_join(labels, 5)}).",
            "Only binary classification is supported. Keep two classes or merge them.",
        )


class TooFewSamplesError(_InputError):
    """Fewer than 20 rows, or fewer than 5 rows in a class."""

    code = "too_few_samples"

    def __init__(self, n: int, counts: dict[Any, int]) -> None:
        shown = ", ".join(f"{_plain(label)!r}: {count}" for label, count in counts.items())
        super().__init__(
            f"Got {n} rows ({shown}).",
            f"Qlassifier needs at least {MIN_ROWS} rows and {MIN_PER_CLASS} per class to tune itself. Add rows.",
        )


class TooManySamplesError(_InputError):
    """More training rows than the N × N kernel allows."""

    code = "too_many_samples"

    def __init__(self, n: int) -> None:
        gb = n * n * 8 / 1e9
        super().__init__(
            f"Got {n} training rows; the kernel would need {gb:.1f} GB.",
            f"Use at most {MAX_TRAIN_ROWS:,} rows (e.g. a stratified subsample).",
        )


class GeometryError(_InputError):
    """The ``geometry`` declaration is invalid (TDD §5.6 lists every case)."""

    code = "bad_geometry"


class InvalidParameterError(_InputError):
    """An argument has an invalid value (e.g. ``shots=0``)."""

    code = "invalid_parameter"


class UnknownBackendError(_InputError):
    """``backend`` names no supported execution mode."""

    code = "unknown_backend"

    def __init__(self, backend: Any) -> None:
        super().__init__(
            f"Unknown backend '{backend}'.",
            "Use 'exact', 'shots', 'fake_torino', 'fake_fez' or 'fake_brisbane'.",
        )


class TooManyQubitsError(_InputError):
    """The circuit needs more qubits than the chosen device has."""

    code = "too_many_qubits"

    def __init__(self, n: int, backend: str, m: int) -> None:
        super().__init__(
            f"This model needs {n} qubits; {backend} has {m}.",
            "Reduce features (e.g. PCA) or use backend='exact'.",
        )


class UnknownDatasetError(_InputError):
    """No bundled dataset with this id."""

    code = "unknown_dataset"
    status = 404

    def __init__(self, name: Any) -> None:
        super().__init__(f"No dataset '{name}'.", f"Choose one of: {', '.join(DATASET_IDS)}.")


class UnknownModelError(_InputError):
    """``benchmark(models=…)`` names a model the protocol does not have."""

    code = "unknown_model"

    def __init__(self, model: Any, ids: Sequence[str]) -> None:
        super().__init__(f"No model '{model}'.", f"Choose from: {', '.join(ids)}.")


class DatasetIntegrityError(QlassifierError):
    """A bundled data file does not match its recorded SHA-256 hash."""

    code = "dataset_integrity"
    status = 500

    def __init__(self, file: str) -> None:
        super().__init__(
            f"Bundled file '{file}' failed its SHA-256 check.", "Reinstall: pip install --force-reinstall ."
        )


class RunInProgressError(QlassifierError):
    """The local server already runs a benchmark (one run at a time)."""

    code = "run_in_progress"
    status = 409

    def __init__(self, dataset: str, pct: int) -> None:
        super().__init__(f"A run is already in progress ({dataset}, {pct}%).", "Wait for it or stop it first.")


class RunNotFoundError(QlassifierError):
    """The local server has no run with this id."""

    code = "run_not_found"
    status = 404

    def __init__(self, run_id: str) -> None:
        super().__init__(f"No run '{run_id}' in this session.", "Start a new run.")


class NoResultError(QlassifierError):
    """No saved benchmark result exists for a dataset (stale or partial install)."""

    code = "no_result"
    status = 404

    def __init__(self, dataset: str) -> None:
        super().__init__(f"No saved result for '{dataset}'.", f"Run it once: qlassifier benchmark {dataset}.")


class ShotNoiseWarning(UserWarning):
    """Shot noise is expected to cost noticeable accuracy at the chosen ``shots``."""

    def __init__(self, shots: int, pp: float) -> None:
        self.shots = shots
        self.pp = pp
        self.message = (
            f"Shot noise at shots={shots} is expected to cost {pp:.1f} pp balanced accuracy "
            "(simulated on the training folds, 0 circuits)."
        )
        self.hint = "This data needs a narrow kernel: raise shots (4× shots halve the noise) or use backend='exact'."
        super().__init__(f"{self.message} {self.hint}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.shots, self.pp))


def error_body(exc: BaseException) -> dict[str, Any]:
    """Map any exception to the API error body ``{code, message, hint, status}`` (TDD §6.5).

    Typed errors keep their code and text. Anything else becomes ``internal`` without a stack trace.
    """
    if isinstance(exc, QlassifierError):
        return exc.to_dict()
    return {
        "code": "internal",
        "message": f"Unexpected error ({type(exc).__name__}).",
        "hint": "See the terminal running `qlassifier ui`.",
        "status": 500,
    }
