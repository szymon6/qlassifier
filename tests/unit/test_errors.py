"""Every typed error: trigger -> class, code, HTTP status, exact text (TDD §5.6, PRD US-5)."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from qlassifier import (
    DatasetIntegrityError,
    InvalidParameterError,
    MissingValueError,
    MultiClassError,
    NonNumericError,
    NoResultError,
    Qlassifier,
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
    load_dataset,
)
from qlassifier.errors import error_body


def _heart() -> tuple[pd.DataFrame, pd.Series]:
    return load_dataset("heart", return_X_y=True)


def check(exc: QlassifierError, cls: type, code: str, status: int, message: str, hint: str) -> None:
    assert type(exc) is cls
    assert exc.code == code
    assert exc.status == status
    assert exc.message == message
    assert exc.hint == hint
    assert str(exc) == f"{message} {hint}"
    assert exc.to_dict() == {"code": code, "message": message, "hint": hint, "status": status}
    assert error_body(exc) == exc.to_dict()


def test_missing_values_in_x() -> None:
    X, y = _heart()
    X = X.copy()
    X.loc[3, "age"] = np.nan
    X.loc[5, "chol"] = np.inf
    with pytest.raises(MissingValueError) as info:
        Qlassifier().fit(X, y)
    check(
        info.value, MissingValueError, "missing_values", 422,
        "X has 2 missing or infinite values in 2 columns (age, chol).",
        "Drop or impute those rows, then call fit again.",
    )  # fmt: skip
    assert isinstance(info.value, ValueError)


def test_missing_values_at_predict_name_the_method() -> None:
    X, y = _heart()
    clf = Qlassifier().fit(X, y)
    bad = X.to_numpy()[:4].copy()
    bad[0, 0] = np.nan
    with pytest.raises(MissingValueError, match="then call predict again"):
        clf.predict(bad)


def test_missing_values_in_y() -> None:
    X, y = _heart()
    y = y.astype(float)
    y.iloc[0] = np.nan
    with pytest.raises(MissingValueError) as info:
        Qlassifier().fit(X, y)
    check(
        info.value, MissingValueError, "missing_values", 422, "y has 1 missing value.",
        "Drop or impute those rows, then call fit again.",
    )  # fmt: skip


def test_non_numeric_dataframe_column() -> None:
    X, y = _heart()
    X = X.assign(city=["Paris"] * len(X))
    with pytest.raises(NonNumericError) as info:
        Qlassifier().fit(X, y)
    check(
        info.value, NonNumericError, "non_numeric", 422, "Column 'city' is not numeric (e.g. 'Paris').",
        "Encode it as numbers (e.g. one-hot) or drop it.",
    )  # fmt: skip


def test_non_numeric_array_column() -> None:
    X, y = _heart()
    arr = X.to_numpy().astype(object)
    arr[7, 2] = "high"
    with pytest.raises(NonNumericError) as info:
        Qlassifier().fit(arr, y)
    assert info.value.message == "Column '2' is not numeric (e.g. 'high')."


def test_single_class() -> None:
    X, _y = _heart()
    with pytest.raises(SingleClassError) as info:
        Qlassifier().fit(X, np.ones(len(X), dtype=int))
    check(
        info.value, SingleClassError, "single_class", 422, "y has only one class (1).", "Qlassifier needs two classes."
    )


def test_multi_class() -> None:
    X, _ = _heart()
    y = np.arange(len(X)) % 3
    with pytest.raises(MultiClassError) as info:
        Qlassifier().fit(X, y)
    check(
        info.value, MultiClassError, "multi_class", 422, "y has 3 classes (0, 1, 2).",
        "Only binary classification is supported. Keep two classes or merge them.",
    )  # fmt: skip


@pytest.mark.parametrize(("n", "positives", "counts"), [(12, 6, "'no': 6, 'yes': 6"), (30, 4, "'no': 26, 'yes': 4")])
def test_too_few_samples(n: int, positives: int, counts: str) -> None:
    X, _ = _heart()
    y = np.array(["yes"] * positives + ["no"] * (n - positives))
    with pytest.raises(TooFewSamplesError) as info:
        Qlassifier().fit(X.iloc[:n], y)
    check(
        info.value, TooFewSamplesError, "too_few_samples", 422, f"Got {n} rows ({counts}).",
        "Qlassifier needs at least 20 rows and 5 per class to tune itself. Add rows.",
    )  # fmt: skip


def test_too_many_samples() -> None:
    rng = np.random.default_rng(0)  # mechanism test: the size guard runs before any computation
    X = rng.normal(size=(10_001, 2))
    y = np.arange(10_001) % 2
    with pytest.raises(TooManySamplesError) as info:
        Qlassifier().fit(X, y)
    check(
        info.value, TooManySamplesError, "too_many_samples", 422,
        "Got 10001 training rows; the kernel would need 0.8 GB.",
        "Use at most 10,000 rows (e.g. a stratified subsample).",
    )  # fmt: skip


def test_unknown_backend() -> None:
    X, y = _heart()
    with pytest.raises(UnknownBackendError) as info:
        Qlassifier(backend="ibm_quantum").fit(X, y)  # type: ignore[arg-type]
    check(
        info.value, UnknownBackendError, "unknown_backend", 422, "Unknown backend 'ibm_quantum'.",
        "Use 'exact', 'shots', 'fake_torino', 'fake_fez' or 'fake_brisbane'.",
    )  # fmt: skip


def test_too_many_qubits() -> None:
    X, y = _heart()
    wide = np.tile(X.to_numpy()[:40], (1, 21))  # 273 columns -> 137 qubits > 133 on FakeTorino
    with pytest.raises(TooManyQubitsError) as info:
        Qlassifier(backend="fake_torino").fit(wide, y.iloc[:40])
    check(
        info.value, TooManyQubitsError, "too_many_qubits", 422, "This model needs 137 qubits; fake_torino has 133.",
        "Reduce features (e.g. PCA) or use backend='exact'.",
    )  # fmt: skip


def test_unknown_dataset() -> None:
    with pytest.raises(UnknownDatasetError) as info:
        load_dataset("iris")
    check(
        info.value, UnknownDatasetError, "unknown_dataset", 404, "No dataset 'iris'.",
        "Choose one of: breast_cancer, ionosphere, sonar, heart, protein_angles.",
    )  # fmt: skip


def test_unknown_model() -> None:
    from qlassifier import benchmark

    with pytest.raises(UnknownModelError) as info:
        benchmark("heart", models=["svm"])
    assert info.value.code == "unknown_model"
    assert info.value.status == 422
    assert info.value.message == "No model 'svm'."
    assert info.value.hint.startswith("Choose from: trivial, logistic_regression, rbf_svm,")


def test_dataset_integrity(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import qlassifier.datasets as ds

    tampered = tmp_path / "data"
    tampered.mkdir()
    for f in ds.DATA_DIR.iterdir():
        (tampered / f.name).write_bytes(f.read_bytes())
    (tampered / "sonar.csv").write_bytes((ds.DATA_DIR / "sonar.csv").read_bytes() + b"\n")
    monkeypatch.setattr(ds, "DATA_DIR", tampered)
    ds._verified.cache_clear()
    ds._load.cache_clear()
    try:
        with pytest.raises(DatasetIntegrityError) as info:
            load_dataset("sonar")
        check(
            info.value, DatasetIntegrityError, "dataset_integrity", 500,
            "Bundled file 'sonar.csv' failed its SHA-256 check.", "Reinstall: pip install --force-reinstall .",
        )  # fmt: skip
        (tampered / "heart.csv").unlink()
        with pytest.raises(DatasetIntegrityError, match=r"heart\.csv"):
            load_dataset("heart")
    finally:
        ds._verified.cache_clear()
        ds._load.cache_clear()


def test_run_errors() -> None:
    check(
        RunInProgressError("sonar", 45), RunInProgressError, "run_in_progress", 409,
        "A run is already in progress (sonar, 45%).", "Wait for it or stop it first.",
    )  # fmt: skip
    check(
        RunNotFoundError("r-123abc"), RunNotFoundError, "run_not_found", 404, "No run 'r-123abc' in this session.",
        "Start a new run.",
    )  # fmt: skip
    check(
        NoResultError("heart"), NoResultError, "no_result", 404, "No saved result for 'heart'.",
        "Run it once: qlassifier benchmark heart.",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"shots": 0}, "shots must be a positive integer; got 0."),
        ({"shots": True}, "shots must be a positive integer; got True."),
        ({"gamma": -1.0}, "gamma must be 'auto' or a positive number; got -1.0."),
        ({"C": "big"}, "C must be 'auto' or a positive number; got 'big'."),
        ({"random_state": 1.5}, "random_state must be an integer or None; got 1.5."),
    ],
)
def test_invalid_parameters(params: dict[str, object], message: str) -> None:
    X, y = _heart()
    with pytest.raises(InvalidParameterError) as info:
        Qlassifier(**params).fit(X, y)  # type: ignore[arg-type]
    assert info.value.code == "invalid_parameter"
    assert info.value.message == message


def test_shot_noise_warning_text() -> None:
    w = ShotNoiseWarning(1024, 8.24)
    assert str(w) == (
        "Shot noise at shots=1024 is expected to cost 8.2 pp balanced accuracy (simulated on the training folds, "
        "0 circuits). This data needs a narrow kernel: raise shots (4× shots halve the noise) or use backend='exact'."
    )
    assert isinstance(w, UserWarning)
    assert str(pickle.loads(pickle.dumps(w))) == str(w)


def test_errors_survive_pickling() -> None:
    for exc in (UnknownDatasetError("iris"), TooFewSamplesError(3, {0: 2, 1: 1}), NonNumericError("c", "v")):
        back = pickle.loads(pickle.dumps(exc))
        assert type(back) is type(exc)
        assert str(back) == str(exc)
        assert back.to_dict() == exc.to_dict()


def test_error_body_for_unexpected_errors() -> None:
    assert error_body(KeyError("x")) == {
        "code": "internal",
        "message": "Unexpected error (KeyError).",
        "hint": "See the terminal running `qlassifier ui`.",
        "status": 500,
    }


def test_not_fitted_is_sklearns_error() -> None:
    from sklearn.exceptions import NotFittedError

    X, _ = _heart()
    with warnings.catch_warnings(), pytest.raises(NotFittedError):
        Qlassifier().predict(X)
