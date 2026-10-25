"""pandas in, pandas-friendly out: DataFrames and Series through fit, predict, geometry, datasets and the benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlassifier import MissingValueError, Qlassifier, load_dataset


@pytest.fixture(scope="module")
def heart() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return load_dataset("heart").split(seed=0)


def test_dataframes_and_series_go_straight_into_fit_and_predict(heart: tuple[pd.DataFrame, ...]) -> None:
    X_train, X_test, y_train, y_test = heart
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    clf = Qlassifier().fit(X_train, y_train)
    assert list(clf.feature_names_in_) == list(X_train.columns)
    assert clf.predict(X_test).shape == (len(X_test),)
    assert clf.predict_proba(X_test).shape == (len(X_test), 2)
    assert 0.75 < clf.score(X_test, y_test) <= 1.0
    with pytest.raises(ValueError, match="feature names should match"):  # a reordered frame is refused, not guessed
        clf.predict(X_test[list(X_test.columns)[::-1]])


def test_string_labels_and_nullable_dtypes(heart: tuple[pd.DataFrame, ...]) -> None:
    X_train, X_test, y_train, _ = heart
    clf = Qlassifier().fit(X_train, y_train.map({0: "no", 1: "disease"}))
    assert list(clf.classes_) == ["disease", "no"]
    assert set(clf.predict(X_test)) <= {"disease", "no"}
    nullable = X_train.astype({"age": "Int64", "sex": "boolean"})
    assert Qlassifier().fit(nullable, y_train).score(X_test, y_train.iloc[: len(X_test)]) >= 0.0
    with pytest.raises(MissingValueError, match="age"):
        Qlassifier().fit(X_train.assign(age=np.nan), y_train)


def test_geometry_is_declared_by_column_name() -> None:
    data = load_dataset("protein_angles")
    X_train, X_test, y_train, y_test = data.split(seed=0)
    assert set(data.geometry) == set(X_train.columns)
    clf = Qlassifier(geometry=data.geometry).fit(X_train.iloc[:300], y_train.iloc[:300])
    assert clf.resources_.qubits == X_train.shape[1]  # one circle per angle column
    assert clf.score(X_test, y_test) > 0.8
    renamed = X_test.rename(columns=str.upper)
    with pytest.raises(ValueError, match="feature names"):
        clf.predict(renamed)
