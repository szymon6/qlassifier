"""Own CSV files as datasets: rules, errors, ids, provenance and splits (README "Your own CSV", TDD §5.4)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlassifier import load_dataset
from qlassifier.bench import _resolve_dataset
from qlassifier.datasets import MAX_CSV_ROWS, csv_dataset, is_csv_name, looks_like_csv
from qlassifier.errors import InvalidParameterError, UnknownDatasetError


@pytest.fixture
def heart_csv(tmp_path: Path) -> Path:
    X_train, _, y_train, _ = load_dataset("heart").split(seed=0)
    path = tmp_path / "heart.csv"
    pd.DataFrame(X_train).assign(target=y_train).to_csv(path, index=False)
    return path


def _write(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_csv_loads_like_a_bundled_dataset(heart_csv: Path) -> None:
    name = csv_dataset(heart_csv, target="target")
    assert is_csv_name(name)
    assert name.endswith("?target=target")
    data = load_dataset(name)
    assert data.info.id == "heart_csv"  # a bundled id is never shadowed
    assert (data.info.name, data.info.recipe, data.info.grouped) == ("heart.csv", "pca8", False)
    assert (data.info.n_samples, data.info.n_features) == (207, 13)
    assert data.info.class_counts == {0: 112, 1: 95}
    assert data.info.positive_class == "1"
    assert data.info.licence == "user data"
    assert data.info.source == str(heart_csv.resolve())
    assert data.info.sha256 == hashlib.sha256(heart_csv.read_bytes()).hexdigest()
    assert list(data.X.columns)[:3] == ["age", "sex", "cp"]
    assert data.geometry is None
    assert data.groups is None
    X_train, X_test, _y_train, y_test = data.split(seed=0)
    assert len(X_train) + len(X_test) == 207
    assert set(y_test.unique()) == {0, 1}


def test_path_default_target_and_return_x_y(heart_csv: Path) -> None:
    assert load_dataset(str(heart_csv)).info.n_samples == 207  # the last column is the target by default
    assert load_dataset(heart_csv, target="target").info.id == "heart_csv"
    X, y = load_dataset(heart_csv, return_X_y=True)
    assert X.shape == (207, 13)
    assert set(y.unique()) == {0, 1}
    assert looks_like_csv(heart_csv)
    assert looks_like_csv("x.CSV")
    assert not looks_like_csv("heart")
    with pytest.raises(UnknownDatasetError):
        load_dataset("heart.tsv")  # neither bundled nor a CSV file


def test_minority_value_is_the_positive_class(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    info = load_dataset(_write(tmp_path, "strings.csv", df.assign(label=["yes"] * 30 + ["no"] * 70))).info
    assert (info.positive_class, info.class_counts, info.id) == ("yes", {0: 70, 1: 30}, "strings")
    tie = load_dataset(_write(tmp_path, "Tie Data.csv", df.assign(label=["b"] * 50 + ["a"] * 50))).info
    assert (tie.positive_class, tie.id) == ("b", "tie_data")  # a tie takes the later value in text order


def test_large_files_are_sampled(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = MAX_CSV_ROWS + 100
    df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"]).assign(y=(rng.random(n) < 0.3).astype(int))
    with pytest.warns(UserWarning, match="stratified sample"):
        info = load_dataset(_write(tmp_path, "big.csv", df)).info
    assert info.n_samples == MAX_CSV_ROWS
    assert info.name == f"big.csv ({MAX_CSV_ROWS:,}-row stratified sample of {n:,})"
    assert abs(info.class_counts[1] / MAX_CSV_ROWS - df["y"].mean()) < 0.01


@pytest.mark.parametrize(
    ("name", "frame", "message"),
    [
        ("one.csv", pd.DataFrame({"y": [0, 1] * 30}), "needs feature columns"),
        ("allnan.csv", pd.DataFrame({"a": [None] * 60, "y": [0, 1] * 30}), "Every row"),
        ("alltext.csv", pd.DataFrame({"a": ["x", "y"] * 30, "y": [0, 1] * 30}), "no numeric feature column"),
        ("three.csv", pd.DataFrame({"a": range(60), "y": [0, 1, 2] * 20}), "distinct values"),
        ("few.csv", pd.DataFrame({"a": range(60), "y": [0] * 50 + [1] * 10}), "Too few rows"),
    ],
)
def test_rules_give_fixable_errors(tmp_path: Path, name: str, frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        load_dataset(_write(tmp_path, name, frame))


def test_messy_files_are_cleaned_with_a_note(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    df["id"] = [f"row{i}" for i in range(120)]  # text column: left out
    df.loc[[3, 7], "a"] = None  # incomplete rows: dropped
    df["y"] = [0, 0, 1] * 40
    with pytest.warns(UserWarning, match="left out 1 non-numeric column\\(s\\): id; dropped 2 of 120 rows"):
        data = load_dataset(_write(tmp_path, "messy.csv", df))
    assert (data.info.n_samples, data.info.n_features) == (118, 3)
    assert list(data.X.columns) == ["a", "b", "c"]
    assert (
        data.info.name == "messy.csv (left out 1 non-numeric column(s): id; dropped 2 of 120 rows with missing values)"
    )
    assert data.info.class_counts == {0: 78, 1: 40}


def test_missing_file_and_unknown_target(tmp_path: Path, heart_csv: Path) -> None:
    with pytest.raises(InvalidParameterError, match="No CSV file"):
        csv_dataset(tmp_path / "none.csv")
    with pytest.raises(InvalidParameterError, match="No column 'nope'"):
        load_dataset(heart_csv, target="nope")


def test_benchmark_dataset_resolution(heart_csv: Path) -> None:
    assert _resolve_dataset("heart", None) == "heart"
    name = _resolve_dataset(heart_csv, "target")
    assert is_csv_name(name)
    assert _resolve_dataset(name, None) == name
    assert _resolve_dataset(str(heart_csv), None) == f"csv:{heart_csv.resolve()}"
    with pytest.raises(InvalidParameterError, match="CSV files only"):
        _resolve_dataset("heart", "target")
    with pytest.raises(InvalidParameterError, match="registered CSV name"):
        _resolve_dataset(name, "target")
    with pytest.raises(UnknownDatasetError):
        _resolve_dataset("nope", None)
    frame = pd.read_csv(heart_csv)
    assert _resolve_dataset(frame, "target").endswith("dataframe.csv?target=target")
