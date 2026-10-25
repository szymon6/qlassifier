"""Bundled real binary datasets, loaded offline, plus the protocol's outer split (TDD §5.4, §7.1–7.2).

Files in ``data/`` are byte-identical copies of the phase-7 files and are checked against their SHA-256 hash
at load (``DatasetIntegrityError``). Breast Cancer comes from scikit-learn's bundled copy of UCI 17.
Licences and citations: ``data/LICENSES.md`` · Decision record: ADR 0003.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qlassifier.errors import DATASET_IDS, DatasetIntegrityError, InvalidParameterError, UnknownDatasetError

__all__ = [
    "Dataset",
    "DatasetInfo",
    "SplitMode",
    "csv_dataset",
    "glyph",
    "is_csv_name",
    "list_datasets",
    "load_dataset",
    "outer_split",
    "split_mode",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_SIZE = 0.3
DRIFT_LIMIT = 0.03  # grouped splits: max |test positive share - dataset share| before the stratified variant
GLYPH_ROWS = 120
CSV_PREFIX = "csv:"
CSV_TARGET_SEP = "?target="
MAX_CSV_ROWS = 1_000  # above this a stratified sample (seed 0) is used: kernel models cost N^2; ~5 min at the cap
MIN_CSV_CLASS_ROWS = 20  # 70/30 split and 5 inner folds need rows of both classes everywhere
SplitMode = Literal["stratified", "grouped", "stratified_grouped"]
Recipe = Literal["pca8", "native"]


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset metadata: size, class balance, source, licence, citation and benchmark recipe.

    ``class_counts`` maps label to row count; label 1 is the positive (minority) class.
    """

    id: str
    name: str
    n_samples: int
    n_features: int
    class_counts: dict[int, int]
    positive_class: str
    source: str
    licence: str
    citation: str
    recipe: Recipe
    grouped: bool
    sha256: str | None


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset.

    Attributes:
        X: raw feature columns, float64, named.
        y: labels in {0, 1}; 1 is the positive (minority) class.
        groups: source unit per row for leakage-safe splits (``protein_angles``: PDB chain id), else None.
        geometry: angle-native columns for ``Qlassifier(geometry=...)`` (``protein_angles``), else None.
        info: metadata.
    """

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    geometry: dict[str, float] | None
    info: DatasetInfo

    def split(self, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Protocol outer split (TDD §7.2): stratified 70/30, or chain-grouped for grouped data.

        Args:
            seed: split seed (the benchmark uses seeds 0-9).

        Returns:
            ``(X_train, X_test, y_train, y_test)``.

        Example:
            >>> X_train, X_test, y_train, y_test = load_dataset("heart").split(seed=0)
            >>> len(X_train), len(X_test)
            (207, 90)
        """
        groups = None if self.groups is None else self.groups.to_numpy()
        y = self.y.to_numpy()
        mode = split_mode(y, groups, range(10))
        tr, te = outer_split(y, groups, seed, mode)
        return self.X.iloc[tr], self.X.iloc[te], self.y.iloc[tr], self.y.iloc[te]


@dataclass(frozen=True)
class _Spec:
    id: str
    name: str
    file: str | None
    sha256: str | None
    positive_class: str
    source: str
    licence: str
    citation: str
    recipe: Recipe
    parse: Callable[[Path | None], tuple[pd.DataFrame, NDArray[np.int64], NDArray[Any] | None]]
    geometry: dict[str, float] | None = None


def _parse_breast_cancer(_: Path | None) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=[str(c) for c in d.feature_names])
    return X, (1 - d.target).astype(np.int64), None  # 1 = malignant (212)


def _parse_ionosphere(path: Path | None) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].set_axis([f"a{j + 1:02d}" for j in range(df.shape[1] - 1)], axis=1)
    return X, (df.iloc[:, -1] == "b").to_numpy(np.int64), None  # 1 = bad (126)


def _parse_sonar(path: Path | None) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].set_axis([f"e{j + 1:02d}" for j in range(df.shape[1] - 1)], axis=1)
    return X, (df.iloc[:, -1] == "R").to_numpy(np.int64), None  # 1 = rock (97)


_HEART = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def _parse_heart(path: Path | None) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(path, header=None, na_values="?").dropna().reset_index(drop=True)
    X = df.iloc[:, :13].set_axis(_HEART, axis=1)
    return X, (df.iloc[:, 13] > 0).to_numpy(np.int64), None  # 1 = disease (137 of 297)


_PROTEIN = [f"{a}_{t}" for t in ("m2", "m1", "0", "p1", "p2") for a in ("phi", "psi")]


def _parse_protein(path: Path | None) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[Any]]:
    df = pd.read_csv(path)
    return df[_PROTEIN].copy(), (df["ss"] == "E").to_numpy(np.int64), df["pdb"].to_numpy()  # 1 = strand (996)


_SPECS: dict[str, _Spec] = {
    s.id: s
    for s in (
        _Spec(
            "breast_cancer", "Breast Cancer (Wisconsin)", None, None, "malignant",
            "https://archive.ics.uci.edu/dataset/17", "CC BY 4.0 (UCI); BSD-3-Clause (scikit-learn copy)",
            "Street, Wolberg, Mangasarian (1993), Proc. SPIE 1905:861-870", "pca8", _parse_breast_cancer,
        ),
        _Spec(
            "ionosphere", "Ionosphere", "ionosphere.csv",
            "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd", "bad",
            "https://archive.ics.uci.edu/dataset/52", "CC BY 4.0",
            "Sigillito, Wing, Hutton, Baker (1989), Johns Hopkins APL Tech. Dig. 10:262-266", "pca8",
            _parse_ionosphere,
        ),
        _Spec(
            "sonar", "Sonar (mines vs rocks)", "sonar.csv",
            "e90434cdbf00fcf93ffa911fe447ae25606979658e60f1d32e155c3b5240234d", "rock",
            "https://archive.ics.uci.edu/dataset/151", "CC BY 4.0",
            "Gorman, Sejnowski (1988), Neural Networks 1:75-89", "pca8", _parse_sonar,
        ),
        _Spec(
            "heart", "Heart Disease (Cleveland)", "heart.csv",
            "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8", "disease",
            "https://archive.ics.uci.edu/dataset/45", "CC BY 4.0",
            "Detrano et al. (1989), Am. J. Cardiol. 64:304-310", "pca8", _parse_heart,
        ),
        _Spec(
            "protein_angles", "Protein backbone angles", "protein_angles.csv",
            "b1d01c7566dcc3ce19745107a037946dfd11c26e57d52a3a9aa99af2c6e44555", "strand",
            "https://www.rcsb.org", "CC0 1.0", "Berman et al. (2000), Nucleic Acids Res. 28:235-242", "native",
            _parse_protein, dict.fromkeys(_PROTEIN, 360.0),
        ),
    )
}  # fmt: skip
assert tuple(_SPECS) == DATASET_IDS


def is_csv_name(name: object) -> bool:
    """True for the dataset names :func:`csv_dataset` returns (``csv:<path>?target=<column>``)."""
    return isinstance(name, str) and name.startswith(CSV_PREFIX)


def looks_like_csv(name: object) -> bool:
    """True for a path-like argument that names a CSV file (``.csv`` suffix or an existing file)."""
    return isinstance(name, str | Path) and (str(name).lower().endswith(".csv") or Path(name).is_file())


def csv_dataset(path: str | Path, *, target: str | None = None) -> str:
    """Register a CSV file as a dataset and return its name for :func:`load_dataset` and ``benchmark``.

    Rules: a header row; one binary target column (``target``, default: the last column); at least 20 rows per
    class. Rows with a missing value are dropped and non-numeric columns are left out, each with a warning that
    also appears in the dataset name. Above 1,000 rows a stratified sample (seed 0) is used (the full leaderboard
    takes a few minutes at that size). Label 1 is the minority
    value. The file is parsed once per process and hashed (SHA-256) for provenance; the benchmark's ``pca8`` recipe
    applies (standardise, PCA to 8 features, standardise; fit on training rows only).

    Args:
        path: the CSV file.
        target: target column name (default: the last column).

    Returns:
        The dataset name, ``csv:<absolute path>?target=<column>``; its id is the file stem.

    Raises:
        InvalidParameterError: the file is missing or breaks one of the rules (the message says which).

    Example:
        >>> name = csv_dataset("my_data.csv", target="label")  # doctest: +SKIP
        >>> load_dataset(name).info.n_samples  # doctest: +SKIP
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise InvalidParameterError(f"No CSV file at '{p}'.", "Give the path of an existing .csv file.")
    name = f"{CSV_PREFIX}{p.resolve()}" + ("" if target is None else f"{CSV_TARGET_SEP}{target}")
    _spec(name)  # parse and validate now, so errors surface before any run
    return name


def _parse_csv(path: Path, target: str | None) -> tuple[pd.DataFrame, NDArray[np.int64], str, str]:
    """Read and validate a user CSV: ``(X, y, positive_class, note)``; y is 1 for the minority value."""
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise InvalidParameterError(
            f"Could not read '{path.name}' as CSV: {exc}", "Give a comma-separated file with a header row."
        ) from None
    if df.shape[1] < 2:
        raise InvalidParameterError(
            f"'{path.name}' has {df.shape[1]} column(s); a dataset needs feature columns and a target column.",
            "Give a CSV with a header row, numeric feature columns and one binary target column.",
        )
    df.columns = [str(c) for c in df.columns]
    target = str(df.columns[-1]) if target is None else target
    if target not in df.columns:
        raise InvalidParameterError(
            f"No column '{target}' in '{path.name}'.", f"Choose one of: {', '.join(df.columns)}."
        )
    notes = []
    features = df.drop(columns=[target])
    text = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if text:
        features = features.drop(columns=text)
        shown = ", ".join(text[:5]) + ("…" if len(text) > 5 else "")
        notes.append(f"left out {len(text)} non-numeric column(s): {shown}")
    if features.shape[1] == 0:
        raise InvalidParameterError(
            f"'{path.name}' has no numeric feature column.", "Encode the feature columns as numbers."
        )
    complete = ~pd.concat([features, df[[target]]], axis=1).isna().any(axis=1)
    missing = int((~complete).sum())
    if missing:
        notes.append(f"dropped {missing:,} of {len(df):,} rows with missing values")
        features, df = features[complete], df[complete]
    if len(df) == 0:
        raise InvalidParameterError(f"Every row of '{path.name}' has a missing value.", "Impute or fill the file.")
    labels = df[target]
    values = sorted(pd.unique(labels), key=str)
    if len(values) != 2:
        raise InvalidParameterError(
            f"Target '{target}' has {len(values)} distinct values; it must have exactly 2.",
            "Give a binary target column, or pass target=<column>.",
        )
    counts = labels.value_counts()
    a, b = values
    positive = b if counts[a] == counts[b] else (a if counts[a] < counts[b] else b)
    y = (labels == positive).to_numpy(dtype=np.int64)
    low = min(int(counts[a]), int(counts[b]))
    if low < MIN_CSV_CLASS_ROWS:
        raise InvalidParameterError(
            f"Too few rows in '{path.name}': the smaller class has {low} rows; each class needs at least "
            f"{MIN_CSV_CLASS_ROWS}.",
            "Add rows or choose another target column.",
        )
    if len(df) > MAX_CSV_ROWS:
        from sklearn.model_selection import train_test_split

        keep, _ = train_test_split(np.arange(len(df)), train_size=MAX_CSV_ROWS, stratify=y, random_state=0)
        keep = np.sort(keep)
        features, y = features.iloc[keep], y[keep]
        notes.append(f"{MAX_CSV_ROWS:,}-row stratified sample of {len(df):,}")
    note = f" ({'; '.join(notes)})" if notes else ""
    if notes:
        warnings.warn(f"{path.name}: {'; '.join(notes)}.", stacklevel=4)
    return features.astype(np.float64), y, str(positive), note


@cache
def _csv_spec(name: str) -> _Spec:
    body = name.removeprefix(CSV_PREFIX)
    path_str, sep, target = body.partition(CSV_TARGET_SEP)
    path = Path(path_str)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise InvalidParameterError(f"No CSV file at '{path}'.", "Give the path of an existing .csv file.") from None
    X, y, positive, note = _parse_csv(path, target if sep else None)
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "csv"
    ident = stem if stem not in _SPECS else f"{stem}_csv"
    return _Spec(
        id=ident,
        name=f"{path.name}{note}",
        file=None,
        sha256=digest,
        positive_class=positive,
        source=str(path),
        licence="user data",
        citation="user data (not redistributed)",
        recipe="pca8",
        parse=lambda _: (X, y, None),
    )


def _spec(name: str) -> _Spec:
    if is_csv_name(name):
        return _csv_spec(name)
    if not isinstance(name, str) or name not in _SPECS:
        raise UnknownDatasetError(name)
    return _SPECS[name]


@cache
def _verified(file: str, sha256: str) -> Path:
    path = DATA_DIR / file
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise DatasetIntegrityError(file) from None
    if digest != sha256:
        raise DatasetIntegrityError(file)
    return path


@cache
def _load(name: str) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[Any] | None]:
    spec = _spec(name)
    path = _verified(spec.file, spec.sha256) if spec.file is not None and spec.sha256 is not None else None
    X, y, groups = spec.parse(path)
    X = X.astype(np.float64)
    X = X.loc[:, X.to_numpy().std(axis=0) > 0]  # zero-variance columns carry nothing (Ionosphere a02)
    return X, y, groups


def _info(name: str) -> DatasetInfo:
    spec = _spec(name)
    X, y, groups = _load(name)
    counts = np.bincount(y, minlength=2)
    return DatasetInfo(
        id=spec.id,
        name=spec.name,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        class_counts={0: int(counts[0]), 1: int(counts[1])},
        positive_class=spec.positive_class,
        source=spec.source,
        licence=spec.licence,
        citation=spec.citation,
        recipe=spec.recipe,
        grouped=groups is not None,
        sha256=spec.sha256,
    )


@overload
def load_dataset(name: str | Path, *, target: str | None = ..., return_X_y: Literal[False] = ...) -> Dataset: ...
@overload
def load_dataset(
    name: str | Path, *, target: str | None = ..., return_X_y: Literal[True]
) -> tuple[pd.DataFrame, pd.Series]: ...
def load_dataset(
    name: str | Path, *, target: str | None = None, return_X_y: bool = False
) -> Dataset | tuple[pd.DataFrame, pd.Series]:
    """Load a bundled dataset (offline) or your own CSV file.

    Args:
        name: ``"breast_cancer"``, ``"ionosphere"``, ``"sonar"``, ``"heart"`` or ``"protein_angles"``; or the path
            of a CSV file (numeric feature columns, one binary target column; rules in :func:`csv_dataset`).
        target: for a CSV file, its target column (default: the last column).
        return_X_y: return ``(X, y)`` instead of a :class:`Dataset`.

    Returns:
        A :class:`Dataset` with raw named columns, or ``(X, y)``. y is 1 for the positive (minority) class.

    Raises:
        UnknownDatasetError: no bundled dataset with this name.
        DatasetIntegrityError: a bundled file failed its SHA-256 check.
        InvalidParameterError: a CSV file breaks one of the rules (the message says which).

    Example:
        >>> X, y = load_dataset("breast_cancer", return_X_y=True)
        >>> X.shape
        (569, 30)
        >>> data = load_dataset("my_data.csv", target="label")  # doctest: +SKIP
    """
    if target is not None or looks_like_csv(name):
        name = csv_dataset(name, target=target)
    name = str(name)
    spec = _spec(name)
    frame, labels, groups = _load(name)
    X = frame.copy()
    y = pd.Series(labels.copy(), name="target")
    if return_X_y:
        return X, y
    return Dataset(
        X=X,
        y=y,
        groups=None if groups is None else pd.Series(groups.copy(), name="chain"),
        geometry=None if spec.geometry is None else dict(spec.geometry),
        info=_info(name),
    )


def list_datasets() -> list[DatasetInfo]:
    """Return metadata for every bundled dataset, in dashboard order."""
    return [_info(name) for name in _SPECS]


# ----------------------------------------------------------------------------- protocol outer split (§7.2)
def split_mode(y: NDArray[Any], groups: NDArray[Any] | None, seeds: Sequence[int] | range) -> SplitMode:
    """Decide the outer split rule from labels and groups only, before any model runs (drift rule, §7.2).

    Ungrouped data: ``"stratified"``. Grouped data: ``"grouped"`` unless any seed's test positive share drifts
    more than 3 pp from the dataset share; then every seed uses ``"stratified_grouped"``.
    """
    if groups is None:
        return "stratified"
    share = float(np.mean(y))
    for seed in seeds:
        _, te = outer_split(y, groups, int(seed), "grouped")
        if abs(float(np.mean(y[te])) - share) > DRIFT_LIMIT:
            return "stratified_grouped"
    return "grouped"


def outer_split(
    y: NDArray[Any], groups: NDArray[Any] | None, seed: int, mode: SplitMode
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return ``(train_idx, test_idx)`` of the protocol outer split for one seed.

    - ``stratified``: ``StratifiedShuffleSplit(1, test_size=0.3, random_state=seed)``.
    - ``grouped``: ``GroupShuffleSplit(1, test_size=0.3, random_state=seed)``; no group on both sides.
    - ``stratified_grouped``: ``StratifiedGroupKFold(10, shuffle=True, random_state=seed)``, folds 0-2 = test.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit

    X0 = np.zeros((len(y), 1))
    if mode == "stratified":
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(X0, y))
    elif mode == "grouped":
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(X0, y, groups))
    else:
        folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(X0, y, groups))
        te = np.sort(np.concatenate([folds[k][1] for k in range(3)]))
        tr = np.setdiff1d(np.arange(len(y)), te)
    return np.asarray(tr, dtype=np.intp), np.asarray(te, dtype=np.intp)


# ----------------------------------------------------------------------------- dashboard glyph (§6.3)
@cache
def glyph(name: str) -> dict[str, Any]:
    """Return the dataset's picture-tile glyph: 120 stratified rows (seed 0), display only.

    ``kind="pca"``: first two principal components scaled to [0, 1]. ``kind="torus"`` (angle data): the
    central (phi, psi) pair mapped to [0, 1]² (a Ramachandran square). Each point is ``[x, y, class]``.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    spec = _spec(name)
    frame, y, _ = _load(name)
    keep, _ = train_test_split(np.arange(len(y)), train_size=GLYPH_ROWS, stratify=y, random_state=0)
    keep = np.sort(keep)
    if spec.geometry is not None:
        pts = (frame[["phi_0", "psi_0"]].to_numpy()[keep] + 180.0) / 360.0
        kind = "torus"
    else:
        Z = StandardScaler().fit_transform(frame.to_numpy())
        P = PCA(2, random_state=0).fit_transform(Z)[keep]
        lo, hi = P.min(axis=0), P.max(axis=0)
        pts = (P - lo) / np.where(hi > lo, hi - lo, 1.0)
        kind = "pca"
    points = [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(pts, y[keep], strict=True)]
    return {"kind": kind, "points": points}
