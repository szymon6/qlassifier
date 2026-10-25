"""Bundled datasets: shape, classes, names, SHA-256, groups, geometry, splits, drift rule, glyphs (TDD §5.4, §7)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from qlassifier import DatasetInfo, list_datasets, load_dataset
from qlassifier.datasets import DATA_DIR, glyph, outer_split, split_mode

EXPECTED = {
    # id: (rows, features, positives, positive class, recipe, grouped)
    "breast_cancer": (569, 30, 212, "malignant", "pca8", False),
    "ionosphere": (351, 33, 126, "bad", "pca8", False),
    "sonar": (208, 60, 97, "rock", "pca8", False),
    "heart": (297, 13, 137, "disease", "pca8", False),
    "protein_angles": (4000, 10, 996, "strand", "native", True),
}
PROTEIN = ["phi_m2", "psi_m2", "phi_m1", "psi_m1", "phi_0", "psi_0", "phi_p1", "psi_p1", "phi_p2", "psi_p2"]


@pytest.mark.parametrize("name", list(EXPECTED))
def test_shape_classes_and_info(name: str) -> None:
    rows, d, pos, pos_name, recipe, grouped = EXPECTED[name]
    data = load_dataset(name)
    assert data.X.shape == (rows, d)
    assert (data.X.dtypes == np.float64).all()
    assert set(np.unique(data.y)) == {0, 1}
    assert int(data.y.sum()) == pos
    assert pos < rows - pos  # label 1 = minority class
    info = data.info
    assert isinstance(info, DatasetInfo)
    assert (info.id, info.n_samples, info.n_features, info.positive_class) == (name, rows, d, pos_name)
    assert info.class_counts == {0: rows - pos, 1: pos}
    assert (info.recipe, info.grouped) == (recipe, grouped)
    assert info.licence
    assert info.citation
    assert info.source.startswith("https://")
    assert (data.groups is not None) == grouped
    assert not data.X.isna().any().any()


def test_list_datasets_in_dashboard_order() -> None:
    assert [i.id for i in list_datasets()] == list(EXPECTED)


def test_return_x_y() -> None:
    X, y = load_dataset("sonar", return_X_y=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (208, 60)


def test_column_names() -> None:
    assert list(load_dataset("protein_angles").X.columns) == PROTEIN
    assert list(load_dataset("heart").X.columns)[:3] == ["age", "sex", "cp"]
    iono = list(load_dataset("ionosphere").X.columns)
    assert iono[:2] == ["a01", "a03"]  # a02 is constant and dropped
    assert "a02" not in iono
    sonar = list(load_dataset("sonar").X.columns)
    assert (sonar[0], sonar[-1]) == ("e01", "e60")
    assert next(iter(load_dataset("breast_cancer").X.columns)) == "mean radius"


@pytest.mark.parametrize(
    ("file", "sha"),
    [
        ("ionosphere.csv", "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd"),
        ("sonar.csv", "e90434cdbf00fcf93ffa911fe447ae25606979658e60f1d32e155c3b5240234d"),
        ("heart.csv", "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"),
        ("protein_angles.csv", "b1d01c7566dcc3ce19745107a037946dfd11c26e57d52a3a9aa99af2c6e44555"),
    ],
)
def test_bundled_files_match_phase7_hashes(file: str, sha: str) -> None:
    assert hashlib.sha256((DATA_DIR / file).read_bytes()).hexdigest() == sha
    assert any(i.sha256 == sha for i in list_datasets())


def test_licences_file_lists_every_dataset() -> None:
    text = (DATA_DIR / "LICENSES.md").read_text()
    for needle in ("CC BY 4.0", "CC0 1.0", "BSD-3-Clause", "Berman", "Detrano", "Gorman", "Sigillito", "Street"):
        assert needle in text


def test_protein_geometry_and_groups() -> None:
    data = load_dataset("protein_angles")
    assert data.geometry == dict.fromkeys(PROTEIN, 360.0)
    assert data.groups is not None
    assert data.groups.nunique() == 1773
    assert data.groups.value_counts().max() <= 3
    assert data.X.abs().max().max() <= 180.0


def test_loaders_return_copies() -> None:
    a = load_dataset("heart")
    a.X.iloc[0, 0] = -1.0
    assert load_dataset("heart").X.iloc[0, 0] != -1.0


@pytest.mark.parametrize(
    ("name", "n_train", "n_test"), [("breast_cancer", 398, 171), ("heart", 207, 90), ("sonar", 145, 63)]
)
def test_stratified_split_sizes_and_shares(name: str, n_train: int, n_test: int) -> None:
    data = load_dataset(name)
    X_tr, X_te, y_tr, y_te = data.split(seed=3)
    assert (len(X_tr), len(X_te)) == (n_train, n_test)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_split_is_seeded() -> None:
    data = load_dataset("ionosphere")
    a, b = data.split(seed=1), data.split(seed=1)
    assert a[0].index.equals(b[0].index)
    assert not data.split(seed=2)[0].index.equals(a[0].index)


def test_protein_split_is_chain_grouped_and_drift_rule_keeps_plain_groups() -> None:
    data = load_dataset("protein_angles")
    y, g = data.y.to_numpy(), data.groups.to_numpy()  # type: ignore[union-attr]
    assert split_mode(y, g, range(10)) == "grouped"  # phase-8 pre-check: drift -0.68 to +2.81 pp <= 3 pp
    share = y.mean()
    n_te, ratios = [], []
    for seed in range(10):
        tr, te = outer_split(y, g, seed, "grouped")
        assert set(g[tr]).isdisjoint(g[te])  # no chain on both sides
        assert abs(y[te].mean() - share) <= 0.03
        n_te.append(len(te))
        ratios.append(len(te) / len(tr))
    assert min(n_te) == 1162
    assert max(n_te) == 1219
    assert round(float(np.mean(ratios)), 3) == 0.428
    X_tr, X_te, _, _ = data.split(seed=0)
    assert set(g[X_tr.index]).isdisjoint(g[X_te.index])


def test_drift_rule_switches_to_stratified_groups() -> None:
    # 40 groups of 10 rows: 30 negative groups, 10 positive groups -> some seeds drift > 3 pp
    y = np.repeat([0] * 30 + [1] * 10, 10)
    g = np.repeat(np.arange(40), 10)
    assert split_mode(y, g, range(10)) == "stratified_grouped"
    tr, te = outer_split(y, g, 0, "stratified_grouped")
    assert set(g[tr]).isdisjoint(g[te])
    assert len(te) == 120  # folds 0-2 of 10
    assert abs(y[te].mean() - y.mean()) <= 0.03
    assert split_mode(y, None, range(3)) == "stratified"


@pytest.mark.parametrize("name", list(EXPECTED))
def test_glyph(name: str) -> None:
    gl = glyph(name)
    assert gl["kind"] == ("torus" if name == "protein_angles" else "pca")
    pts = np.array(gl["points"])
    assert pts.shape == (120, 3)
    assert pts[:, :2].min() >= 0.0
    assert pts[:, :2].max() <= 1.0
    assert set(pts[:, 2]) == {0.0, 1.0}
