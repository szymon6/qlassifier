"""Qlassifier estimator: self-tuning, fitted attributes, resources, determinism, labels, probabilities (TDD §5.2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qlassifier import Qlassifier, Resources, ShotNoiseWarning, load_dataset
from qlassifier.tuning import RIDGE_GRID

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@pytest.fixture(scope="module")
def fitted(breast_cancer_split: Split) -> Qlassifier:
    X_tr, _, y_tr, _ = breast_cancer_split
    return Qlassifier(random_state=0).fit(X_tr, y_tr)


def test_quickstart_scores_well(fitted: Qlassifier, breast_cancer_split: Split) -> None:
    _, X_te, _, y_te = breast_cancer_split
    assert fitted.score(X_te, y_te) > 0.93


def test_fitted_attributes(fitted: Qlassifier, breast_cancer_split: Split) -> None:
    X_tr = breast_cancer_split[0]
    assert list(fitted.classes_) == [0, 1]
    assert fitted.n_features_in_ == 30
    assert list(fitted.feature_names_in_) == list(X_tr.columns)
    assert fitted.n_qubits_ == 15
    assert len(fitted.encoding_) == 15
    assert fitted.encoding_[0].columns == ("mean radius", "mean texture")
    assert (fitted.best_params_["gamma"], fitted.best_params_["C"]) in RIDGE_GRID
    assert fitted.gamma0_ > 0
    cv = fitted.cv_results_
    assert len(cv["params"]) == 30
    assert cv["mean_test_score"].shape == (30,)
    assert cv["std_test_score"].shape == (30,)
    best = int(np.argmax(cv["mean_test_score"]))
    assert cv["params"][best] == fitted.best_params_
    assert cv["rank_test_score"][best] == 1
    assert cv["rank_test_score"].min() == 1


def test_resources_exact(fitted: Qlassifier, breast_cancer_split: Split) -> None:
    X_te = breast_cancer_split[1]
    clf = Qlassifier(gamma=1.0, C=10.0).fit(breast_cancer_split[0], breast_cancer_split[2])
    r = clf.resources_
    assert isinstance(r, Resources)
    assert r == Resources("exact", 15, 0, 5, 3, 3 * 398, 0, 0, None)
    clf.predict(X_te)
    clf.predict(X_te.iloc[:10])
    assert clf.resources_.circuits_predict == 3 * (171 + 10)
    assert clf.resources_.to_dict()["circuits_fit"] == 1194
    assert not hasattr(Qlassifier(), "resources_")


def test_fixed_hyperparameters_skip_tuning(breast_cancer_split: Split) -> None:
    X_tr, _X_te, y_tr, _ = breast_cancer_split
    clf = Qlassifier(gamma=0.25, C=4.0).fit(X_tr, y_tr)
    assert clf.cv_results_ == {}
    assert clf.best_params_ == {"gamma": 0.25, "C": 4.0}
    one_axis = Qlassifier(gamma=0.25).fit(X_tr, y_tr)
    assert [p["gamma"] for p in one_axis.cv_results_["params"]] == [0.25] * 5
    assert len(Qlassifier(C=4.0).fit(X_tr, y_tr).cv_results_["params"]) == 6


def test_random_state_makes_runs_repeatable(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, _ = breast_cancer_split
    a = Qlassifier(random_state=3).fit(X_tr, y_tr)
    b = Qlassifier(random_state=3).fit(X_tr, y_tr)
    assert a.best_params_ == b.best_params_
    assert np.array_equal(a.decision_function(X_te), b.decision_function(X_te))
    c = Qlassifier().fit(X_tr, y_tr)  # None: unshuffled folds, exact mode is still deterministic
    d = Qlassifier().fit(X_tr, y_tr)
    assert np.array_equal(c.predict(X_te), d.predict(X_te))


def test_any_two_labels_and_arrays(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, _ = breast_cancer_split
    labels = np.where(y_tr.to_numpy() == 1, "malignant", "benign")
    clf = Qlassifier(gamma=0.25, C=4.0).fit(X_tr.to_numpy(), labels)
    assert list(clf.classes_) == ["benign", "malignant"]
    pred = clf.predict(X_te.to_numpy())
    assert set(pred) <= {"benign", "malignant"}
    frame = Qlassifier(gamma=0.25, C=4.0).fit(X_tr, y_tr).predict(X_te)
    assert np.array_equal(pred == "malignant", frame == 1)
    assert not hasattr(clf, "feature_names_in_")


def test_probabilities_and_margins(fitted: Qlassifier, breast_cancer_split: Split) -> None:
    X_te = breast_cancer_split[1]
    proba = fitted.predict_proba(X_te)
    assert proba.shape == (171, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    d = fitted.decision_function(X_te)
    assert d.shape == (171,)
    pred = fitted.predict(X_te)
    assert np.array_equal(pred, fitted.classes_[(d > 0).astype(int)])
    order = np.argsort(d)
    assert np.all(np.diff(proba[order, 1]) >= -1e-12)  # Platt is monotone in the margin


def test_feature_count_mismatch_uses_sklearn_message(fitted: Qlassifier, breast_cancer_split: Split) -> None:
    with pytest.warns(UserWarning, match="feature names"), pytest.raises(ValueError, match="features"):
        fitted.predict(breast_cancer_split[1].to_numpy()[:, :5])


def test_tags() -> None:
    tags = Qlassifier().__sklearn_tags__()
    assert tags.classifier_tags.multi_class is False
    assert tags.input_tags.allow_nan is False
    assert tags.non_deterministic is False
    assert Qlassifier(backend="shots").__sklearn_tags__().non_deterministic is True
    assert Qlassifier(backend="shots", random_state=1).__sklearn_tags__().non_deterministic is False


def test_shots_backend_is_seeded_and_resources_report_shots(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, y_te = breast_cancer_split
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShotNoiseWarning)  # Breast Cancer: shot noise is harmless
        clf = Qlassifier(backend="shots", random_state=0).fit(X_tr, y_tr)
    first = clf.predict(X_te)
    assert np.array_equal(first, clf.predict(X_te))  # same predict stream every call
    assert clf.score(X_te, y_te) > 0.9
    r = clf.resources_
    assert (r.backend, r.shots, r.two_qubit_gates, r.tuning_circuits) == ("shots", 1024, 0, 0)
    other = Qlassifier(backend="shots", random_state=1).fit(X_tr, y_tr)
    assert not np.array_equal(clf._F, other._F)


def test_shots_without_seed_is_fresh(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, _ = breast_cancer_split
    clf = Qlassifier(backend="shots", gamma=0.25, C=4.0).fit(X_tr, y_tr)
    a, b = clf.decision_function(X_te), clf.decision_function(X_te)
    assert not np.array_equal(a, b)


def test_geometry_mode_on_protein_angles() -> None:
    data = load_dataset("protein_angles")
    X_tr, X_te, y_tr, y_te = data.split(seed=0)
    clf = Qlassifier(geometry=data.geometry, gamma=0.25, C=40.0).fit(X_tr.iloc[:800], y_tr.iloc[:800])
    assert clf.n_qubits_ == 10
    assert all(q.kind == "circle" for q in clf.encoding_)
    assert clf.score(X_te, y_te) > 0.85
    arr = Qlassifier(geometry=dict.fromkeys(range(10), 360.0), gamma=0.25, C=40.0)
    arr.fit(X_tr.to_numpy()[:800], y_tr.to_numpy()[:800])
    assert np.array_equal(arr.predict(X_te.to_numpy()), clf.predict(X_te))
