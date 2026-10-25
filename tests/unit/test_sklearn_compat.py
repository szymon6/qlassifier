"""scikit-learn compatibility (TDD §5.2, §9.1): clone, params, pickling, GridSearchCV, Pipeline, CV, checks."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import estimator_checks_generator

from qlassifier import Qlassifier, load_dataset

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

# Only checks that fit fewer than 20 rows fail, by design: TooFewSamplesError (PRD US-5: at least 20 rows and
# 5 per class to tune itself). Multiclass checks pass because MultiClassError uses scikit-learn's wording.
TOO_FEW = "fits fewer than 20 rows; Qlassifier raises TooFewSamplesError by design (PRD US-5)"
EXPECTED_FAILED = {
    "check_n_features_in_after_fitting": TOO_FEW + " (15 rows)",
    "check_estimators_nan_inf": TOO_FEW + " (10 rows)",
    "check_classifier_data_not_an_array": TOO_FEW + " (12 rows)",
    "check_fit2d_1feature": TOO_FEW + " (10 rows)",
}


# Not applicable: array-API dispatch needs SCIPY_ARRAY_API=1 at interpreter start, and Qlassifier declares no
# array-API support (tags.array_api_support is False); scikit-learn would only report it as skipped.
NOT_APPLICABLE = {"check_array_api_input"}


def _check_name(check: object) -> str:
    return str(getattr(getattr(check, "func", check), "__name__", check))


def _checks() -> list[object]:
    params = []
    for est in (Qlassifier(), Qlassifier(gamma=1.0, C=10.0)):
        for item in estimator_checks_generator(
            est, legacy=True, expected_failed_checks=EXPECTED_FAILED, mark="xfail", xfail_strict=True
        ):
            values = item.values if hasattr(item, "values") else item
            marks = item.marks if hasattr(item, "marks") else ()
            name = _check_name(values[1])
            if name in NOT_APPLICABLE:
                continue
            params.append(pytest.param(*values, marks=marks, id=f"{values[0]!r}-{name}"))
    return params


@pytest.mark.parametrize(("estimator", "check"), _checks())
def test_sklearn_estimator_checks(estimator, check) -> None:  # type: ignore[no-untyped-def]
    check(estimator)


def test_every_documented_exception_is_a_real_check() -> None:
    names = {_check_name(p.values[1]) for p in _checks()}  # type: ignore[attr-defined]
    assert set(EXPECTED_FAILED) <= names


def test_get_set_params_and_clone() -> None:
    clf = Qlassifier(geometry={"phi_0": 360}, backend="shots", shots=256, gamma=2.0, C="auto", random_state=4)
    params = clf.get_params()
    assert params == {"geometry": {"phi_0": 360}, "backend": "shots", "shots": 256, "gamma": 2.0, "C": "auto",
                      "random_state": 4}  # fmt: skip
    twin = clone(clf)
    assert twin.get_params() == params
    assert twin.geometry is not clf.geometry
    clf.set_params(shots=2048, gamma="auto")
    assert (clf.shots, clf.gamma) == (2048, "auto")


def test_pickle_exact(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, _ = breast_cancer_split
    clf = Qlassifier(random_state=0).fit(X_tr, y_tr)
    back = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(back.decision_function(X_te), clf.decision_function(X_te))
    assert back.resources_.circuits_fit == clf.resources_.circuits_fit


def test_pickle_fake_backend_keeps_transpiled_circuits() -> None:
    X, y = load_dataset("heart", return_X_y=True)
    clf = Qlassifier(backend="fake_fez", gamma=0.25, C=4.0, random_state=0).fit(X.iloc[:40], y.iloc[:40])
    back = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(back.predict(X.iloc[40:60]), clf.predict(X.iloc[40:60]))
    assert back.resources_.backend == "fake_fez"


def test_grid_search_over_gamma_and_c(breast_cancer_split: Split) -> None:
    X_tr, X_te, y_tr, y_te = breast_cancer_split
    search = GridSearchCV(Qlassifier(), {"gamma": [0.25, 1.0], "C": [1.0, 10.0]}, cv=3).fit(X_tr, y_tr)
    assert set(search.best_params_) == {"gamma", "C"}
    assert search.score(X_te, y_te) > 0.9


def test_pipeline_and_cross_val_score() -> None:
    X, y = load_dataset("breast_cancer", return_X_y=True)
    pipe = make_pipeline(StandardScaler(), PCA(8, random_state=0), Qlassifier(random_state=0))
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.mean() > 0.92
    assert pipe.fit(X, y)[-1].n_qubits_ == 4


@pytest.mark.parametrize("as_frame", [True, False])
def test_dataframes_and_arrays_agree(breast_cancer_split: Split, as_frame: bool) -> None:
    X_tr, X_te, y_tr, _ = breast_cancer_split
    clf = Qlassifier(gamma=0.25, C=4.0)
    if as_frame:
        pred = clf.fit(X_tr, y_tr).predict(X_te)
    else:
        pred = clf.fit(X_tr.to_numpy(), y_tr.to_numpy()).predict(X_te.to_numpy())
    ref = Qlassifier(gamma=0.25, C=4.0).fit(X_tr, y_tr).predict(X_te)
    assert np.array_equal(pred, ref)
