"""Tuning: ridge grid, gamma0 invariances, Gram, CV scores, tie rule, Platt (TDD §3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from qlassifier import load_dataset
from qlassifier.encoding import Encoding
from qlassifier.model import fold_features, tuning_grid
from qlassifier.tuning import (
    CT_GRID,
    M_GRID,
    RIDGE_GRID,
    cv_kernel_scores,
    gamma0,
    gaussian_gram,
    platt_fit,
    platt_predict,
    select,
)


@pytest.fixture(scope="module")
def heart_features() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_dataset("heart", return_X_y=True)
    enc = Encoding.plan(13, None, None).fit(X.to_numpy())
    return enc.bloch(X.to_numpy()), y.to_numpy()


def test_ridge_grid_order_and_size() -> None:
    assert len(RIDGE_GRID) == 30
    assert RIDGE_GRID[0] == (1 / 64, 0.1 * 64)
    assert RIDGE_GRID[4] == (1 / 64, 1000 * 64)
    assert RIDGE_GRID[-1] == (16.0, 1000 / 16)
    gs = [g for g, _ in RIDGE_GRID]
    assert gs == sorted(gs)  # simple (wide kernels) -> flexible
    for (g, C), (m, ct) in zip(RIDGE_GRID, [(m, ct) for m in M_GRID for ct in CT_GRID], strict=True):
        assert g == m**2
        assert C * g == pytest.approx(ct)


def test_tuning_grid_fixes_an_axis() -> None:
    assert tuning_grid("auto", "auto") == RIDGE_GRID
    assert tuning_grid(2.0, "auto") == tuple((2.0, ct / 2.0) for ct in CT_GRID)
    assert tuning_grid("auto", 5.0) == tuple((m**2, 5.0) for m in M_GRID)
    assert tuning_grid(2.0, 5.0) == ((2.0, 5.0),)


def test_gamma0_invariances(heart_features: tuple[np.ndarray, np.ndarray]) -> None:
    F, _ = heart_features
    g = gamma0(F)
    assert g == pytest.approx(1.0 / F.reshape(len(F), -1).var(axis=0).sum())
    assert gamma0(-F) == pytest.approx(g)  # sign flips
    assert gamma0(F[:, ::-1]) == pytest.approx(g)  # qubit permutation
    shrunk = 0.7 * F  # uniform shrink of every Bloch vector: the kernel stays the same
    assert np.allclose(gaussian_gram(shrunk, shrunk, gamma0(shrunk)), gaussian_gram(F, F, g))


def test_gaussian_gram(heart_features: tuple[np.ndarray, np.ndarray]) -> None:
    F, _ = heart_features
    K = gaussian_gram(F[:30], F[:30], 0.5)
    flat = F[:30].reshape(30, -1)
    ref = np.exp(-0.5 * ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1))
    assert np.allclose(K, ref, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_cv_scores_shape_and_range(heart_features: tuple[np.ndarray, np.ndarray]) -> None:
    from sklearn.model_selection import StratifiedKFold

    F, y = heart_features
    folds = [(F[a], y[a], F[b], y[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y)]
    scores = cv_kernel_scores(folds, RIDGE_GRID)
    assert scores.shape == (30, 5)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert scores.mean(axis=1).max() > 0.75


def test_select_takes_first_of_tied_configs() -> None:
    scores = np.array([[0.7, 0.7], [0.8, 0.9], [0.9, 0.8], [0.85, 0.85]])
    assert select(scores) == 1  # 0.85 ties: rows 1, 2, 3 -> first in grid order
    assert select(np.array([[0.5], [0.6], [0.6]])) == 1


def test_fold_features_fit_scaler_on_fold_train_only() -> None:
    X, y = load_dataset("sonar", return_X_y=True)
    Xa, Xb = X.to_numpy()[:100], X.to_numpy()[100:]
    plan = Encoding.plan(60, None, None)
    _Fa, _, Fb, _ = fold_features(plan, Xa, y.to_numpy()[:100], Xb, y.to_numpy()[100:])
    expected = plan.fit(Xa).bloch(Xb)
    assert np.array_equal(Fb, expected)
    assert not np.allclose(Fb, plan.fit(Xb).bloch(Xb))


def test_platt_probabilities(heart_features: tuple[np.ndarray, np.ndarray]) -> None:
    from sklearn.svm import SVC

    F, y = heart_features
    K = gaussian_gram(F, F, gamma0(F))
    d = SVC(C=1.0, kernel="precomputed").fit(K[:200, :200], y[:200]).decision_function(K[200:, :200])
    A, B = platt_fit(d, y[200:])
    assert A < 0  # larger margin -> higher P(class 1)
    p = platt_predict(d, A, B)
    assert np.all((p > 0) & (p < 1))
    assert np.corrcoef(p, d)[0, 1] > 0.9
    proba = np.column_stack([1 - p, p])
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_platt_slope_stays_negative_for_a_useless_margin() -> None:
    d = np.array([-2.0, -1.0, 1.0, 2.0])
    A, B = platt_fit(d, np.array([1, 1, 0, 0]))  # margins anti-correlated with the labels
    assert A < 0
    assert B == 0.0
    p = platt_predict(d, A, B)
    assert np.array_equal(p > 0.5, d > 0)


def test_platt_handles_separable_and_extreme_margins() -> None:
    d = np.array([-50.0, -3.0, -1.0, 1.0, 3.0, 50.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    A, B = platt_fit(d, y)
    p = platt_predict(np.array([-1e4, 0.0, 1e4]), A, B)
    assert np.all(np.isfinite(p))
    assert p[0] < 0.5 < p[2]
