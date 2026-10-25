"""The one tuning procedure: ridge grid, data-derived kernel scale, Gram-cached CV, selection, Platt (TDD §3.3).

Used by ``Qlassifier.fit`` (self-tuning) and by the benchmark engine (same grid, folds, score and tie rule;
ADR 0002, ADR 0004). Hyperparameters act after measurement, so tuning runs 0 circuits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CT_GRID",
    "M_GRID",
    "RIDGE_GRID",
    "FoldFeatures",
    "cv_kernel_scores",
    "gamma0",
    "gaussian_gram",
    "platt_fit",
    "platt_predict",
    "select",
]

FloatArray = NDArray[np.float64]
M_GRID: tuple[float, ...] = (1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0)
CT_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
# (gamma multiplier g = m^2, SVM C = C~/g) in grid order: simple -> flexible (narrow kernels last).
RIDGE_GRID: tuple[tuple[float, float], ...] = tuple((m**2, ct / m**2) for m in M_GRID for ct in CT_GRID)
FoldFeatures = tuple[FloatArray, NDArray[Any], FloatArray, NDArray[Any]]  # (F_a, y_a, F_b, y_b)


def gamma0(F: FloatArray) -> float:
    """Data scale of the Gaussian kernel: 1 / total variance of the training features (sum over components).

    Invariant to sign flips and permutations of components and to a uniform shrink of all Bloch vectors.
    """
    return float(1.0 / F.reshape(len(F), -1).var(axis=0).sum())


def gaussian_gram(FA: FloatArray, FB: FloatArray, gamma: float) -> FloatArray:
    """K_ij = exp(-gamma ||FA_i - FB_j||^2) for feature rows (any trailing shape, flattened)."""
    A = FA.reshape(len(FA), -1)
    B = FB.reshape(len(FB), -1)
    a2 = (A**2).sum(1)[:, None]
    b2 = (B**2).sum(1)[None, :]
    D = np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)
    K: FloatArray = np.exp(-gamma * D)
    return K


def cv_kernel_scores(folds: Sequence[FoldFeatures], grid: Sequence[tuple[float, float]]) -> FloatArray:
    """Balanced accuracy of every (g, C) config on every fold, shape (n_configs, n_folds).

    Per fold: gamma0 from the fold-train features; one Gram pair per bandwidth g (consecutive in the grid); C swept.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.svm import SVC

    scores = np.zeros((len(grid), len(folds)))
    for f, (FA, ya, FB, yb) in enumerate(folds):
        g0 = gamma0(FA)
        current: tuple[float, FloatArray, FloatArray] | None = None  # one Gram pair at a time (memory)
        for i, (g, C) in enumerate(grid):
            if current is None or current[0] != g:
                current = (g, gaussian_gram(FA, FA, g * g0), gaussian_gram(FB, FA, g * g0))
            _, Kaa, Kba = current
            model = SVC(C=C, kernel="precomputed").fit(Kaa, ya)
            scores[i, f] = balanced_accuracy_score(yb, model.predict(Kba))
    return scores


def select(scores: FloatArray) -> int:
    """Index of the config with the best mean score over folds; ties -> first in grid order."""
    mean = np.zeros(scores.shape[0])
    for f in range(scores.shape[1]):  # fold-by-fold sum, as the phase-7 harness (bit-identical ties)
        mean += scores[:, f]
    mean /= scores.shape[1]
    return int(np.argmax(mean))


def platt_fit(decision: FloatArray, y01: NDArray[Any], max_iter: int = 100) -> tuple[float, float]:
    """Fit P(y=1 | d) = 1 / (1 + exp(A d)) on out-of-fold decision values (Platt 1999, slope only).

    The intercept is fixed at 0 so probabilities agree with ``predict`` (P > 0.5 exactly when d > 0). Newton's method
    with backtracking on Platt's smoothed targets (Lin, Lin & Weng 2007). Returns ``(A, B)`` with ``B = 0``.
    """
    d = np.asarray(decision, dtype=np.float64)
    y = np.asarray(y01, dtype=np.int64)
    n_pos = float((y == 1).sum())
    n_neg = float(len(y) - n_pos)
    t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

    def objective(a: float) -> float:
        f = d * a
        return float(np.sum(np.where(f >= 0, t * f + np.log1p(np.exp(-f)), (t - 1) * f + np.log1p(np.exp(f)))))

    A = 0.0
    fval = objective(A)
    for _ in range(max_iter):
        p = platt_predict(d, A, 0.0)
        g = float(np.sum(d * (t - p)))
        h = float(np.sum(d * d * p * (1.0 - p))) + 1e-12
        if abs(g) < 1e-5:
            break
        step_dir = -g / h
        step = 1.0
        while step >= 1e-10:
            nA = A + step * step_dir
            nf = objective(nA)
            if nf < fval + 1e-4 * step * g * step_dir:
                A, fval = nA, nf
                break
            step /= 2.0
        else:
            break
    return min(A, -1e-12), 0.0  # a negative slope keeps P > 0.5 exactly when d > 0 (even for a useless margin)


def platt_predict(decision: FloatArray, A: float, B: float) -> FloatArray:
    """P(y = 1) for decision values under a fitted Platt sigmoid (numerically stable)."""
    f = np.asarray(decision, dtype=np.float64) * A + B
    out: FloatArray = np.where(f >= 0, np.exp(-f) / (1.0 + np.exp(-f)), 1.0 / (1.0 + np.exp(f)))
    return out
