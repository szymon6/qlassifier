"""Statistics of the protocol (TDD §7.7, ADR 0004).

Repeated random splits overlap, so a plain t-test over seeds is anti-conservative. Every CI here uses the
corrected resampled t (Nadeau & Bengio 2003): SE = sd * sqrt(1/J + n_test/n_train), t with J - 1 df.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MIN_TIE_SEEDS",
    "clip_unit",
    "corrected_ci",
    "mean_ci",
    "paired",
    "sign_test_counts",
    "tied_with_best",
    "wilcoxon_over_datasets",
]

FloatArray = NDArray[np.float64]

# A tie needs at least this many common seeds: with 2-4 seeds the corrected t has 1-3 degrees of freedom and its
# interval spans most of [0, 1], so "tied with the best" would mark almost every model (UAT, stage 1).
MIN_TIE_SEEDS = 5


def clip_unit(value: float) -> float:
    """A CI end clipped to the metric's range [0, 1] for display; the JSON keeps the raw value."""
    return min(1.0, max(0.0, float(value)))


def corrected_ci(
    diff: Sequence[float] | FloatArray, ratio: float, level: float = 0.95
) -> tuple[float, float | None, float | None, float | None]:
    """Mean, corrected CI and two-sided p of per-seed values (one-sample t against 0).

    Args:
        diff: per-seed values (paired differences, or raw scores for a CI of the mean).
        ratio: mean realised n_test / n_train of the splits.
        level: confidence level.

    Returns:
        ``(mean, ci_low, ci_high, p)``; CI and p are ``None`` below 2 values. Zero spread gives a point CI.
    """
    from scipy import stats

    d = np.asarray(diff, dtype=np.float64)
    J = len(d)
    m = float(d.mean())
    if J < 2:
        return m, None, None, None
    v = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / J + ratio) * v))
    if se == 0:
        return m, m, m, (0.0 if m != 0 else 1.0)
    t = m / se
    q = float(stats.t.ppf(0.5 + level / 2, J - 1))
    p = float(2 * stats.t.sf(abs(t), J - 1))
    return m, m - q * se, m + q * se, p


def mean_ci(
    values: Sequence[float] | FloatArray, ratio: float, level: float = 0.95
) -> tuple[float, float | None, float | None, float | None]:
    """Mean, sample sd and corrected CI of the mean of per-seed scores: ``(mean, sd, lo, hi)``."""
    x = np.asarray(values, dtype=np.float64)
    m, lo, hi, _ = corrected_ci(x, ratio, level)
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    return m, sd, lo, hi


def paired(
    a: Sequence[float] | FloatArray, b: Sequence[float] | FloatArray, ratio: float, level: float = 0.95
) -> tuple[float, float | None, float | None, float | None, float, int]:
    """Paired comparison a - b over seeds: ``(mean_diff, ci_low, ci_high, p, win_rate, n)``."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    m, lo, hi, p = corrected_ci(d, ratio, level)
    return m, lo, hi, p, float((d > 0).mean()), len(d)


def tied_with_best(
    per_seed: dict[str, dict[int, float]], ratio: float, level: float = 0.95, min_seeds: int = MIN_TIE_SEEDS
) -> dict[str, bool]:
    """Per model: does the corrected CI of its paired difference vs the top-mean model include 0?

    ``per_seed`` maps model id to ``{seed: value}``; differences use the seeds both models finished. The top
    model (first in model order on equal means) is tied with itself. Ties are evaluated only with at least
    ``min_seeds`` common seeds (TDD §7.7); below that every model, the top one included, reads not tied.
    """
    if not per_seed:
        return {}
    best = max(per_seed, key=lambda k: float(np.mean(list(per_seed[k].values()))))
    ref = per_seed[best]
    out: dict[str, bool] = {}
    for k, vals in per_seed.items():
        common = sorted(set(vals) & set(ref))
        if len(common) < min_seeds:
            out[k] = False
            continue
        if k == best:
            out[k] = True
            continue
        _, lo, hi, _ = corrected_ci([vals[s] - ref[s] for s in common], ratio, level)
        out[k] = lo is not None and hi is not None and lo <= 0.0 <= hi
    return out


def sign_test_counts(diffs: Sequence[float]) -> tuple[int, int, int]:
    """Across datasets: how many mean differences are positive, zero and negative."""
    d = np.asarray(diffs, dtype=np.float64)
    return int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


def wilcoxon_over_datasets(diffs: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p over per-dataset mean differences (None below 2 datasets or all zero)."""
    from scipy import stats

    d = np.asarray(diffs, dtype=np.float64)
    if len(d) < 2 or np.all(d == 0):
        return None
    return float(stats.wilcoxon(d).pvalue)
