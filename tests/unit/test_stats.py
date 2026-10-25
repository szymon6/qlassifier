"""Statistics (TDD §7.7, ADR 0004): corrected resampled t against the phase-7 reference numbers (e11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlassifier.bench import stats
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def e11() -> pd.DataFrame:
    """Per-seed balanced accuracy of e11 (protein_strands, chain-grouped splits, fresh seeds 200-209)."""
    return pd.read_csv(FIXTURES / "e11_protein_strands.csv")


def _ba(df: pd.DataFrame, model: str) -> np.ndarray:
    return df[df.model == model].sort_values("seed").ba.to_numpy()


def test_paired_difference_matches_e11(e11: pd.DataFrame) -> None:
    geo = e11[e11.model == "qlass_geo"].sort_values("seed")
    ratio = float((geo.n_test / geo.n_train).mean())  # mean realised n_te / n_tr of grouped splits
    assert ratio == pytest.approx(0.4437, abs=1e-4)
    m, lo, hi, p, wins, n = stats.paired(_ba(e11, "qlass_geo"), _ba(e11, "mlp"), ratio)
    assert (round(100 * m, 1), round(100 * lo, 1), round(100 * hi, 1)) == (0.7, -0.6, 2.1)  # e11 README
    assert round(p, 3) == 0.254
    assert (wins, n) == (0.9, 10)
    m, lo, hi, p, wins, _ = stats.paired(_ba(e11, "qlass_geo"), _ba(e11, "rbf_svm"), ratio)
    assert (round(100 * m, 1), round(100 * lo, 1), round(100 * hi, 1)) == (1.6, 0.3, 2.9)
    assert round(p, 3) == 0.021
    assert wins == 1.0


def test_corrected_ci_is_wider_than_naive(e11: pd.DataFrame) -> None:
    d = _ba(e11, "qlass_geo") - _ba(e11, "mlp")
    _, lo, hi, _ = stats.corrected_ci(d, 0.0)  # ratio 0: plain t
    _, clo, chi, _ = stats.corrected_ci(d, 0.4437)
    assert lo is not None
    assert hi is not None
    assert clo is not None
    assert chi is not None
    assert chi - clo > 2 * (hi - lo)  # sqrt((1/10 + 0.44) / (1/10)) = 2.3


def test_mean_ci_and_degenerate_cases() -> None:
    m, sd, lo, hi = stats.mean_ci([0.9, 0.92, 0.94], 0.43)
    assert m == pytest.approx(0.92)
    assert sd == pytest.approx(0.02)
    assert lo is not None
    assert hi is not None
    assert lo < m < hi
    assert stats.mean_ci([0.9], 0.43) == (0.9, None, None, None)
    assert stats.corrected_ci([0.5, 0.5, 0.5], 0.4) == (0.5, 0.5, 0.5, 0.0)
    assert stats.corrected_ci([0.0, 0.0], 0.4) == (0.0, 0.0, 0.0, 1.0)


def test_tied_with_best_uses_common_seeds() -> None:
    per_seed = {
        "a": {0: 0.90, 1: 0.92, 2: 0.94, 3: 0.91},
        "b": {0: 0.89, 1: 0.93, 2: 0.93, 3: 0.92},  # small, noisy differences -> tied
        "c": {0: 0.60, 1: 0.61, 2: 0.62, 3: 0.60},  # far below -> not tied
        "d": {0: 0.91},  # one common seed -> not decidable, not tied
    }
    ties = stats.tied_with_best(per_seed, 0.43, min_seeds=2)
    assert ties == {"a": True, "b": True, "c": False, "d": False}
    assert stats.tied_with_best({}, 0.4) == {}
    two = {"x": {0: 0.5, 1: 0.6}, "y": {0: 0.5, 1: 0.6}}
    assert stats.tied_with_best(two, 0.4, min_seeds=2) == {"x": True, "y": True}


def test_ties_need_five_seeds() -> None:
    """With 2-4 seeds the corrected t has 1-3 df and spans [0, 1]: no tie is declared, not even for the top model."""
    assert stats.MIN_TIE_SEEDS == 5
    four = {"a": {s: 0.9 + 0.01 * s for s in range(4)}, "b": dict.fromkeys(range(4), 0.5)}
    assert stats.tied_with_best(four, 0.43) == {"a": False, "b": False}
    five = {
        "a": {s: 0.9 + 0.01 * s for s in range(5)},
        "b": dict.fromkeys(range(5), 0.5),
        "c": dict.fromkeys(range(5), 0.91),
    }
    assert stats.tied_with_best(five, 0.43) == {"a": True, "b": False, "c": True}
    # a model with fewer common seeds than the minimum is not tied even when the top model has enough
    mixed = {"a": dict.fromkeys(range(6), 0.9), "b": dict.fromkeys(range(4), 0.9)}
    assert stats.tied_with_best(mixed, 0.43) == {"a": True, "b": False}


def test_clip_unit() -> None:
    assert stats.clip_unit(1.043) == 1.0
    assert stats.clip_unit(-0.2) == 0.0
    assert stats.clip_unit(0.826) == 0.826


def test_across_datasets() -> None:
    assert stats.sign_test_counts([0.01, -0.02, 0.0, 0.03]) == (2, 1, 1)
    p = stats.wilcoxon_over_datasets([0.01, 0.02, 0.015, 0.03, 0.025])
    assert p is not None
    assert 0 < p < 0.1
    assert stats.wilcoxon_over_datasets([0.01]) is None
    assert stats.wilcoxon_over_datasets([0.0, 0.0]) is None
