"""Shot-noise guard (TDD §3.3 step 5, ADR 0002): warns on narrow-kernel data, stays quiet on Breast Cancer."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qlassifier import Qlassifier, ShotNoiseWarning

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def test_warns_when_the_label_needs_a_narrow_kernel() -> None:
    rng = np.random.default_rng(0)  # mechanism test: a label that flips every 0.25 units along one axis
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (np.sin(12 * X[:, 0]) > 0).astype(int)
    with pytest.warns(ShotNoiseWarning) as record:
        Qlassifier(backend="shots", shots=64, random_state=0).fit(X, y)
    w = record[0].message
    assert isinstance(w, ShotNoiseWarning)
    assert w.shots == 64
    assert w.pp > 2.0
    assert str(w).startswith("Shot noise at shots=64 is expected to cost ")


def test_quiet_on_breast_cancer(breast_cancer_split: Split) -> None:
    X_tr, _, y_tr, _ = breast_cancer_split
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShotNoiseWarning)
        Qlassifier(backend="shots", random_state=0).fit(X_tr, y_tr)


def test_exact_mode_never_runs_the_guard() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (np.sin(12 * X[:, 0]) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShotNoiseWarning)
        Qlassifier(random_state=0).fit(X, y)
