"""Shared test setup (TDD §9.1): offline guard, real-data fixtures, paths.

- Session-wide socket guard: any connection to a non-loopback address fails the test (NFR-2, offline).
- Fixtures use the bundled datasets, the e01 golden rows (``tests/fixtures/e01_rows.csv``) and shipped results;
  synthetic data appears only in mechanism tests (error triggers, the narrow-kernel shot-noise warning).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _loopback(address: Any) -> bool:
    if isinstance(address, str | bytes):  # AF_UNIX path
        return True
    host = address[0] if isinstance(address, tuple) else address
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True, scope="session")
def _offline() -> Iterator[None]:
    """Fail any attempt to reach the network: only loopback and local sockets are allowed."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if not _loopback(address):
            raise RuntimeError(f"network access blocked in tests: {address!r}")
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if not _loopback(address):
            raise RuntimeError(f"network access blocked in tests: {address!r}")
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    yield
    socket.socket.connect = original_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


@pytest.fixture(scope="session")
def breast_cancer_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Protocol split (seed 0) of the bundled Breast Cancer data."""
    from qlassifier import load_dataset

    return load_dataset("breast_cancer").split(seed=0)


@pytest.fixture(scope="session")
def heart_small() -> tuple[np.ndarray, np.ndarray]:
    """First 60 rows of Heart (real data, both classes), as arrays."""
    from qlassifier import load_dataset

    X, y = load_dataset("heart", return_X_y=True)
    return X.to_numpy()[:60], y.to_numpy()[:60]


@pytest.fixture(scope="session")
def e01_rows() -> pd.DataFrame:
    """Golden rows of the phase-7 reference leaderboard (e01): heart seed 0, Breast Cancer seed 0."""
    names = {"logreg": "logistic_regression", "grad_boost": "gradient_boosting", "fqk_bw_full": "fqk_bandwidth_full",
             "fqk_bw_linear": "fqk_bandwidth_linear", "bloch_gauss": "qlassifier"}  # fmt: skip
    df = pd.read_csv(FIXTURES / "e01_rows.csv")
    df["model"] = df["model"].map(lambda m: names.get(m, m))
    return df


SMALL_MODELS = ["trivial", "logistic_regression", "zz_qsvc", "fqk_bandwidth_linear", "qlassifier"]


@pytest.fixture(scope="session")
def heart_result() -> Any:
    """A real benchmark result: Heart, seeds 0-1, 5 models, noise part A (shots + 3 IBM fake devices) and cost."""
    from qlassifier import benchmark

    return benchmark("heart", seeds=[0, 1], models=SMALL_MODELS, noise=True, n_jobs=2)


@pytest.fixture(scope="session")
def bt10_result() -> Any:
    """The saved phase-9 BT-10 run: Breast Cancer, 12 models x 10 seeds, cost study, no noise study."""
    from qlassifier.bench.result import BenchmarkResult

    return BenchmarkResult.from_json(ROOT / "benchmarks" / "bt10" / "breast_cancer" / "result.json")
