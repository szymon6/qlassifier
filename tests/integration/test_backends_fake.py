"""Device modes end to end (TDD §3.4, US-3): IBM fake backends on real rows, resources, pinned layout, determinism."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qlassifier import Qlassifier, ShotNoiseWarning, load_dataset

DEVICES = ["fake_torino", "fake_fez", "fake_brisbane"]


@pytest.fixture(scope="module")
def heart() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_dataset("heart", return_X_y=True)
    return X.to_numpy(), y.to_numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_fake_device_fit_predict_and_resources(device: str, heart: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = heart
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShotNoiseWarning)
        clf = Qlassifier(backend=device, random_state=0).fit(X[:40], y[:40])
    pred = clf.predict(X[40:60])
    assert pred.shape == (20,)
    assert clf.score(X[40:100], y[40:100]) > 0.6
    r = clf.resources_
    assert r.backend == device
    assert r.qubits == 7
    assert r.two_qubit_gates == 0
    assert 3 <= r.depth <= 12
    assert r.circuits_per_sample == 3
    assert r.circuits_fit == 120
    assert r.circuits_predict == 3 * (20 + 60)
    assert r.tuning_circuits == 0
    assert r.shots == 1024
    layouts = {
        tuple(tc.layout.final_index_layout(filter_ancillas=True)[:7]) for tc in clf._transpiled.values()
    }  # one set of physical qubits for the X, Y and Z settings
    assert len(layouts) == 1
    assert np.allclose(np.linalg.norm(clf._F, axis=-1), 1.0)


def test_fake_device_runs_repeat_with_random_state(heart: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = heart
    kw = {"backend": "fake_fez", "gamma": 0.25, "C": 4.0, "random_state": 7}
    a = Qlassifier(**kw).fit(X[:40], y[:40])
    b = Qlassifier(**kw).fit(X[:40], y[:40])
    assert np.array_equal(a._F, b._F)
    assert np.array_equal(a.decision_function(X[40:70]), b.decision_function(X[40:70]))
    assert np.array_equal(a.predict(X[40:70]), a.predict(X[40:70]))  # same predict stream every call
    c = Qlassifier(**{**kw, "random_state": 8}).fit(X[:40], y[:40])
    assert not np.array_equal(a._F, c._F)


def test_device_noise_costs_little_on_breast_cancer(breast_cancer_split: tuple) -> None:  # type: ignore[type-arg]
    X_tr, X_te, y_tr, y_te = breast_cancer_split
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    exact = make_pipeline(StandardScaler(), PCA(8, random_state=0), Qlassifier(gamma=1.0, C=1.0)).fit(X_tr, y_tr)
    noisy = make_pipeline(
        StandardScaler(), PCA(8, random_state=0), Qlassifier(backend="fake_torino", gamma=1.0, C=1.0, random_state=0)
    ).fit(X_tr, y_tr)
    assert noisy[-1].resources_.qubits == 4
    assert abs(exact.score(X_te, y_te) - noisy.score(X_te, y_te)) <= 0.02
