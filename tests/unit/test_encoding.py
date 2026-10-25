"""Encoding: closed forms vs Qiskit statevectors, geometry declarations and errors, classical twin (TDD §3.1-3.2)."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, Statevector

from qlassifier import GeometryError, Qlassifier, QubitMap, load_dataset
from qlassifier.backends import bloch_circuits, templates
from qlassifier.encoding import Encoding, bloch_from_angles, twin_features
from qlassifier.tuning import gamma0, gaussian_gram


def bloch_of(sv: Statevector, n: int) -> np.ndarray:
    r = np.zeros((n, 3))
    for q in range(n):
        for k, P in enumerate("XYZ"):
            label = ["I"] * n
            label[n - 1 - q] = P
            r[q, k] = np.real(sv.expectation_value(Pauli("".join(label))))
    return r


def statevector_bloch(enc: Encoding, X: np.ndarray) -> np.ndarray:
    """Bloch vectors from the very circuits the device modes run (Z setting, measurements removed)."""
    circ = bloch_circuits(templates(enc))["Z"].remove_final_measurements(inplace=False)
    A = enc.angles(X).reshape(len(X), -1)
    idx = [int(p.index) for p in circ.parameters]
    return np.array([bloch_of(Statevector(circ.assign_parameters(A[i, idx])), enc.n_qubits) for i in range(len(X))])


def test_pair_closed_form_matches_qiskit() -> None:
    X = load_dataset("heart", return_X_y=True)[0].to_numpy()[:8]  # 13 plain columns -> 7 qubits, last one alone
    enc = Encoding.plan(13, None, None).fit(X)
    assert enc.n_qubits == 7
    assert enc.kinds == ("pair",) * 7
    assert np.abs(enc.bloch(X) - statevector_bloch(enc, X)).max() <= 1e-12


def test_circle_closed_form_matches_qiskit() -> None:
    X = load_dataset("protein_angles", return_X_y=True)[0].to_numpy()[:6]
    geometry = dict.fromkeys(range(10), 360.0)
    enc = Encoding.plan(10, geometry, None).fit(X)
    R = enc.bloch(X)
    assert np.abs(R - statevector_bloch(enc, X)).max() <= 1e-12
    t = 2 * np.pi * X[:, 0] / 360.0
    assert np.allclose(R[:, 0], np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], 1), atol=1e-15)


def test_sphere_and_mixed_closed_form_matches_qiskit() -> None:
    rng = np.random.default_rng(3)  # mechanism test: (lat, lon) directions, a 24 h clock and two plain columns
    X = np.c_[rng.uniform(-90, 90, 5), rng.uniform(-180, 180, 5), rng.uniform(0, 24, 5), rng.normal(size=(5, 2))]
    enc = Encoding.plan(5, {2: 24.0, (0, 1): "sphere"}, None).fit(X)
    assert [q.kind for q in enc.qubits] == ["circle", "sphere", "pair"]
    R = enc.bloch(X)
    assert np.abs(R - statevector_bloch(enc, X)).max() <= 1e-12
    lat, lon = np.deg2rad(X[:, 0]), np.deg2rad(X[:, 1])
    direction = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    assert np.allclose(R[:, 1], direction, atol=1e-12)


def test_unified_closed_form_reduces_to_each_kind() -> None:
    rng = np.random.default_rng(1)
    a, b, f = rng.uniform(-3, 3, (3, 50))
    pair = bloch_from_angles(np.stack([a, b, np.zeros(50)], -1))
    assert np.allclose(pair, np.stack([np.cos(a) * np.sin(b), -np.sin(a), np.cos(a) * np.cos(b)], -1))
    sphere = bloch_from_angles(np.stack([np.zeros(50), b, f], -1))
    assert np.allclose(sphere, np.stack([np.sin(b) * np.cos(f), np.sin(b) * np.sin(f), np.cos(b)], -1))
    assert np.allclose(np.linalg.norm(bloch_from_angles(np.stack([a, b, f], -1)), axis=-1), 1.0)


def test_core_scale_and_zscore() -> None:
    X = load_dataset("sonar", return_X_y=True)[0].to_numpy()
    enc = Encoding.plan(60, None, None).fit(X)
    assert enc.scale == pytest.approx(2 / np.sqrt(60))
    Z = enc.zscores(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-12)
    assert np.allclose(Z.std(0), 1)
    A = enc.angles(X)
    assert np.allclose(A[:, 0, 0], enc.scale * Z[:, 0])
    assert np.allclose(A[:, 0, 1], enc.scale * Z[:, 1])


def test_constant_plain_column_keeps_unit_sd() -> None:
    X = np.c_[np.ones(30), np.arange(30.0)]
    enc = Encoding.plan(2, None, None).fit(X)
    assert np.all(np.isfinite(enc.bloch(X)))


def test_qubit_maps_use_names_for_dataframes() -> None:
    data = load_dataset("protein_angles")
    clf = Qlassifier(geometry={"phi_0": 360, "psi_0": 360}, gamma=1.0, C=1.0).fit(data.X.iloc[:300], data.y.iloc[:300])
    assert clf.encoding_[0] == QubitMap("circle", ("phi_0",), 360.0)
    assert clf.encoding_[1] == QubitMap("circle", ("psi_0",), 360.0)
    assert clf.encoding_[2] == QubitMap("pair", ("phi_m2", "psi_m2"))
    assert clf.n_qubits_ == 2 + 4


def test_twin_gram_equals_geometry_gram() -> None:
    X = load_dataset("protein_angles", return_X_y=True)[0].to_numpy()[:200]
    enc = Encoding.plan(10, dict.fromkeys(range(10), 360.0), None).fit(X)
    F = enc.bloch(X)
    T = twin_features(X, enc)
    Kg = gaussian_gram(F, F, 0.5 * gamma0(F))
    Kt = gaussian_gram(T, T, 0.5 / T.var(axis=0).sum())
    assert np.abs(Kg - Kt).max() <= 1e-12


def test_twin_features_include_spheres_and_plain() -> None:
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(-90, 90, 9), rng.uniform(-180, 180, 9), rng.normal(size=9)]
    enc = Encoding.plan(3, {(0, 1): "sphere"}, None).fit(X)
    T = twin_features(X, enc)
    assert T.shape == (9, 4)
    assert np.allclose(np.linalg.norm(T[:, :3], axis=1), 1)


NAMES = ["lat", "lon", "hour", "x"]


@pytest.mark.parametrize(
    ("geometry", "names", "message", "hint"),
    [
        ({"phi2": 360}, NAMES, "Geometry names column 'phi2', but X has no such column.",
         "Use one of: lat, lon, … or integer positions for arrays."),
        ({"hour": 24}, None, "Geometry names column 'hour', but X has no column names.",
         "Pass a pandas DataFrame or use integer positions."),
        ({7: 24}, NAMES, "Geometry names column 7, but X has only 4 columns.", "Use positions 0 to 3."),
        ({"hour": 0}, NAMES, "Geometry period for column 'hour' must be a positive number; got 0.",
         "Use the column's full cycle, e.g. 360 for degrees, 24 for hours."),
        ({"hour": float("inf")}, NAMES, "Geometry period for column 'hour' must be a positive number; got inf.",
         "Use the column's full cycle, e.g. 360 for degrees, 24 for hours."),
        ({"hour": "daily"}, NAMES, "Geometry period for column 'hour' must be a positive number; got 'daily'.",
         "Use the column's full cycle, e.g. 360 for degrees, 24 for hours."),
        ({("lat", "lon"): 360}, NAMES, "Geometry maps the pair ('lat', 'lon') to 360.",
         "Pairs mean (latitude, longitude): use ('lat', 'lon'): 'sphere'; declare circles one column at a time."),
        ({"lat": "sphere"}, NAMES, "Geometry maps column 'lat' to 'sphere'.", "Write {('lat', 'lon'): 'sphere'}."),
        ({("lat", "lon"): "sphere", "lat": 360}, NAMES, "Column 'lat' appears twice in geometry.",
         "Each column sits on one qubit only."),
        ({("lat", "lon", "x"): "sphere"}, NAMES,
         "Geometry key ('lat', 'lon', 'x') is not a column or a (latitude, longitude) pair.",
         "Use a column name, a position, or ('lat', 'lon'): 'sphere'."),
        ({True: 24}, NAMES, "Geometry key True is not a column or a (latitude, longitude) pair.",
         "Use a column name, a position, or ('lat', 'lon'): 'sphere'."),
    ],
)  # fmt: skip
def test_geometry_errors(geometry: dict, names: list[str] | None, message: str, hint: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(GeometryError) as info:
        Encoding.plan(4, geometry, names)
    assert info.value.code == "bad_geometry"
    assert info.value.status == 422
    assert info.value.message == message
    assert info.value.hint == hint


def test_geometry_must_be_a_mapping() -> None:
    with pytest.raises(GeometryError, match="geometry must be a dict"):
        Encoding.plan(4, ["hour"], NAMES)  # type: ignore[arg-type]


def test_latitude_outside_range_fails_at_fit() -> None:
    X = np.c_[np.array([10.0, 95.0, -20.0]), np.zeros(3)]
    plan = Encoding.plan(2, {(0, 1): "sphere"}, None)
    with pytest.raises(GeometryError) as info:
        plan.fit(X)
    assert info.value.message == "Latitude column 0 has values outside [-90, 90] (e.g. 95)."
    assert info.value.hint == "Use degrees, latitude first."


def test_zscores_need_fit() -> None:
    with pytest.raises(RuntimeError):
        Encoding.plan(2, None, None).zscores(np.zeros((2, 2)))


def test_geometry_only_plan_has_no_plain_columns() -> None:
    enc = Encoding.plan(2, {0: 360.0, 1: 360.0}, None).fit(np.zeros((3, 2)))
    assert enc.plain == ()
    assert enc.bloch(np.zeros((3, 2))).shape == (3, 2, 3)
