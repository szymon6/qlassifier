"""Exact baseline kernels vs Qiskit statevectors (port of phase-7 e00) and the device circuits behind them (§7.4)."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Pauli, Statevector

from qlassifier.bench.cost import pqk_circuits
from qlassifier.bench.models import (
    fidelity_from_states,
    heisenberg_pqk_features,
    local_paulis,
    model_specs,
    pqk_start,
    separable_kernel,
    zz_states,
)
from qlassifier.bench.noise import _compute_uncompute


def bloch_of(sv: Statevector, n: int) -> np.ndarray:
    r = np.zeros((n, 3))
    for q in range(n):
        for k, P in enumerate("XYZ"):
            label = ["I"] * n
            label[n - 1 - q] = P
            r[q, k] = np.real(sv.expectation_value(Pauli("".join(label))))
    return r


@pytest.mark.parametrize("entanglement", ["full", "linear"])
def test_zz_states_match_qiskit_feature_map(entanglement: str) -> None:
    X = np.random.default_rng(7).uniform(0, np.pi, size=(5, 5))  # mechanism test: arbitrary angles
    S = zz_states(X, 2, entanglement)
    fm = zz_feature_map(5, reps=2, entanglement=entanglement)
    Sq = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
    assert np.abs(fidelity_from_states(S, S) - fidelity_from_states(Sq, Sq)).max() <= 1e-12
    assert np.min(np.abs(np.sum(S.conj() * Sq, axis=1))) > 1 - 1e-12  # same states up to a global phase


def test_pqk_features_match_a_heisenberg_circuit() -> None:
    d, t, T = 4, 0.9, 3
    X = np.random.default_rng(7).normal(size=(3, d))
    F = heisenberg_pqk_features(X, t, T).reshape(3, d + 1, 3)
    v = pqk_start(d + 1)
    for i, x in enumerate(X):
        qc = QuantumCircuit(d + 1)
        for q in range(d + 1):
            qc.initialize(v[q], q)
        for _ in range(T):
            for j in range(d):
                th = (t / T) * x[j]
                qc.rxx(2 * th, j, j + 1)
                qc.ryy(2 * th, j, j + 1)
                qc.rzz(2 * th, j, j + 1)
        assert np.abs(bloch_of(Statevector(qc), d + 1) - F[i]).max() <= 1e-12


@pytest.mark.parametrize("setting", ["X", "Y", "Z"])
def test_pqk_device_circuits_reproduce_the_exact_features(setting: str) -> None:
    d = 3
    X = np.random.default_rng(2).normal(size=(4, d))
    F = heisenberg_pqk_features(X).reshape(4, d + 1, 3)
    p = ParameterVector("x", d)
    circ = pqk_circuits(d, setting, p).remove_final_measurements(inplace=False)
    k = "XYZ".index(setting)
    for i, x in enumerate(X):
        sv = Statevector(circ.assign_parameters(x))
        z = bloch_of(sv, d + 1)[:, 2]  # after the basis change every setting is read in Z
        assert np.abs(z - F[i, :, k]).max() <= 1e-9


def test_separable_kernel_per_feature_and_product() -> None:
    xs = np.random.default_rng(7).uniform(-2, 2, size=5)
    svs = []
    for x in xs:
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 4, 0)
        qc.ry(x, 0)
        svs.append(Statevector(qc))
    Kq = np.array([[abs(a.inner(b)) ** 2 for b in svs] for a in svs])
    assert np.abs(separable_kernel(xs[:, None], xs[:, None]) - Kq).max() <= 1e-12
    P = np.random.default_rng(1).uniform(-1.5, 1.5, size=(4, 3))
    K = separable_kernel(P, P)
    ref = np.prod([separable_kernel(P[:, [j]], P[:, [j]]) for j in range(3)], axis=0)
    assert np.allclose(K, ref)


@pytest.mark.parametrize("model", ["fqk_bandwidth_linear", "separable"])
def test_compute_uncompute_circuits_give_the_exact_kernel(model: str) -> None:
    d = 4
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-1, 1, d), rng.uniform(-1, 1, d)
    circ = _compute_uncompute(model, d).remove_final_measurements(inplace=False)
    values = {p: (a if p.vector.name == "a" else b)[p.index] for p in circ.parameters}
    p0 = abs(Statevector(circ.assign_parameters(values)).data[0]) ** 2
    if model == "separable":
        exact = separable_kernel(a[None, :], b[None, :])[0, 0]
    else:
        exact = fidelity_from_states(zz_states(a[None, :], 2, "linear"), zz_states(b[None, :], 2, "linear"))[0, 0]
    assert p0 == pytest.approx(exact, abs=1e-12)


def test_local_paulis_of_product_states() -> None:
    psi = np.kron([1, 0], [1, 1]) / np.sqrt(2)  # little-endian: qubit 0 = |+>, qubit 1 = |0>
    R = local_paulis(psi[None, :].astype(complex), 2)
    assert np.allclose(R[0], [[1, 0, 0], [0, 0, 1]])


def test_registry_shapes_and_grids() -> None:
    specs = {s.id: s for s in model_specs(8)}
    assert list(specs) == [
        "trivial", "logistic_regression", "rbf_svm", "random_forest", "gradient_boosting", "mlp", "zz_qsvc",
        "fqk_bandwidth_full", "fqk_bandwidth_linear", "pqk", "separable", "qlassifier",
    ]  # fmt: skip
    assert {k: len(s.grid) for k, s in specs.items()} == {
        "trivial": 1, "logistic_regression": 30, "rbf_svm": 30, "random_forest": 30, "gradient_boosting": 30,
        "mlp": 30, "zz_qsvc": 4, "fqk_bandwidth_full": 30, "fqk_bandwidth_linear": 30, "pqk": 30, "separable": 30,
        "qlassifier": 30,
    }  # fmt: skip
    assert specs["rbf_svm"].grid[0]["bw"] == pytest.approx(1 / 8 * (1 / 8) ** 2)  # gamma = m^2 / d
    c_full = 1 / np.sqrt(8 + 2 * np.pi**2 * 28)
    assert specs["fqk_bandwidth_full"].grid[5]["bw"] == pytest.approx(c_full / 4)  # c = m c_zz, second m
    assert [s.role for s in specs.values()].count("published_quantum") == 4
    geo = {s.id: s for s in model_specs(10, dict.fromkeys(range(10), 360.0))}
    assert "classical_twin" in geo
    assert geo["qlassifier"].raw
    assert geo["classical_twin"].family == "control"


def test_twin_kernel_equals_geometry_gram() -> None:
    from qlassifier import load_dataset
    from qlassifier.encoding import Encoding
    from qlassifier.tuning import gamma0, gaussian_gram

    X = load_dataset("protein_angles", return_X_y=True)[0].to_numpy()[:150]
    geometry = dict.fromkeys(range(10), 360.0)
    twin = next(s for s in model_specs(10, geometry) if s.id == "classical_twin")
    assert twin.kernel is not None
    Kt, _ = twin.kernel(X, X[:5], 4.0, {})
    F = Encoding.plan(10, geometry, None).fit(X).bloch(X)
    assert np.abs(Kt - gaussian_gram(F, F, 4.0 * gamma0(F))).max() <= 1e-12
