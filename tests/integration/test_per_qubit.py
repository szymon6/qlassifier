"""Qubit-by-qubit device simulation (ADR 0006 amendment, e14): exactness, its guard, and the whole-register fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qlassifier import Qlassifier, backends, load_dataset
from qlassifier.bench.protocol import Prep, arrays
from qlassifier.encoding import Encoding

DEVICES = list(backends.DEVICES)


@pytest.fixture(scope="module")
def bc_rows() -> tuple[Encoding, np.ndarray]:
    """Breast Cancer rows on 4 qubits (PCA-8), as the benchmark encodes them."""
    a = arrays("breast_cancer")
    X = Prep("pca8").fit(a.X).transform(a.X)
    return Encoding.plan(8, None, None).fit(X), X[:3]


@pytest.mark.parametrize("name", DEVICES)
def test_fake_backend_noise_models_are_local(name: str) -> None:
    errors = backends.noise_errors(name)
    assert backends.local_noise(errors)
    assert any(e["type"] == "roerror" for e in errors)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "roerror", "operations": ["measure"], "gate_qubits": [[0, 1]]},  # correlated readout
        {"type": "qerror", "operations": ["delay"], "gate_qubits": [[0]]},  # idle noise
        {"type": "qerror", "operations": ["sx"], "gate_qubits": [[0, 1]]},  # crosstalk on a single-qubit gate
    ],
)
def test_non_local_noise_is_detected(bad: dict[str, Any]) -> None:
    errors = [*backends.noise_errors("fake_fez")[:5], bad]
    assert not backends.local_noise(errors)


def test_exactness_guard_rejects_entangling_or_delayed_circuits() -> None:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backends.device("fake_fez"), seed_transpiler=0)
    assert not backends.per_qubit_exact({"Z": pm.run(qc)}, "fake_fez")
    delayed = QuantumCircuit(1)
    delayed.delay(100, 0)
    delayed.measure_all()
    assert not backends.per_qubit_exact({"Z": delayed}, "fake_fez")
    product = backends.transpile_pinned(backends.bloch_circuits((("x", "y"), ("y",))), "fake_fez")
    assert backends.per_qubit_exact(product, "fake_fez")


@pytest.mark.parametrize("name", DEVICES)
def test_exact_marginals_equal_the_whole_register(name: str, bc_rows: tuple[Encoding, np.ndarray]) -> None:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    enc, X = bc_rows
    tmpl = backends.templates(enc)
    flat = enc.angles(X).reshape(len(X), -1)
    whole = backends.transpile_pinned(backends.bloch_circuits(tmpl), name)
    layout = backends.pinned_layout(whole, enc.n_qubits)
    single = backends.per_qubit_circuits(tmpl, name, layout)
    sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(backends.device(name)))

    def prepared(circuit: Any, row: np.ndarray, qubits: list[int]) -> Any:
        bound = circuit.assign_parameters(backends._values(circuit, row[None, :])[0])
        bound = bound.remove_final_measurements(inplace=False)
        bound.save_density_matrix(qubits=qubits)
        return bound

    def marginals(density: Any, n: int) -> np.ndarray:
        diag = np.real(np.diag(np.asarray(density)))
        idx = np.arange(len(diag))
        return np.array([diag[(idx >> k) & 1 == 1].sum() for k in range(n)]) / diag.sum()

    jobs: list[tuple[str, int, int | None]] = []  # (setting, row, qubit or None for the whole register)
    circuits = []
    for setting in backends.SETTINGS:
        for r, row in enumerate(flat):
            jobs.append((setting, r, None))
            circuits.append(prepared(whole[setting], row, layout))
            for q, phys in enumerate(layout):
                jobs.append((setting, r, q))
                circuits.append(prepared(single[q][setting], row, [phys]))
    result = sim.run(circuits).result()  # one Aer call for every density matrix
    whole_p: dict[tuple[str, int], np.ndarray] = {}
    worst = 0.0
    for i, (setting, r, q) in enumerate(jobs):
        density = result.data(i)["density_matrix"]
        if q is None:
            whole_p[(setting, r)] = marginals(density, len(layout))
        else:
            worst = max(worst, abs(whole_p[(setting, r)][q] - marginals(density, 1)[0]))
    assert worst <= 1e-12


@pytest.mark.parametrize("name", DEVICES)
def test_counts_path_equals_the_sampler_path(name: str, bc_rows: tuple[Encoding, np.ndarray]) -> None:
    """P(1) read from counts in one AerSimulator.run equals SamplerV2's per-shot memory path bit for bit.

    The UCI noise studies were measured through SamplerV2 (memory=True, ~60 bytes per shot and row); the counts
    path keeps the same experiment order and seeds and needs no per-shot strings (ADR 0006 amendment).
    """
    from qiskit_aer.primitives import SamplerV2

    enc, X = bc_rows
    tmpl = backends.templates(enc)
    flat = enc.angles(X).reshape(len(X), -1)
    whole = backends.transpile_pinned(backends.bloch_circuits(tmpl), name)
    single = backends.per_qubit_circuits(tmpl, name, backends.pinned_layout(whole, enc.n_qubits))
    for circuits, method, seed in (
        ([single[q][s] for q in range(enc.n_qubits) for s in backends.SETTINGS], "density_matrix", 11),
        ([whole[s] for s in backends.SETTINGS], "automatic", 5),
    ):
        sim = backends._simulator(name, method)
        sim.set_options(max_parallel_threads=1)
        pubs = [(tc, backends._values(tc, flat)) for tc in circuits]
        res = SamplerV2.from_backend(sim, seed=seed).run(pubs, shots=512).result()
        reference = [
            np.stack(
                [
                    np.asarray(res[i].data.meas.slice_bits([q]).bitcount(), float).mean(axis=-1)
                    for q in range(tc.num_clbits)
                ],
                axis=-1,
            )
            for i, tc in enumerate(circuits)
        ]
        counts = backends._p1_from_counts(
            circuits, [backends._binds(tc, flat) for tc in circuits], name, 512, seed, 1, method
        )
        for a, b in zip(reference, counts, strict=True):
            assert np.array_equal(a, b)


def test_shots_track_exact_vectors(bc_rows: tuple[Encoding, np.ndarray]) -> None:
    enc, X = bc_rows
    tmpl = backends.templates(enc)
    whole = backends.transpile_pinned(backends.bloch_circuits(tmpl), "fake_torino")
    single = backends.per_qubit_circuits(tmpl, "fake_torino", backends.pinned_layout(whole, enc.n_qubits))
    R = backends.measure_bloch_per_qubit(single, enc.angles(X), "fake_torino", 8192, seed=3, threads=1)
    assert R.shape == (3, 4, 3)
    assert float(np.abs(R - enc.bloch(X)).max()) < 0.06
    again = backends.measure_bloch_per_qubit(single, enc.angles(X), "fake_torino", 8192, seed=3, threads=1)
    assert np.array_equal(R, again)


def test_estimator_falls_back_to_the_whole_register(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = load_dataset("heart", return_X_y=True)
    local = Qlassifier(backend="fake_fez", gamma=0.25, C=4.0, random_state=0).fit(X.iloc[:40], y.iloc[:40])
    assert local._per_qubit is not None
    assert len(local._per_qubit) == local.n_qubits_
    correlated = [
        *backends.noise_errors("fake_fez"),
        {"type": "roerror", "operations": ["measure"], "gate_qubits": [[0, 1]]},
    ]
    monkeypatch.setattr(backends, "noise_errors", lambda name: tuple(correlated))
    clf = Qlassifier(backend="fake_fez", gamma=0.25, C=4.0, random_state=0).fit(X.iloc[:40], y.iloc[:40])
    assert clf._per_qubit is None  # not provably exact: simulate the whole register
    assert clf.predict(X.iloc[40:50]).shape == (10,)
    assert clf.resources_.two_qubit_gates == 0
