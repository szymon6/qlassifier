"""Backends: shot sampling law, renormalisation, circuit templates, pinned transpilation, circuit stats (§3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from qlassifier import backends, load_dataset
from qlassifier.encoding import Encoding


@pytest.fixture(scope="module")
def heart_enc() -> tuple[Encoding, np.ndarray]:
    X = load_dataset("heart", return_X_y=True)[0].to_numpy()
    return Encoding.plan(13, None, None).fit(X), X


def test_sample_shots_follows_the_binomial_law(heart_enc: tuple[Encoding, np.ndarray]) -> None:
    enc, X = heart_enc
    R = enc.bloch(X)
    noisy = backends.sample_shots(R, 1024, np.random.default_rng(0))
    assert noisy.shape == R.shape
    assert np.allclose(np.linalg.norm(noisy, axis=-1), 1.0)
    err = noisy - R
    assert 0.01 < float(np.abs(err).mean()) < 0.05  # ~ 1/sqrt(1024) per component
    big = backends.sample_shots(R, 2**20, np.random.default_rng(0))
    assert float(np.abs(big - R).max()) < 0.01
    again = backends.sample_shots(R, 1024, np.random.default_rng(0))
    assert np.array_equal(noisy, again)


def test_renormalise_floors_tiny_vectors() -> None:
    R = np.array([[[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]]])
    out = backends.renormalise(R)
    assert np.allclose(out[0, 1], [0.6, 0.8, 0.0])
    assert np.all(np.isfinite(out))


def test_templates_and_logical_depth() -> None:
    enc = Encoding.plan(5, {0: 24.0, (1, 2): "sphere"}, None)
    tmpl = backends.templates(enc)
    assert tmpl == (("y",), ("y", "z"), ("x", "y"))
    assert backends.logical_depth(tmpl) == 2 + 2 + 1
    odd = backends.templates(Encoding.plan(3, None, None))
    assert odd == (("x", "y"), ("x",))


def test_bloch_circuits_per_setting() -> None:
    tmpl = (("x", "y"), ("y",))
    circs = backends.bloch_circuits(tmpl)
    assert list(circs) == ["X", "Y", "Z"]
    ops = {s: dict(c.count_ops()) for s, c in circs.items()}
    assert ops["Z"].get("h", 0) == 0
    assert ops["X"]["h"] == 2
    assert ops["Y"]["sdg"] == 2
    assert all(o["measure"] == 2 for o in ops.values())
    assert sorted(int(p.index) for p in circs["Z"].parameters) == [0, 1, 4]  # a[3q + k]: RX q0, RY q0, RY q1
    for c in circs.values():
        assert backends.circuit_stats(c)[0] == 0  # the measure_all barrier is not a gate


@pytest.mark.parametrize("device", backends.DEVICES)
def test_transpile_pins_one_layout_and_keeps_zero_two_qubit_gates(device: str) -> None:
    circs = backends.bloch_circuits((("x", "y"),) * 4)
    tcs = backends.transpile_pinned(circs, device)
    layouts = [list(tc.layout.final_index_layout(filter_ancillas=True)[:4]) for tc in tcs.values()]
    assert layouts[0] == layouts[1] == layouts[2]
    for tc in tcs.values():
        two_q, depth = backends.circuit_stats(tc)
        assert two_q == 0
        assert 3 <= depth <= 12


def test_device_qubits_and_check() -> None:
    assert backends.device_qubits("fake_torino") == 133
    assert backends.device_qubits("fake_fez") == 156
    assert backends.device_qubits("fake_brisbane") == 127
    backends.check_qubits(127, "fake_brisbane")
    from qlassifier import TooManyQubitsError

    with pytest.raises(TooManyQubitsError):
        backends.check_qubits(128, "fake_brisbane")


def test_circuit_stats_counts_entangling_gates() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.barrier()
    qc.measure_all()
    two_q, depth = backends.circuit_stats(qc)
    assert two_q == 2
    assert depth == qc.depth()


def test_measure_bloch_on_a_device_tracks_exact_vectors(heart_enc: tuple[Encoding, np.ndarray]) -> None:
    enc, X = heart_enc
    rows = X[:12]
    tcs = backends.transpile_pinned(backends.bloch_circuits(backends.templates(enc)), "fake_fez")
    R = backends.measure_bloch(tcs, enc.angles(rows), "fake_fez", 2048, seed=11, threads=1)
    assert R.shape == (12, enc.n_qubits, 3)
    assert np.allclose(np.linalg.norm(R, axis=-1), 1.0)
    assert float(np.abs(R - enc.bloch(rows)).mean()) < 0.06
    again = backends.measure_bloch(tcs, enc.angles(rows), "fake_fez", 2048, seed=11, threads=1)
    assert np.array_equal(R, again)
