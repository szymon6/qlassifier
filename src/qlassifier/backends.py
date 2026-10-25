"""Where Bloch vectors come from: exact closed form, ideal shot sampling, or IBM fake devices (TDD §3.4).

- ``"exact"``: closed form (``encoding``), 0 circuits executed.
- ``"shots"``: Binomial(shots, (1 + r_k) / 2) per Pauli component, renormalised (exact law for product states).
- ``"fake_torino"`` / ``"fake_fez"`` / ``"fake_brisbane"``: per-qubit templates measured in X, Y and Z on
  ``SamplerV2.from_backend(AerSimulator.from_backend(Fake...()))`` (a plain ``SamplerV2()`` attaches no noise),
  transpiled once at optimisation level 3 with the Z circuit's layout pinned for X and Y (ADR 0006).

Qiskit is imported lazily here only, so ``import qlassifier`` and exact mode never load it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from qlassifier.errors import TooManyQubitsError

if TYPE_CHECKING:
    from qlassifier.encoding import Encoding

__all__ = [
    "DEVICES",
    "SETTINGS",
    "bloch_circuits",
    "circuit_stats",
    "device",
    "device_qubits",
    "local_noise",
    "logical_depth",
    "measure_bloch",
    "measure_bloch_per_qubit",
    "noise_errors",
    "per_qubit_circuits",
    "per_qubit_exact",
    "pinned_layout",
    "renormalise",
    "sample_shots",
    "templates",
    "transpile_pinned",
]

FloatArray = NDArray[np.float64]
DEVICES: tuple[str, ...] = ("fake_torino", "fake_fez", "fake_brisbane")
SETTINGS: tuple[str, ...] = ("X", "Y", "Z")
_BASIS_GATES = {"X": 1, "Y": 2, "Z": 0}  # H; S-dagger then H; nothing
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
Template = tuple[str, ...]


@cache
def device(name: str) -> Any:
    """The fake backend (``FakeTorino`` / ``FakeFez`` / ``FakeBrisbane``), created once per process."""
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez, FakeTorino

    return {"fake_torino": FakeTorino, "fake_fez": FakeFez, "fake_brisbane": FakeBrisbane}[name]()


@cache
def _simulator(name: str, method: str = "automatic") -> Any:
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(device(name), method=method)


def device_qubits(name: str) -> int:
    """Number of physical qubits of a fake device."""
    return int(device(name).num_qubits)


def check_qubits(n_qubits: int, name: str) -> None:
    """Raise ``TooManyQubitsError`` when the encoding needs more qubits than the device has."""
    m = device_qubits(name)
    if n_qubits > m:
        raise TooManyQubitsError(n_qubits, name, m)


def templates(enc: Encoding) -> tuple[Template, ...]:
    """Gate template per qubit: pair RX-RY (RX only for a lone column), circle RY, sphere RY-RZ."""
    out: list[Template] = []
    for q, pos in zip(enc.qubits, enc.positions, strict=True):
        if q.kind == "circle":
            out.append(("y",))
        elif q.kind == "sphere":
            out.append(("y", "z"))
        else:
            out.append(("x", "y") if len(pos) == 2 else ("x",))
    return tuple(out)


def logical_depth(tmpl: Sequence[Template]) -> int:
    """Depth of the deepest untranspiled setting circuit: template + basis change (S-dagger, H) + measure."""
    return max(len(t) for t in tmpl) + max(_BASIS_GATES.values()) + 1


def bloch_circuits(tmpl: Sequence[Template]) -> dict[str, Any]:
    """One parameterised circuit per Pauli setting (X, Y, Z): template on every qubit, basis change, measure all.

    Parameters are ``a[3q + k]`` (k = 0 RX angle, 1 RY angle, 2 RZ angle), matching ``Encoding.angles``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n = len(tmpl)
    p = ParameterVector("a", 3 * n)
    out: dict[str, Any] = {}
    for setting in SETTINGS:
        qc = QuantumCircuit(n)
        for q, gates in enumerate(tmpl):
            for axis in gates:
                getattr(qc, f"r{axis}")(p[3 * q + _AXIS_INDEX[axis]], q)
        for q in range(n):
            if setting == "X":
                qc.h(q)
            elif setting == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        out[setting] = qc
    return out


def transpile_pinned(circuits: Mapping[str, Any], name: str, seed_transpiler: int = 0) -> dict[str, Any]:
    """Transpile the Z circuit at optimisation level 3; pin its initial layout for the X and Y circuits."""
    from qiskit.transpiler import generate_preset_pass_manager

    backend = device(name)
    tz = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed_transpiler).run(
        circuits["Z"]
    )
    n = circuits["Z"].num_qubits
    layout = list(tz.layout.initial_index_layout(filter_ancillas=True)[:n])
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, initial_layout=layout, seed_transpiler=seed_transpiler
    )
    return {s: (tz if s == "Z" else pm.run(circuits[s])) for s in SETTINGS}


def circuit_stats(tc: Any) -> tuple[int, int]:
    """Two-qubit gate count and depth of a (transpiled) circuit; barriers and measurements are not gates."""
    two_q = sum(1 for inst in tc.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    return int(two_q), int(tc.depth())


def _values(tc: Any, flat: FloatArray) -> FloatArray:
    """Parameter values in the order of ``tc.parameters`` (columns of the flat (N, 3n) angle matrix)."""
    idx = [int(p.index) for p in tc.parameters]
    return flat[:, idx]


def _p1_from_counts(
    circuits: Sequence[Any],
    binds: Sequence[Mapping[Any, FloatArray]],
    name: str,
    shots: int,
    seed: int | None,
    threads: int | None,
    method: str,
) -> list[FloatArray]:
    """P(1) of every measured qubit for every parameter set, from counts (no per-shot memory).

    One ``AerSimulator.run`` call binds every parameter set of every circuit, exactly as ``SamplerV2`` does (same
    experiment order, so the same per-experiment seeds); counts are read instead of the per-shot memory strings
    that ``SamplerV2`` requests (4,000 rows x 3 settings x 10 qubits x 1,024 shots = 123 M strings, ~7 GB).
    Returns one (n_binds, n_qubits) array per circuit.
    """
    sim = _simulator(name, method)
    sim.set_options(max_parallel_threads=threads or 0)
    result = sim.run(
        list(circuits), shots=shots, seed_simulator=seed, parameter_binds=list(binds), memory=False
    ).result()
    out: list[FloatArray] = []
    i = 0
    for circuit, bind in zip(circuits, binds, strict=True):
        n = circuit.num_clbits
        n_binds = len(next(iter(bind.values()))) if bind else 1
        p1 = np.zeros((n_binds, n))
        for r in range(n_binds):
            for key, count in result.get_counts(i).items():
                bits = key.replace(" ", "")
                for q in range(n):
                    if bits[n - 1 - q] == "1":
                        p1[r, q] += count
            i += 1
        out.append(p1 / shots)
    return out


def _binds(tc: Any, flat: FloatArray) -> dict[Any, FloatArray]:
    """Parameter -> column of the flat (N, 3n) angle matrix, as ``SamplerV2`` binds them."""
    return {p: flat[:, int(p.index)] for p in tc.parameters}


def measure_bloch(
    transpiled: Mapping[str, Any],
    angles: FloatArray,
    name: str,
    shots: int,
    seed: int | None,
    threads: int | None = None,
) -> FloatArray:
    """Measure Bloch vectors of every row on a fake device: one experiment per row and setting, whole register.

    Returns unit-renormalised vectors, shape (N, n_qubits, 3); component = 1 - 2 P(1) per qubit.
    """
    N, n, _ = angles.shape
    flat = angles.reshape(N, 3 * n)
    circuits = [transpiled[s] for s in SETTINGS]
    p1 = _p1_from_counts(circuits, [_binds(tc, flat) for tc in circuits], name, shots, seed, threads, "automatic")
    return renormalise(np.stack([1.0 - 2.0 * p for p in p1], axis=-1).reshape(N, n, 3))


def pinned_layout(transpiled: Mapping[str, Any], n_qubits: int) -> list[int]:
    """Physical qubit of every logical qubit in the pinned layout of a ``transpile_pinned`` result."""
    return [int(v) for v in transpiled["Z"].layout.initial_index_layout(filter_ancillas=True)[:n_qubits]]


@cache
def noise_errors(name: str) -> tuple[dict[str, Any], ...]:
    """The error entries of the device noise model Aer builds for a fake backend (``NoiseModel.to_dict``)."""
    from qiskit_aer.noise import NoiseModel

    return tuple(NoiseModel.from_backend(device(name)).to_dict()["errors"])


def local_noise(errors: Sequence[Mapping[str, Any]]) -> bool:
    """True when the noise model acts qubit by qubit: single-qubit readout errors, no idle (delay) noise, no error of
    a single-qubit instruction that spans several qubits. These are the conditions under which simulating each qubit
    of a product circuit alone gives exactly the whole-register marginals (ADR 0006 amendment)."""
    one_qubit_ops = {"sx", "x", "id", "rz", "measure", "reset", "delay"}
    for e in errors:
        widths = {len(q) for q in e.get("gate_qubits", [])}
        if "delay" in e["operations"]:
            return False
        if e["type"] == "roerror" and widths != {1}:
            return False
        if e["type"] == "qerror" and one_qubit_ops & set(e["operations"]) and widths - {1}:
            return False
    return True


def per_qubit_exact(transpiled: Mapping[str, Any], name: str) -> bool:
    """Per-qubit simulation is exact for these circuits on this device: 0 two-qubit gates in every setting, no delay
    instruction, and local noise. Otherwise the whole register is simulated (``measure_bloch``)."""
    for tc in transpiled.values():
        if circuit_stats(tc)[0] != 0 or any(inst.operation.name == "delay" for inst in tc.data):
            return False
    return local_noise(noise_errors(name))


def per_qubit_circuits(tmpl: Sequence[Template], name: str, layout: Sequence[int]) -> list[dict[str, Any]]:
    """Each qubit's template + setting as a one-qubit circuit, transpiled onto its pinned physical qubit.

    The fake backends' noise models are local (per-qubit gate errors and readout errors, no crosstalk), and the
    readout uses per-qubit marginals only, so running each qubit alone samples exactly the same distribution as
    the whole register, at a cost linear instead of exponential in the qubit count (ADR 0006 amendment).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.transpiler import generate_preset_pass_manager

    p = ParameterVector("a", 3 * len(tmpl))  # same parameter names as the whole-register circuits
    out: list[dict[str, Any]] = []
    for q, gates in enumerate(tmpl):
        pm = generate_preset_pass_manager(
            optimization_level=3, backend=device(name), initial_layout=[int(layout[q])], seed_transpiler=0
        )
        circuits: dict[str, Any] = {}
        for setting in SETTINGS:
            qc = QuantumCircuit(1)
            for axis in gates:
                getattr(qc, f"r{axis}")(p[3 * q + _AXIS_INDEX[axis]], 0)
            if setting == "X":
                qc.h(0)
            elif setting == "Y":
                qc.sdg(0)
                qc.h(0)
            qc.measure_all()
            circuits[setting] = pm.run(qc)
        out.append(circuits)
    return out


def measure_bloch_per_qubit(
    circuits: Sequence[Mapping[str, Any]],
    angles: FloatArray,
    name: str,
    shots: int,
    seed: int | None,
    threads: int | None = None,
) -> FloatArray:
    """Bloch vectors of every row from one-qubit circuits (``per_qubit_circuits``): 3 experiments per qubit, one run."""
    N, n, _ = angles.shape
    flat = angles.reshape(N, 3 * n)
    ordered = [circuits[q][s] for q in range(n) for s in SETTINGS]
    p1 = _p1_from_counts(ordered, [_binds(tc, flat) for tc in ordered], name, shots, seed, threads, "density_matrix")
    R = np.empty((N, n, 3))
    for i, p in enumerate(p1):
        q, k = divmod(i, len(SETTINGS))
        R[:, q, k] = 1.0 - 2.0 * p[:, 0]
    return renormalise(R)


def sample_shots(R: FloatArray, shots: int, rng: np.random.Generator) -> FloatArray:
    """Ideal finite-shot estimate of exact Bloch vectors: Binomial per component, then unit renormalisation."""
    k = rng.binomial(shots, np.clip((1.0 + R) / 2.0, 0.0, 1.0))
    return renormalise(2.0 * k / shots - 1.0)


def renormalise(R: FloatArray, floor: float = 1e-3) -> FloatArray:
    """Scale every per-qubit vector to unit length (norm floored at ``floor``)."""
    out: FloatArray = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), floor)
    return out
