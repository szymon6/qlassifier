"""Cost study: transpiled qubits, two-qubit gates and depth per device; circuit counts (TDD §7.9, BT-7, BT-8).

One representative input per quantum model (``numpy`` seed 0), transpiled to FakeTorino, FakeFez and FakeBrisbane
at optimisation level 3 with transpiler seeds 0-2 (median); per-sample readouts report their deepest setting.
Circuit counts use the first seed's split: Qlassifier and PQK 3 (N_tr + N_te); fidelity kernels
N_tr (N_tr - 1) / 2 + N_te N_tr; on-device tuning only for models whose hyperparameters change circuits
(FQK-bw bandwidths, separable scales). Phase-7 reference: e06.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from qlassifier import backends
from qlassifier.bench.models import PQK_T, PQK_TROTTER, pqk_start
from qlassifier.bench.protocol import arrays, fold_inputs
from qlassifier.bench.result import CostModel, CostStudy
from qlassifier.datasets import SplitMode
from qlassifier.encoding import Encoding, Geometry
from qlassifier.tuning import M_GRID

__all__ = ["QUANTUM_MODELS", "cost_study", "fqk_pipeline_ratio", "representative_circuits"]

QUANTUM_MODELS: tuple[str, ...] = (
    "zz_qsvc",
    "fqk_bandwidth_full",
    "fqk_bandwidth_linear",
    "pqk",
    "separable",
    "qlassifier",
)
TRANSPILER_SEEDS = (0, 1, 2)
_ON_DEVICE_TUNING = {"fqk_bandwidth_full": len(M_GRID), "fqk_bandwidth_linear": len(M_GRID), "separable": len(M_GRID)}


def _fidelity_pairs(n_a: int, n_b: int) -> int:
    return n_a * (n_a - 1) // 2 + n_b * n_a


def representative_circuits(
    model: str, d: int, geometry: dict[int, float] | None, rng: np.random.Generator
) -> list[Any]:
    """Measured circuits of one input (``numpy`` seed 0): 3 Pauli settings for per-sample readouts, else one."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import zz_feature_map

    if model == "qlassifier":
        enc = Encoding.plan(d, cast(Geometry | None, geometry), None)
        circs = backends.bloch_circuits(backends.templates(enc))
        values = rng.uniform(-np.pi, np.pi, size=len(circs["Z"].parameters))
        return [circs[s].assign_parameters(values) for s in backends.SETTINGS]
    if model in ("zz_qsvc", "fqk_bandwidth_full", "fqk_bandwidth_linear"):
        fm = zz_feature_map(d, reps=2, entanglement="linear" if model.endswith("linear") else "full")
        a, b = rng.uniform(0, np.pi, d), rng.uniform(0, np.pi, d)
        qc = fm.assign_parameters(a).compose(fm.assign_parameters(b).inverse())
        qc.measure_all()
        return [qc]
    if model == "separable":
        qc = QuantumCircuit(d)
        a, b = rng.uniform(-1.5, 1.5, d), rng.uniform(-1.5, 1.5, d)
        for q in range(d):
            qc.rx(np.pi / 4, q)
            qc.ry(a[q], q)
            qc.ry(-b[q], q)
            qc.rx(-np.pi / 4, q)
        qc.measure_all()
        return [qc]
    if model == "pqk":
        x = rng.normal(size=d)
        return [pqk_circuits(d, s, x) for s in backends.SETTINGS]
    raise KeyError(model)


def pqk_circuits(d: int, setting: str, x: Any) -> Any:
    """PQK circuit for one Pauli setting: Haar-random product start, Trotterised Heisenberg chain, basis change.

    ``x`` may be numbers or a ``ParameterVector`` of length d.
    """
    from qiskit import QuantumCircuit

    n = d + 1
    v = pqk_start(n)
    qc = QuantumCircuit(n)
    for q in range(n):
        theta = 2 * np.arccos(np.clip(abs(v[q, 0]), 0, 1))
        qc.u(theta, float(np.angle(v[q, 1]) - np.angle(v[q, 0])), 0, q)
    for _ in range(PQK_TROTTER):
        for j in range(d):
            angle = 2 * (PQK_T / PQK_TROTTER) * x[j]
            qc.rxx(angle, j, j + 1)
            qc.ryy(angle, j, j + 1)
            qc.rzz(angle, j, j + 1)
    for q in range(n):
        if setting == "X":
            qc.h(q)
        elif setting == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def cost_study(name: str, seed: int, mode: SplitMode, models: tuple[str, ...] | list[str]) -> CostStudy:
    """Transpiled resources and circuit counts for the quantum models in ``models``."""
    from qiskit.transpiler import generate_preset_pass_manager

    a = arrays(name)
    d = a.d_used
    fi = fold_inputs(name, seed, mode)
    n_tr, n_te = len(fi.y_train), len(fi.y_test)
    search_per_value = sum(_fidelity_pairs(len(ia), len(ib)) for ia, ib in fi.raw_folds)
    out: list[CostModel] = []
    for model in (m for m in QUANTUM_MODELS if m in models):
        circuits = representative_circuits(model, d, a.geometry, np.random.default_rng(0))
        gates: dict[str, int] = {}
        depth: dict[str, int] = {}
        for dev in backends.DEVICES:
            pms = [
                generate_preset_pass_manager(optimization_level=3, backend=backends.device(dev), seed_transpiler=ts)
                for ts in TRANSPILER_SEEDS
            ]
            per_circuit = [[backends.circuit_stats(pm.run(c)) for pm in pms] for c in circuits]
            gates[dev] = max(int(np.median([st[0] for st in stats])) for stats in per_circuit)
            depth[dev] = max(int(np.median([st[1] for st in stats])) for stats in per_circuit)
        per_sample = model in ("qlassifier", "pqk")
        out.append(
            CostModel(
                model=model,
                qubits=int(circuits[0].num_qubits),
                two_qubit_gates=gates,
                depth=depth,
                circuits_fit_predict=3 * (n_tr + n_te) if per_sample else _fidelity_pairs(n_tr, n_te),
                tuning_circuits_on_device=_ON_DEVICE_TUNING.get(model, 0) * search_per_value,
            )
        )
    return CostStudy(n_train=n_tr, n_test=n_te, devices=list(backends.DEVICES), models=out)


def fqk_pipeline_ratio(cost: CostStudy | None) -> float | None:
    """BT-7: circuits of the bandwidth-tuned fidelity-kernel pipeline (incl. its search) / Qlassifier circuits."""
    if cost is None:
        return None
    by = {m.model: m for m in cost.models}
    fqk = by.get("fqk_bandwidth_linear") or by.get("fqk_bandwidth_full")
    q = by.get("qlassifier")
    if fqk is None or q is None or q.circuits_fit_predict == 0:
        return None
    return (fqk.circuits_fit_predict + fqk.tuning_circuits_on_device) / q.circuits_fit_predict
