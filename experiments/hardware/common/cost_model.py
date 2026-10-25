"""QPU-time cost model (protocol §2.4, BT-4): exact counts + IBM's job-time model, checked against published timings.

Up: ../README.md §2.4 · Experiment: ../e03-cost-model/README.md

Model (seconds): low  = sum over circuits of shots x (rep_delay + duration) + 2 s x ceil(circuits / 300)
                 high = sum over circuits of shots x 350 us              + 2 s x ceil(circuits / 300)
rep_delay = backend.default_rep_delay (250 us on the fake Heron / Eagle backends), duration from
QuantumCircuit.estimate_duration(target) of the transpiled circuit, 350 us = IBM's quick per-execution figure
(https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time), 300 circuits per job (arXiv:2604.18837).

Usage: PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python \
           experiments/hardware/common/cost_model.py --out experiments/hardware/e03-cost-model
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from qlassifier import backends
from qlassifier.bench import cost
from qlassifier.bench.protocol import arrays, fold_inputs

DEVICES = backends.DEVICES
OVERHEAD_S = 2.0
JOB_CIRCUITS = 300
QUICK_PER_EXEC_S = 350e-6
PUBLISHED = [  # per-circuit QPU time measured on IBM hardware (sources audited in docs/references/rea)
    {"source": "arXiv:2607.20377", "device": "ibm_fez (Heron r2)", "circuits": 900, "shots": 1024, "qpu_s": 244.0,
     "circuit": "ZZ kernel, 4 qubits, compute-uncompute, compiled depth <= 102, <= 22 CZ (3 jobs of 300; paper text)"},
    {"source": "arXiv:2604.18837 (code archive, qpu_budget_analysis.py)", "device": "ibm_fez (Heron r2)", "circuits": 1830,
     "shots": 1024, "qpu_s": 584.0, "circuit": "Belis / Rot2DoF kernels, 5 qubits, depth 22-74, 0-16 CZ (7 jobs)"},
    {"source": "arXiv:2605.14672", "device": "ibm_pittsburgh (Heron r3)", "circuits": 1675, "shots": 2048, "qpu_s": 19.3 * 60,
     "circuit": "fidelity kernel circuits, 6 jobs of <= 300"},
]  # fmt: skip


def qpu_time(circuits: int, shots: int, duration_s: float, rep_delay_s: float) -> tuple[float, float]:
    """(low, high) QPU seconds for ``circuits`` circuits of ``shots`` shots each."""
    jobs = math.ceil(circuits / JOB_CIRCUITS) if circuits else 0
    low = circuits * shots * (rep_delay_s + duration_s) + OVERHEAD_S * jobs
    high = circuits * shots * QUICK_PER_EXEC_S + OVERHEAD_S * jobs
    return low, high


def _transpiled(circuit: Any, dev: str, seed: int = 0) -> Any:
    from qiskit.transpiler import generate_preset_pass_manager

    return generate_preset_pass_manager(optimization_level=3, backend=backends.device(dev), seed_transpiler=seed).run(circuit)


def device_params(dev: str) -> dict[str, float]:
    b = backends.device(dev)
    conf = b.configuration()
    return {"default_rep_delay_s": float(conf.default_rep_delay), "dt_s": float(b.dt), "num_qubits": int(b.num_qubits)}


def model_table(dataset: str = "breast_cancer", seed: int = 0, shots: int = 1024) -> dict[str, Any]:
    """Per quantum model and device: 2Q gates, depth, duration, circuits and the QPU-time model for one fit + predict."""
    a = arrays(dataset)
    fi = fold_inputs(dataset, seed, "stratified")
    n_tr, n_te = len(fi.y_train), len(fi.y_test)
    study = cost.cost_study(dataset, seed, "stratified", cost.QUANTUM_MODELS)
    by = {m.model: m for m in study.models}
    rows = []
    for model in cost.QUANTUM_MODELS:
        circs = cost.representative_circuits(model, a.d_used, a.geometry, np.random.default_rng(0))
        per_dev = {}
        for dev in DEVICES:
            tcs = [_transpiled(c, dev) for c in circs]
            stats = [backends.circuit_stats(tc) for tc in tcs]
            dur = max(float(tc.estimate_duration(backends.device(dev).target, unit="s")) for tc in tcs)
            rd = device_params(dev)["default_rep_delay_s"]
            n_c = by[model].circuits_fit_predict
            low, high = qpu_time(n_c, shots, dur, rd)
            per_dev[dev] = {
                "two_qubit_gates": max(s[0] for s in stats), "depth": max(s[1] for s in stats), "duration_us": 1e6 * dur,
                "qpu_low_s": low, "qpu_high_s": high, "per_circuit_low_s": low / n_c, "per_circuit_high_s": high / n_c,
            }  # fmt: skip
        rows.append({
            "model": model, "qubits": by[model].qubits, "circuits_fit_predict": by[model].circuits_fit_predict,
            "tuning_circuits_on_device": by[model].tuning_circuits_on_device, "shots": shots, "devices": per_dev,
        })  # fmt: skip
    return {"dataset": dataset, "seed": seed, "n_train": n_tr, "n_test": n_te, "models": rows}


def published_check() -> list[dict[str, Any]]:
    """Bracket each published per-circuit timing with the model (durations of the matching depth class on FakeFez)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map

    rd = device_params("fake_fez")["default_rep_delay_s"]
    target = backends.device("fake_fez").target
    rng = np.random.default_rng(0)

    def zz_cu(n: int, ent: str) -> Any:
        fm = zz_feature_map(n, reps=2, entanglement=ent)
        qc = fm.assign_parameters(rng.uniform(0, np.pi, n)).compose(fm.assign_parameters(rng.uniform(0, np.pi, n)).inverse())
        qc.measure_all()
        return qc

    def rot2dof(n: int) -> Any:  # product map compute-uncompute (0 two-qubit gates), the Kakavand HW-02 class
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.rx(rng.uniform(0, np.pi), q)
            qc.ry(rng.uniform(0, np.pi), q)
            qc.ry(-rng.uniform(0, np.pi), q)
            qc.rx(-rng.uniform(0, np.pi), q)
        qc.measure_all()
        return qc

    classes = {
        # paper text: "compiled depth was at most 102 with at most 22 two-qubit gates (cz)" -> 4-qubit ZZ, linear
        "arXiv:2607.20377": [zz_cu(4, "linear")],
        "arXiv:2604.18837 (code archive, qpu_budget_analysis.py)": [rot2dof(5), zz_cu(5, "linear")],
        "arXiv:2605.14672": [zz_cu(4, "full"), zz_cu(8, "linear")],
    }
    out = []
    for p in PUBLISHED:
        durs = []
        for qc in classes[p["source"]]:
            tc = _transpiled(qc, "fake_fez")
            durs.append((float(tc.estimate_duration(target, unit="s")), *backends.circuit_stats(tc)))
        dur = max(d[0] for d in durs)
        low, high = qpu_time(p["circuits"], p["shots"], dur, rd)
        per = p["qpu_s"] / p["circuits"]
        out.append({
            **p, "per_circuit_s": per, "model_duration_us": 1e6 * dur, "model_classes": [{"duration_us": 1e6 * d, "two_qubit_gates": g, "depth": dp} for d, g, dp in durs],
            "model_low_s": low, "model_high_s": high, "model_per_circuit_low_s": low / p["circuits"],
            "model_per_circuit_high_s": high / p["circuits"], "inside": bool(low <= p["qpu_s"] <= high),
        })  # fmt: skip
    return out


def executed_counts_check() -> dict[str, Any]:
    """Counts of the model vs circuits actually executed by the shipped estimator on FakeFez (Breast Cancer, seed 0)."""
    from qlassifier import Qlassifier

    fi = fold_inputs("breast_cancer", 0, "stratified")
    executed = {"experiments": 0, "parameter_sets": 0}
    orig = backends._p1_from_counts

    def counting(circuits, binds, name, shots, seed, threads, method):  # type: ignore[no-untyped-def]
        executed["experiments"] += len(circuits)
        executed["parameter_sets"] += sum(len(next(iter(b.values()))) for b in binds)
        return orig(circuits, binds, name, shots, seed, threads, method)

    backends._p1_from_counts = counting  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = Qlassifier(backend="fake_fez", random_state=0).fit(fi.X_train, fi.y_train)
            clf.predict(fi.X_test)
        res = clf.resources_
        # whole-register semantics: 3 settings x N rows; the per-qubit path runs n_qubits x 3 experiments with N
        # parameter sets each, i.e. the same N x 3 device executions per qubit column.
        n = len(fi.y_train) + len(fi.y_test)
        whole = backends.measure_bloch(clf._transpiled, clf._enc.angles(fi.X_test[:20]), "fake_fez", 1024, 11)  # noqa: SLF001
    finally:
        backends._p1_from_counts = orig  # type: ignore[assignment]
    return {
        "resources": res.to_dict(), "model_circuits": 3 * n, "executed_parameter_sets_per_qubit": executed["parameter_sets"] / res.qubits,
        "executed_experiments": executed["experiments"], "whole_register_rows_checked": int(whole.shape[0]),
        "match": bool(res.circuits_fit + res.circuits_predict == 3 * n == executed["parameter_sets"] / res.qubits - 60 / res.qubits and res.two_qubit_gates == 0),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = {
            "devices": {d: device_params(d) for d in DEVICES},
            "model": {"overhead_s_per_job": OVERHEAD_S, "job_circuits": JOB_CIRCUITS, "quick_per_execution_s": QUICK_PER_EXEC_S,
                      "low": "shots x (rep_delay + duration) per circuit + overhead", "high": "shots x 350 us per circuit + overhead"},
            "breast_cancer": model_table(),
            "published": published_check(),
            "executed_counts": executed_counts_check(),
        }  # fmt: skip
    result["seconds"] = time.perf_counter() - t0
    (out / "cost_model.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["published"], indent=1))
    print(json.dumps(result["executed_counts"], indent=1))
    for row in result["breast_cancer"]["models"]:
        f = row["devices"]["fake_fez"]
        print(f"{row['model']:<22} q {row['qubits']:>2} circuits {row['circuits_fit_predict']:>9,} 2Q {f['two_qubit_gates']:>3} depth {f['depth']:>4} dur {f['duration_us']:6.2f} us  QPU {f['qpu_low_s']:9.1f}-{f['qpu_high_s']:9.1f} s")
    print(f"{result['seconds']:.0f}s")


if __name__ == "__main__":
    main()
