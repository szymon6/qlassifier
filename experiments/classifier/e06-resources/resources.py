"""e06: hardware resources per circuit (BT-8) and circuit counts per fit + predict incl. FQK bandwidth search (BT-7).

Up: README.md · Protocol: ../README.md §2.9. Copied from archived e06, counts updated. Transpiles one representative circuit per quantum model to FakeTorino (Heron r1), FakeFez (Heron r2),
FakeBrisbane (Eagle r3) at optimization_level 3, transpiler seeds 0-2; reports qubits, 2-qubit gates, depth,
and a crude error-free-shot estimate prod(1 - gate error) * prod(1 - readout error) from backend.target.
Circuit counts use the PRD BT-7 setting: Breast Cancer, 70/30 split -> N_train 398, N_test 171; fidelity kernels
skip the diagonal (N(N-1)/2) and evaluate test x all train rows (sklearn precomputed kernel).
Writes resources.json next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez, FakeTorino

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def product_circuit(d, basis="Y"):
    n = (d + 1) // 2
    qc = QuantumCircuit(n)
    x = rng.normal(size=2 * n) * 0.7
    for q in range(n):
        qc.rx(x[2 * q], q)
        qc.ry(x[2 * q + 1], q)
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def native_l2_circuit(d):
    n = (d + 1) // 2
    qc = QuantumCircuit(n)
    x = rng.normal(size=2 * n) * 0.3
    for _ in range(2):
        for q in range(n):
            qc.rx(x[2 * q], q)
            qc.ry(x[2 * q + 1], q)
        if _ == 0:
            for q in range(n - 1):
                qc.cz(q, q + 1)
    qc.measure_all()
    return qc


def fidelity_circuit(d, ent):
    fm = zz_feature_map(d, reps=2, entanglement=ent)
    a, b = rng.uniform(0, np.pi, d), rng.uniform(0, np.pi, d)
    qc = fm.assign_parameters(a).compose(fm.assign_parameters(b).inverse())
    qc.measure_all()
    return qc


def separable_fidelity_circuit(d):
    qc = QuantumCircuit(d)
    a, b = rng.uniform(-1.5, 1.5, d), rng.uniform(-1.5, 1.5, d)
    for q in range(d):
        qc.rx(np.pi / 4, q)
        qc.ry(a[q], q)
        qc.ry(-b[q], q)
        qc.rx(-np.pi / 4, q)
    qc.measure_all()
    return qc


def pqk_circuit(d, t=0.1, trotter=3):
    n = d + 1
    qc = QuantumCircuit(n)
    x = rng.normal(size=d)
    for q in range(n):
        qc.u(*rng.uniform(0, np.pi, 3), q)
    for _ in range(trotter):
        for j in range(d):
            ang = 2 * (t / trotter) * x[j]
            qc.rxx(ang, j, j + 1)
            qc.ryy(ang, j, j + 1)
            qc.rzz(ang, j, j + 1)
    qc.measure_all()
    return qc


def est_success(tc, be):
    t = be.target
    p = 1.0
    for inst in tc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(tc.find_bit(q).index for q in inst.qubits)
        try:
            props = t[name][qargs]
            if props is not None and props.error is not None:
                p *= 1 - props.error
        except KeyError:
            pass
    return p


def fqk_pipeline_circuits(ntr: int, nte: int, n_bw: int = 6, folds: int = 5) -> dict:
    """BT-7 (protocol §2.9): bandwidth-tuned fidelity-kernel pipeline with its bandwidth search on device.
    Per bandwidth and CV fold: train Gram N_a(N_a-1)/2 (diagonal = 1) + validation block N_b*N_a; then the final Gram
    at the chosen bandwidth: N_tr(N_tr-1)/2 + N_te*N_tr. C is swept on cached Grams (0 extra circuits)."""
    from sklearn.model_selection import StratifiedKFold
    y = np.r_[np.ones(int(round(ntr * 212 / 569))), np.zeros(ntr - int(round(ntr * 212 / 569)))]
    search = 0
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=0).split(np.zeros(ntr), y):
        search += len(a) * (len(a) - 1) // 2 + len(b) * len(a)
    final = ntr * (ntr - 1) // 2 + nte * ntr
    return {"search": n_bw * search, "final": final, "total": n_bw * search + final}


def main():
    out = {"devices": {}}
    for d, tag in ((8, "pca8"), (30, "all30")):
        circuits = {f"qlassifier product, 1 of 3 settings ({tag})": product_circuit(d)}
        if d == 8:
            circuits.update({
                "zz_qsvc pinned (full), compute-uncompute": fidelity_circuit(d, "full"),
                "fqk_bw linear, compute-uncompute": fidelity_circuit(d, "linear"),
                "separable (Bowles), compute-uncompute": separable_fidelity_circuit(d),
                "pqk Heisenberg (d+1 qubits), 1 of 3 settings": pqk_circuit(d),
            })
        for B in (FakeTorino, FakeFez, FakeBrisbane):
            be = B()
            dev = out["devices"].setdefault(be.name, {})
            for name, qc in circuits.items():
                vals = []
                for seed in range(3):
                    tc = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=seed).run(qc)
                    ops = tc.count_ops()
                    vals.append((sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx")), tc.depth(), est_success(tc, be)))
                v = np.array(vals)
                dev[name] = {"qubits": qc.num_qubits, "two_qubit_gates": int(np.median(v[:, 0])),
                             "two_qubit_range": [int(v[:, 0].min()), int(v[:, 0].max())],
                             "depth": int(np.median(v[:, 1])), "est_error_free_shot": round(float(np.median(v[:, 2])), 3)}
    for name, dev in out["devices"].items():
        print(name, json.dumps(dev, indent=1))
    ntr, nte = 398, 171
    fq = fqk_pipeline_circuits(ntr, nte)
    ours = 3 * (ntr + nte)
    out["circuit_counts_breast_cancer"] = {
        "n_train": ntr, "n_test": nte,
        "qlassifier_3_settings": ours,
        "fqk_bw_pipeline_search_6_bandwidths_x_5_folds": fq["search"],
        "fqk_bw_pipeline_final_gram": fq["final"],
        "fqk_bw_pipeline_total": fq["total"],
        "ratio_vs_fqk_bw_pipeline": round(fq["total"] / ours, 1),
        "ratio_vs_fqk_final_gram_only": round(fq["final"] / ours, 1),
        "pqk_3_settings": ours,
        "qlassifier_tuning_circuits": 0,
    }
    print(json.dumps(out["circuit_counts_breast_cancer"], indent=1))
    (HERE / "resources.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
