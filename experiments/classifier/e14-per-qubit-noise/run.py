"""e14: per-qubit vs whole-register simulation of Qlassifier circuits on IBM fake backends (phase 9).

Up: README.md · Log: ../learning-log.md (phase-9 addendum) · Decision: ../../adr/0006-noise-simulation-path.md
Code under test: src/qlassifier/backends.py (per_qubit_circuits, measure_bloch_per_qubit, per_qubit_exact)

Part 1 (exact, no shots): for real rows encoded on 4 qubits (Breast Cancer, PCA-8) and 6 qubits (Sonar, PCA-12),
every Pauli setting and all 3 fake devices, compare the measured marginal P(1) of every qubit:
  whole register: the transpiled register circuit (opt 3, seed 0), noisy density matrix of the active qubits;
  per qubit: each qubit's one-qubit circuit on the same physical qubit, noisy density matrix of that qubit;
then the device readout matrix of that physical qubit (single-qubit, checked) maps P(1) to the measured P(1).
Part 2 (shots): noise study part A on Breast Cancer (seeds 0-4, 1,024 shots, sampler seed 11) with both paths:
accuracy per cell and mean per device, with the binomial standard error of the difference.
Run: .venv/bin/python experiments/classifier/e14-per-qubit-noise/run.py
Writes exact.csv, shots.csv next to this file.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qlassifier import backends
from qlassifier.bench import noise as study
from qlassifier.bench.protocol import Prep, arrays, fold_inputs
from qlassifier.encoding import Encoding

HERE = Path(__file__).resolve().parent
ROWS = 5


def readout(name: str) -> dict[int, np.ndarray]:
    """Physical qubit -> 2x2 readout matrix [[P(0|0), P(1|0)], [P(0|1), P(1|1)]] (single-qubit entries only)."""
    out = {}
    for e in backends.noise_errors(name):
        if e["type"] == "roerror":
            (q,) = e["gate_qubits"][0]
            out[int(q)] = np.asarray(e["probabilities"], dtype=float)
    return out


def marginals(sim: AerSimulator, circuit, values, qubits: list[int]) -> tuple[np.ndarray, float]:  # type: ignore[no-untyped-def]
    """Exact P(1) of each listed physical qubit before readout (noisy density matrix), and the trace deviation.

    Aer's thermal-relaxation channels are trace preserving only to ~1e-10, so a register's density matrix has trace
    1 - O(1e-10); probabilities are normalised by the trace, as sampling does.
    """
    bound = circuit.assign_parameters(values).remove_final_measurements(inplace=False)
    bound.save_density_matrix(qubits=qubits)
    rho = np.asarray(sim.run(bound).result().data(0)["density_matrix"])
    diag = np.real(np.diag(rho))
    idx = np.arange(len(diag))
    trace = diag.sum()
    return np.array([diag[(idx >> k) & 1 == 1].sum() / trace for k in range(len(qubits))]), abs(1.0 - trace)


def encoded_rows(dataset: str, k: int) -> tuple[Encoding, np.ndarray]:
    a = arrays(dataset)
    X = Prep("pca8").fit(a.X).transform(a.X) if k == 8 else _pca(a.X, k)
    enc = Encoding.plan(X.shape[1], None, None).fit(X)
    return enc, X[:ROWS]


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(PCA(k, random_state=0).fit_transform(StandardScaler().fit_transform(X)))


def exact_part() -> pd.DataFrame:
    rows = []
    for dataset, k in (("breast_cancer", 8), ("sonar", 12)):
        enc, X = encoded_rows(dataset, k)
        tmpl = backends.templates(enc)
        flat = enc.angles(X).reshape(len(X), -1)
        for name in backends.DEVICES:
            assert backends.local_noise(backends.noise_errors(name))
            sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(backends.device(name)))
            ro = readout(name)
            whole = backends.transpile_pinned(backends.bloch_circuits(tmpl), name)
            assert backends.per_qubit_exact(whole, name)
            layout = backends.pinned_layout(whole, enc.n_qubits)
            single = backends.per_qubit_circuits(tmpl, name, layout)
            for setting in backends.SETTINGS:
                tc = whole[setting]
                for i in range(len(X)):
                    p_whole, tr_whole = marginals(sim, tc, backends._values(tc, flat[i : i + 1])[0], layout)
                    for q, phys in enumerate(layout):
                        pc = single[q][setting]
                        p_one, tr_one = marginals(sim, pc, backends._values(pc, flat[i : i + 1])[0], [phys])
                        m = ro[phys]
                        meas_whole = (1 - p_whole[q]) * m[0, 1] + p_whole[q] * m[1, 1]
                        meas_one = (1 - p_one[0]) * m[0, 1] + p_one[0] * m[1, 1]
                        rows.append({"dataset": dataset, "qubits": enc.n_qubits, "device": name, "setting": setting,
                                     "row": i, "qubit": q, "physical": phys, "p1_whole": meas_whole,
                                     "p1_per_qubit": meas_one, "abs_diff": abs(meas_whole - meas_one),
                                     "trace_dev_whole": tr_whole, "trace_dev_per_qubit": tr_one})  # fmt: skip
    return pd.DataFrame(rows)


def shots_part() -> pd.DataFrame:
    rows = []
    for seed in range(5):
        fi = fold_inputs("breast_cancer", seed, "stratified")
        g, C = study._tuned("breast_cancer", seed, "stratified")
        enc = Encoding.plan(8, None, None).fit(fi.X_train)
        X = np.vstack([fi.X_train, fi.X_test])
        tmpl = backends.templates(enc)
        for name in backends.DEVICES:
            whole = backends.transpile_pinned(backends.bloch_circuits(tmpl), name)
            single = backends.per_qubit_circuits(tmpl, name, backends.pinned_layout(whole, enc.n_qubits))
            paths = {
                "whole_register": backends.measure_bloch(whole, enc.angles(X), name, study.SHOTS, study.SAMPLER_SEED),
                "per_qubit": backends.measure_bloch_per_qubit(single, enc.angles(X), name, study.SHOTS, study.SAMPLER_SEED),
            }  # fmt: skip
            for path, R in paths.items():
                Ra, Rb = R[: len(fi.X_train)], R[len(fi.X_train) :]
                from qlassifier.tuning import gamma0, gaussian_gram

                n0 = gamma0(Ra)
                acc, ba = study._svc_eval(
                    gaussian_gram(Ra, Ra, g * n0), fi.y_train, gaussian_gram(Rb, Ra, g * n0), fi.y_test, C
                )
                rows.append({"seed": seed, "device": name, "path": path, "accuracy": acc, "balanced_accuracy": ba,
                             "n_test": len(fi.y_test)})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    exact = exact_part()
    exact.to_csv(HERE / "exact.csv", index=False)
    print(exact.groupby(["dataset", "device"]).abs_diff.max().to_string())
    print(f"max |diff| over {len(exact)} marginals: {exact.abs_diff.max():.2e}; "
          f"max trace deviation {exact.trace_dev_whole.max():.1e} (register) / {exact.trace_dev_per_qubit.max():.1e} (one qubit)")
    shots = shots_part()
    shots.to_csv(HERE / "shots.csv", index=False)
    wide = shots.pivot_table(index="device", columns="path", values="accuracy", aggfunc="mean")
    n = shots.n_test.iloc[0] * 5
    p = shots.accuracy.mean()
    wide["diff_pp"] = 100 * (wide["per_qubit"] - wide["whole_register"])
    wide["binomial_se_of_diff_pp"] = 100 * np.sqrt(2 * p * (1 - p) / n)
    print(wide.round(4).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
