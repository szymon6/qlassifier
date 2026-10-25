"""Scaled-noise sweep on FakeFez (protocol §2.3): Qlassifier, FQK-bw linear and pinned ZZ at s x device error rates.

Up: ../README.md §2.3 · Experiment: ../e02-noise-sweep/README.md · Checks: ../e00-setup-checks/README.md

Scale s acts on the device Target (every instruction error x s, T1 and T2 / s); Aer rebuilds the noise model, so
every depolarizing and readout probability scales exactly by s (e00). Layout and gates are fixed at s = 1 (circuits
transpiled once on the unscaled FakeFez). Cells reuse the shipped noise-study routines for subsets, exact tuning,
kernels and metrics (``qlassifier.bench.noise``); only the simulator differs.

Usage (repo root, <= 3 workers, under the watchdog):
  PYTHONPATH=experiments/hardware/common OMP_NUM_THREADS=1 .venv/bin/python \
      experiments/hardware/common/noise_sweep.py --out <dir> --jobs 3
Appends one JSON line per finished cell to <dir>/cells.jsonl (a crash keeps finished cells; finished cells are
skipped on restart).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qlassifier import backends
from qlassifier.bench import noise as shipped
from qlassifier.bench.models import _minmax, fidelity_from_states, zz_states
from qlassifier.bench.protocol import Prep, arrays, fold_inputs
from qlassifier.encoding import Encoding
from qlassifier.tuning import gamma0, gaussian_gram

FloatArray = NDArray[np.float64]
DATASET = "breast_cancer"
DEVICE = "fake_fez"
SCALES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
ZZ_SCALES: tuple[float, ...] = (0.0, 1.0, 3.0)
PART_A_SEEDS = (0, 1, 2, 3, 4)
PART_B_SEEDS = (0, 1, 2)
ZZ_SEEDS = (0, 1)
SHOTS = shipped.SHOTS
SAMPLER_SEED = shipped.SAMPLER_SEED
JOB_CIRCUITS = 300
WEIGHT = {"zz_qsvc": 1100.0, "fqk_bandwidth_linear": 200.0, "qlassifier": 6.0}


# ----------------------------------------------------------------------------- scaled device
def scaled_fez(s: float) -> Any:
    """FakeFez whose Target carries every instruction error x s and T1, T2 / s (s > 0); s = 1 is the device model."""
    from qiskit.providers import QubitProperties
    from qiskit.transpiler import InstructionProperties
    from qiskit_ibm_runtime.fake_provider import FakeFez

    fake = FakeFez()
    if s == 1.0:
        return fake
    t = copy.deepcopy(fake.target)
    for op in list(t.operation_names):
        props = t[op]
        if props is None:
            continue
        for q, p in list(props.items()):
            if p is None or p.error is None:
                continue
            t.update_instruction_properties(
                op, q, InstructionProperties(duration=p.duration, error=min(float(p.error) * s, 0.999))
            )
    if s > 0:
        t.qubit_properties = [
            QubitProperties(
                t1=None if qp.t1 is None else qp.t1 / s,
                t2=None if qp.t2 is None else qp.t2 / s,
                frequency=qp.frequency,
            )
            for qp in t.qubit_properties
        ]
    fake._target = t  # noqa: SLF001  (FakeBackendV2 caches its Target here)
    return fake


@cache
def simulator(s: float) -> Any:
    """Density-matrix Aer simulator of FakeFez at noise scale s (s = 0: ideal sampler, shot noise only)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    fake = scaled_fez(s)
    nm = NoiseModel.from_backend(fake) if s > 0 else NoiseModel(basis_gates=list(fake.operation_names))  # s = 0: ideal, device basis
    sim = AerSimulator.from_backend(fake, noise_model=nm, method="density_matrix")
    sim.set_options(max_parallel_threads=1)
    return sim


def _p1_from_counts(sim: Any, circuits: Sequence[Any], binds: Sequence[Mapping[Any, FloatArray]], seed: int) -> list[FloatArray]:
    """P(1) per measured qubit and parameter set from counts (same bookkeeping as ``backends._p1_from_counts``)."""
    result = sim.run(list(circuits), shots=SHOTS, seed_simulator=seed, parameter_binds=list(binds), memory=False).result()
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
        out.append(p1 / SHOTS)
    return out


# ----------------------------------------------------------------------------- Qlassifier cells (per qubit)
@cache
def _qlassifier_circuits(d: int) -> tuple[list[dict[str, Any]], int, int]:
    """Per-qubit circuits on the s = 1 pinned layout; (circuits, 2Q gates, depth of the whole-register circuits)."""
    enc = Encoding.plan(d, None, None)
    tmpl = backends.templates(enc)
    transpiled = backends.transpile_pinned(backends.bloch_circuits(tmpl), DEVICE)
    assert backends.per_qubit_exact(transpiled, DEVICE)
    stats = [backends.circuit_stats(tc) for tc in transpiled.values()]
    circuits = backends.per_qubit_circuits(tmpl, DEVICE, backends.pinned_layout(transpiled, enc.n_qubits))
    return circuits, max(st[0] for st in stats), max(st[1] for st in stats)


def _measure_bloch(enc: Encoding, X: FloatArray, s: float) -> FloatArray:
    circuits, _, _ = _qlassifier_circuits(len(enc.plain))
    angles = enc.angles(X)
    N, n, _ = angles.shape
    flat = angles.reshape(N, 3 * n)
    ordered = [circuits[q][st] for q in range(n) for st in backends.SETTINGS]
    p1 = _p1_from_counts(simulator(s), ordered, [backends._binds(tc, flat) for tc in ordered], SAMPLER_SEED)
    R = np.empty((N, n, 3))
    for i, p in enumerate(p1):
        q, k = divmod(i, len(backends.SETTINGS))
        R[:, q, k] = 1.0 - 2.0 * p[:, 0]
    return backends.renormalise(R)


def qlassifier_cell(part: str, seed: int, s: float, enc: Encoding, A: FloatArray, ya: Any, B: FloatArray, yb: Any, g: float, C: float) -> dict[str, Any]:
    Fa, Fb = enc.bloch(A), enc.bloch(B)
    g0 = gamma0(Fa)
    e_acc, e_ba = shipped._svc_eval(gaussian_gram(Fa, Fa, g * g0), ya, gaussian_gram(Fb, Fa, g * g0), yb, C)
    R = _measure_bloch(enc, np.vstack([A, B]), s)
    Ra, Rb = R[: len(A)], R[len(A) :]
    n0 = gamma0(Ra)
    Kn_tr, Kn_te = gaussian_gram(Ra, Ra, g * n0), gaussian_gram(Rb, Ra, g * n0)
    acc, ba = shipped._svc_eval(Kn_tr, ya, Kn_te, yb, C)
    _, two_q, depth = _qlassifier_circuits(len(enc.plain))
    return dict(
        part=part, model="qlassifier", scale=s, seed=seed, accuracy=acc, balanced_accuracy=ba, exact_accuracy=e_acc,
        exact_balanced_accuracy=e_ba, kernel_r=shipped._kernel_r(Kn_tr, gaussian_gram(Fa, Fa, g * g0)),
        circuits=3 * (len(A) + len(B)), shots=SHOTS, two_qubit_gates=two_q, depth=depth, n_train=len(A), n_test=len(B),
        config={"gamma": g, "C": C},
    )  # fmt: skip


# ----------------------------------------------------------------------------- fidelity-kernel cells (whole register)
@cache
def _kernel_circuit(model: str, d: int) -> Any:
    """Transpiled compute-uncompute circuit on the unscaled FakeFez (opt 3, seed 0)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map
    from qiskit.transpiler import generate_preset_pass_manager

    pa, pb = ParameterVector("a", d), ParameterVector("b", d)
    ent = "linear" if model == "fqk_bandwidth_linear" else "full"
    fm = zz_feature_map(d, reps=2, entanglement=ent)
    qc: QuantumCircuit = fm.assign_parameters(pa).compose(fm.assign_parameters(pb).inverse())
    qc.measure_all()
    return generate_preset_pass_manager(optimization_level=3, backend=backends.device(DEVICE), seed_transpiler=0).run(qc)


def _fidelity_scaled(tc: Any, A: FloatArray, B: FloatArray | None, s: float) -> FloatArray:
    """Compute-uncompute P(0^n) at scale s for all pairs (training upper triangle, or B x A); shipped bookkeeping."""
    from qiskit_aer.primitives import SamplerV2

    params = list(tc.parameters)

    def row(x: FloatArray, xp: FloatArray) -> list[float]:
        vals = {"a": x, "b": xp}
        return [float(vals[p.vector.name][p.index]) for p in params]

    if B is None:
        iu = np.triu_indices(len(A), 1)
        values = np.array([row(A[i], A[j]) for i, j in zip(*iu, strict=True)])
    else:
        values = np.array([row(b, a_) for b in B for a_ in A])
    res = SamplerV2.from_backend(simulator(s), seed=SAMPLER_SEED).run([(tc, values)], shots=SHOTS).result()[0]
    p0 = (np.asarray(res.data.meas.bitcount()) == 0).mean(axis=-1)
    if B is None:
        K = np.eye(len(A))
        K[iu] = p0
        K[(iu[1], iu[0])] = p0
        w, v = np.linalg.eigh(K)  # symmetric eigen-clip PSD repair (shipped part B)
        return (v * np.maximum(w, 0)) @ v.T
    return p0.reshape(len(B), len(A))


def fidelity_cell(model: str, seed: int, s: float, A: FloatArray, ya: Any, B: FloatArray, yb: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    d = A.shape[1]
    C = float(cfg["C"])
    if model == "fqk_bandwidth_linear":
        bw = float(cfg["bw"])
        P, Q = bw * A, bw * B
        SA, SB = zz_states(P, 2, "linear"), zz_states(Q, 2, "linear")
    else:  # pinned ZZ QSVC: MinMax [0, pi] on training rows, test clipped, full entanglement
        bw = None
        P, Q = _minmax(A, B, 0.0, np.pi, clip=True)
        SA, SB = zz_states(P, 2, "full"), zz_states(Q, 2, "full")
    Ke_tr, Ke_te = fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)
    tc = _kernel_circuit(model, d)
    Kn_tr = _fidelity_scaled(tc, P, None, s)
    Kn_te = _fidelity_scaled(tc, P, Q, s)
    e_acc, e_ba = shipped._svc_eval(Ke_tr, ya, Ke_te, yb, C)
    acc, ba = shipped._svc_eval(Kn_tr, ya, Kn_te, yb, C)
    two_q, depth = backends.circuit_stats(tc)
    return dict(
        part="B", model=model, scale=s, seed=seed, accuracy=acc, balanced_accuracy=ba, exact_accuracy=e_acc,
        exact_balanced_accuracy=e_ba, kernel_r=shipped._kernel_r(Kn_tr, Ke_tr),
        circuits=len(A) * (len(A) - 1) // 2 + len(B) * len(A), shots=SHOTS, two_qubit_gates=two_q, depth=depth,
        n_train=len(A), n_test=len(B), config={"scale": bw, "C": C},
    )  # fmt: skip


# ----------------------------------------------------------------------------- cells
def run_cell(part: str, model: str, s: float, seed: int) -> dict[str, Any]:
    import warnings

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        a = arrays(DATASET)
        if part == "A":
            fi = fold_inputs(DATASET, seed, "stratified")
            g, C = shipped._tuned(DATASET, seed, "stratified")
            enc = Encoding.plan(a.d_used, None, None).fit(fi.X_train)
            rec = qlassifier_cell("A", seed, s, enc, fi.X_train, fi.y_train, fi.X_test, fi.y_test, g, C)
        else:
            Xr, yr, Xs, ys = shipped.part_b_subset(seed)
            p = Prep(a.recipe).fit(Xr)
            A, B = p.transform(Xr), p.transform(Xs)
            cfg = shipped._subset_config(model, seed, Xr, yr)
            if model == "qlassifier":
                enc = Encoding.plan(A.shape[1], None, None).fit(A)
                rec = qlassifier_cell("B", seed, s, enc, A, yr, B, ys, float(cfg["bw"]), float(cfg["C"]))
            else:
                rec = fidelity_cell(model, seed, s, A, yr, B, ys, cfg)
        rec["seconds"] = time.perf_counter() - t0
        rec["drop_pp"] = 100.0 * (rec["exact_accuracy"] - rec["accuracy"])
        rec["drop_ba_pp"] = 100.0 * (rec["exact_balanced_accuracy"] - rec["balanced_accuracy"])
        return rec


def all_cells() -> list[tuple[str, str, float, int]]:
    cells = [("A", "qlassifier", s, seed) for seed in PART_A_SEEDS for s in SCALES]
    cells += [("B", "qlassifier", s, seed) for seed in PART_B_SEEDS for s in SCALES]
    cells += [("B", "fqk_bandwidth_linear", s, seed) for seed in PART_B_SEEDS for s in SCALES]
    cells += [("B", "zz_qsvc", s, seed) for seed in ZZ_SEEDS for s in ZZ_SCALES]
    cells.sort(key=lambda c: -WEIGHT[c[1]])
    return cells


def main() -> None:
    from joblib import Parallel, delayed, parallel_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="model ids to run (default: all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cells.jsonl"
    done: set[tuple[str, str, float, int]] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["part"], r["model"], float(r["scale"]), int(r["seed"])))
    cells = [c for c in all_cells() if c not in done and (a.only is None or c[1] in a.only)]
    print(f"{len(cells)} cells to run ({len(done)} already done), jobs {a.jobs}, {time.strftime('%FT%TZ', time.gmtime())}", flush=True)
    t0 = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(run_cell)(*c) for c in cells)
        for i, rec in enumerate(gen, 1):
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"  {i}/{len(cells)} {rec['part']} {rec['model']} s={rec['scale']} seed={rec['seed']}: acc {rec['accuracy']:.3f}"
                f" (exact {rec['exact_accuracy']:.3f}, drop {rec['drop_pp']:+.1f} pp, r {rec['kernel_r']}) {rec['seconds']:.0f}s;"
                f" elapsed {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print(f"done {time.strftime('%FT%TZ', time.gmtime())}")


if __name__ == "__main__":
    main()
