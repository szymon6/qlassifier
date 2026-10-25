"""Noise study (TDD §7.8, ADR 0006): exact vs shot noise vs IBM device noise models.

- Part A, every dataset: Qlassifier (core or geometry) on all rows, the first 5 seeds; modes ``shots`` + 3 fake
  devices. Hyperparameters from exact-mode tuning of the same split; no re-tuning on noisy data.
- Part B, ``breast_cancer``: a stratified 60 train / 40 test subset of the first 3 seeds; Qlassifier, FQK-bw linear,
  PQK and separable on identical rows, each tuned in exact mode on the subset; 3 fake devices. Fidelity Grams by
  compute-uncompute P(0^n), training diagonal 1, symmetric eigen-clip PSD repair; PQK 3 settings per sample.
1,024 shots; optimisation level 3, transpiler seed 0; sampler seed 11 (phase-7 values, e05 / e08).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import cache
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from qlassifier import backends
from qlassifier.bench.cost import pqk_circuits
from qlassifier.bench.models import fidelity_from_states, heisenberg_pqk_features, separable_kernel, zz_states
from qlassifier.bench.protocol import N_FOLDS, Prep, Task, _grid_scores, arrays, fold_inputs, protocol_models
from qlassifier.bench.protocol import qlassifier_scores as _q_scores
from qlassifier.bench.result import NoisePart, NoiseRecord, NoiseStudy, NoiseSummary
from qlassifier.datasets import SplitMode, outer_split
from qlassifier.encoding import Encoding, Geometry
from qlassifier.tuning import gamma0, gaussian_gram

__all__ = ["DEVICES", "PART_B_MODELS", "SHOTS", "noise_study", "noise_tasks", "run_noise_task"]

FloatArray = NDArray[np.float64]
SHOTS = 1024
SAMPLER_SEED = 11
DEVICES = backends.DEVICES
PART_A_SEEDS = 5
PART_B_SEEDS = 3
PART_B_DATASET = "breast_cancer"
PART_B_MODELS: tuple[str, ...] = ("qlassifier", "fqk_bandwidth_linear", "pqk", "separable")
PART_B_TRAIN, PART_B_TEST = 60, 40
_PART_B_WEIGHT = {"fqk_bandwidth_linear": 1000.0, "separable": 300.0, "pqk": 20.0, "qlassifier": 5.0}


def noise_tasks(name: str, seeds: Sequence[int], part_a_weight: float = 30.0) -> list[Task]:
    """Tasks of the noise study for one dataset (Part B only on Breast Cancer)."""
    tasks = [
        Task("noise", name, int(s), "qlassifier", dev, "A", part_a_weight)
        for s in list(seeds)[:PART_A_SEEDS]
        for dev in ("shots", *DEVICES)
    ]
    if name == PART_B_DATASET:
        tasks += [
            Task("noise", name, int(s), m, dev, "B", _PART_B_WEIGHT[m])
            for s in list(seeds)[:PART_B_SEEDS]
            for dev in DEVICES
            for m in PART_B_MODELS
        ]
    return tasks


def _svc_eval(Ktr: FloatArray, ytr: NDArray[Any], Kte: FloatArray, yte: NDArray[Any], C: float) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.svm import SVC

    pred = SVC(C=C, kernel="precomputed").fit(Ktr, ytr).predict(Kte)
    return float(accuracy_score(yte, pred)), float(balanced_accuracy_score(yte, pred))


def _kernel_r(Kn: FloatArray, Ke: FloatArray) -> float | None:
    iu = np.triu_indices_from(Ke, 1)
    a, b = Kn[iu], Ke[iu]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _measured_bloch(enc: Encoding, X: FloatArray, device: str) -> tuple[FloatArray, int, int]:
    """Bloch vectors of every row under ``device`` (shots or fake), with 2Q gates and depth per circuit.

    Fake devices: each qubit simulated alone on its pinned physical qubit when that is exact (product circuit,
    local noise; ADR 0006 amendment, e14), else the whole register. Resources are the whole-register circuits'.
    """
    tmpl = backends.templates(enc)
    if device == "shots":
        R = backends.sample_shots(enc.bloch(X), SHOTS, np.random.default_rng(SAMPLER_SEED))
        return R, 0, backends.logical_depth(tmpl)
    transpiled = backends.transpile_pinned(backends.bloch_circuits(tmpl), device)
    stats = [backends.circuit_stats(tc) for tc in transpiled.values()]
    if backends.per_qubit_exact(transpiled, device):
        circuits = backends.per_qubit_circuits(tmpl, device, backends.pinned_layout(transpiled, enc.n_qubits))
        R = backends.measure_bloch_per_qubit(circuits, enc.angles(X), device, SHOTS, SAMPLER_SEED, threads=1)
    else:
        R = backends.measure_bloch(transpiled, enc.angles(X), device, SHOTS, SAMPLER_SEED, threads=1)
    return R, max(s[0] for s in stats), max(s[1] for s in stats)


def _bloch_record(
    part: Literal["A", "B"],
    seed: int,
    device: str,
    enc: Encoding,
    A: FloatArray,
    ya: NDArray[Any],
    B: FloatArray,
    yb: NDArray[Any],
    g: float,
    C: float,
    t0: float,
) -> NoiseRecord:
    Fa, Fb = enc.bloch(A), enc.bloch(B)
    g0 = gamma0(Fa)
    Ke_tr, Ke_te = gaussian_gram(Fa, Fa, g * g0), gaussian_gram(Fb, Fa, g * g0)
    e_acc, e_ba = _svc_eval(Ke_tr, ya, Ke_te, yb, C)
    R, two_q, depth = _measured_bloch(enc, np.vstack([A, B]), device)
    Ra, Rb = R[: len(A)], R[len(A) :]
    n0 = gamma0(Ra)
    Kn_tr, Kn_te = gaussian_gram(Ra, Ra, g * n0), gaussian_gram(Rb, Ra, g * n0)
    acc, ba = _svc_eval(Kn_tr, ya, Kn_te, yb, C)
    return NoiseRecord(
        part=part,
        model="qlassifier",
        device=device,
        seed=seed,
        accuracy=acc,
        balanced_accuracy=ba,
        exact_accuracy=e_acc,
        exact_balanced_accuracy=e_ba,
        drop_pp=100.0 * (e_acc - acc),
        kernel_r=_kernel_r(Kn_tr, Ke_tr),
        circuits=3 * (len(A) + len(B)),
        two_qubit_gates=two_q,
        depth=depth,
        n_train=len(A),
        n_test=len(B),
        seconds=time.perf_counter() - t0,
    )


@cache
def _tuned(name: str, seed: int, mode: SplitMode) -> tuple[float, float]:
    """Exact-mode (g, C) of Qlassifier on one seed (cached per worker: every device cell of a seed shares it)."""
    from qlassifier.tuning import select

    a = arrays(name)
    fi = fold_inputs(name, seed, mode)
    spec = next(s for s in protocol_models(name) if s.id == "qlassifier")
    grid = [(float(c["bw"]), float(c["C"])) for c in spec.grid]
    return grid[select(_q_scores(fi.folds, a.d_used, a.geometry, grid))]


def run_part_a(name: str, seed: int, device: str, mode: SplitMode) -> NoiseRecord:
    """Part A: Qlassifier on all rows of one seed under one noise mode."""
    t0 = time.perf_counter()
    a = arrays(name)
    fi = fold_inputs(name, seed, mode)
    g, C = _tuned(name, seed, mode)
    enc = Encoding.plan(a.d_used, cast(Geometry | None, a.geometry), None).fit(fi.X_train)
    return _bloch_record("A", seed, device, enc, fi.X_train, fi.y_train, fi.X_test, fi.y_test, g, C, t0)


# ----------------------------------------------------------------------------- part B (paired subset)
def part_b_subset(seed: int) -> tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    """Stratified 60 / 40 subset of the seed's Breast Cancer split (raw rows)."""
    from sklearn.model_selection import train_test_split

    a = arrays(PART_B_DATASET)
    tr, te = outer_split(a.y, None, seed, "stratified")
    tr_s, _ = train_test_split(tr, train_size=PART_B_TRAIN, stratify=a.y[tr], random_state=seed)
    te_s, _ = train_test_split(te, train_size=PART_B_TEST, stratify=a.y[te], random_state=seed)
    return a.X[tr_s], a.y[tr_s], a.X[te_s], a.y[te_s]


def _subset_config(model: str, seed: int, Xr: FloatArray, yr: NDArray[np.int64]) -> dict[str, Any]:
    """Exact-mode tuning on the subset: 5-fold stratified CV, per-fold preprocessing (same protocol)."""
    from sklearn.model_selection import StratifiedKFold

    a = arrays(PART_B_DATASET)
    spec = next(s for s in protocol_models(PART_B_DATASET) if s.id == model)
    folds = []
    for ia, ib in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xr, yr):
        p = Prep(a.recipe).fit(Xr[ia])
        folds.append((p.transform(Xr[ia]), yr[ia], p.transform(Xr[ib]), yr[ib]))
    if model == "qlassifier":
        grid = [(float(c["bw"]), float(c["C"])) for c in spec.grid]
        scores = _q_scores(folds, a.d_used, None, grid)
        cv = scores.sum(axis=1)
    else:
        cv = np.zeros(len(spec.grid))
        for A, ya, B, yb in folds:
            cv += _grid_scores(spec, A, ya, B, yb, a.recipe)
    return dict(spec.grid[int(np.argmax(cv))])


def _fidelity_device(circuit: Any, A: FloatArray, B: FloatArray | None, device: str) -> tuple[FloatArray, int, int]:
    """Compute-uncompute P(0^n) on a fake device for all pairs (training upper triangle, or B x A)."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2

    tc = generate_preset_pass_manager(optimization_level=3, backend=backends.device(device), seed_transpiler=0).run(
        circuit
    )
    params = list(tc.parameters)

    def row(x: FloatArray, xp: FloatArray) -> list[float]:
        vals = {"a": x, "b": xp}
        return [float(vals[p.vector.name][p.index]) for p in params]

    if B is None:
        iu = np.triu_indices(len(A), 1)
        values = np.array([row(A[i], A[j]) for i, j in zip(*iu, strict=True)])
    else:
        values = np.array([row(b, a_) for b in B for a_ in A])
    sim = backends._simulator(device)
    sim.set_options(max_parallel_threads=1)
    res = SamplerV2.from_backend(sim, seed=SAMPLER_SEED).run([(tc, values)], shots=SHOTS).result()[0]
    p0 = (np.asarray(res.data.meas.bitcount()) == 0).mean(axis=-1)
    if B is None:
        K = np.eye(len(A))
        K[iu] = p0
        K[(iu[1], iu[0])] = p0
        w, v = np.linalg.eigh(K)  # symmetric eigen-clip PSD repair
        K = (v * np.maximum(w, 0)) @ v.T
    else:
        K = p0.reshape(len(B), len(A))
    two_q, depth = backends.circuit_stats(tc)
    return K, two_q, depth


def _compute_uncompute(model: str, d: int) -> Any:
    """Parameterised compute-uncompute circuit U(a) U(b)^dagger + measure for a fidelity kernel."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map

    pa, pb = ParameterVector("a", d), ParameterVector("b", d)
    if model == "fqk_bandwidth_linear":
        fm = zz_feature_map(d, reps=2, entanglement="linear")
        qc = fm.assign_parameters(pa).compose(fm.assign_parameters(pb).inverse())
    else:  # separable: per qubit RX(pi/4) RY(a) then the inverse of RX(pi/4) RY(b)
        qc = QuantumCircuit(d)
        for q in range(d):
            qc.rx(np.pi / 4, q)
            qc.ry(pa[q], q)
            qc.ry(-pb[q], q)
            qc.rx(-np.pi / 4, q)
    qc.measure_all()
    return qc


def _pqk_device(X: FloatArray, device: str) -> tuple[FloatArray, int, int]:
    """PQK local Paulis measured on a fake device (3 settings per sample, d + 1 qubits)."""
    from qiskit.circuit import ParameterVector
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2

    N, d = X.shape
    n = d + 1
    p = ParameterVector("x", d)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backends.device(device), seed_transpiler=0)
    tcs = [pm.run(pqk_circuits(d, s, p)) for s in backends.SETTINGS]
    sim = backends._simulator(device)
    sim.set_options(max_parallel_threads=1)
    pubs = [(tc, X[:, [int(q.index) for q in tc.parameters]]) for tc in tcs]
    res = SamplerV2.from_backend(sim, seed=SAMPLER_SEED).run(pubs, shots=SHOTS).result()
    comps = []
    for i in range(3):
        meas = res[i].data.meas
        p1 = np.stack([np.asarray(meas.slice_bits([q]).bitcount(), float).mean(axis=-1) for q in range(n)], axis=-1)
        comps.append(1.0 - 2.0 * p1)
    F = np.stack(comps, axis=-1).reshape(N, -1)
    stats = [backends.circuit_stats(tc) for tc in tcs]
    return F, max(s[0] for s in stats), max(s[1] for s in stats)


def run_part_b(seed: int, device: str, model: str) -> NoiseRecord:
    """Part B: one model on the paired 60 / 40 Breast Cancer subset under one fake device."""
    t0 = time.perf_counter()
    a = arrays(PART_B_DATASET)
    Xr, yr, Xs, ys = part_b_subset(seed)
    p = Prep(a.recipe).fit(Xr)
    A, B = p.transform(Xr), p.transform(Xs)
    d = A.shape[1]
    cfg = _subset_config(model, seed, Xr, yr)
    if model == "qlassifier":
        enc = Encoding.plan(d, None, None).fit(A)
        return _bloch_record("B", seed, device, enc, A, yr, B, ys, float(cfg["bw"]), float(cfg["C"]), t0)
    C, bw = float(cfg["C"]), cfg["bw"]
    if model == "pqk":
        FA, FB = heisenberg_pqk_features(A), heisenberg_pqk_features(B)
        g0 = 1.0 / (FA.shape[1] * FA.var())
        Ke_tr, Ke_te = gaussian_gram(FA, FA, bw * g0), gaussian_gram(FB, FA, bw * g0)
        F, two_q, depth = _pqk_device(np.vstack([A, B]), device)
        Fa, Fb = F[: len(A)], F[len(A) :]
        n0 = 1.0 / (Fa.shape[1] * Fa.var())
        Kn_tr, Kn_te = gaussian_gram(Fa, Fa, bw * n0), gaussian_gram(Fb, Fa, bw * n0)
        circuits = 3 * (len(A) + len(B))
    else:
        if model == "fqk_bandwidth_linear":
            P, Q = bw * A, bw * B
            SA, SB = zz_states(P, 2, "linear"), zz_states(Q, 2, "linear")
            Ke_tr, Ke_te = fidelity_from_states(SA, SA), fidelity_from_states(SB, SA)
        else:
            from sklearn.preprocessing import MinMaxScaler

            mm = MinMaxScaler((-np.pi / 2, np.pi / 2)).fit(A)
            P, Q = bw * mm.transform(A), bw * mm.transform(B)
            Ke_tr, Ke_te = separable_kernel(P, P), separable_kernel(Q, P)
        circuit = _compute_uncompute(model, d)
        Kn_tr, two_q, depth = _fidelity_device(circuit, P, None, device)
        Kn_te, _, _ = _fidelity_device(circuit, P, Q, device)
        circuits = len(A) * (len(A) - 1) // 2 + len(B) * len(A)
    e_acc, e_ba = _svc_eval(Ke_tr, yr, Ke_te, ys, C)
    acc, ba = _svc_eval(Kn_tr, yr, Kn_te, ys, C)
    return NoiseRecord(
        part="B",
        model=model,
        device=device,
        seed=seed,
        accuracy=acc,
        balanced_accuracy=ba,
        exact_accuracy=e_acc,
        exact_balanced_accuracy=e_ba,
        drop_pp=100.0 * (e_acc - acc),
        kernel_r=_kernel_r(Kn_tr, Ke_tr),
        circuits=circuits,
        two_qubit_gates=two_q,
        depth=depth,
        n_train=len(A),
        n_test=len(B),
        seconds=time.perf_counter() - t0,
    )


def run_noise_task(task: Task, mode: SplitMode) -> NoiseRecord:
    """Run one noise-study cell."""
    assert task.seed is not None
    assert task.device is not None
    if task.part == "A":
        return run_part_a(task.dataset, task.seed, task.device, mode)
    return run_part_b(task.seed, task.device, task.model)


# ----------------------------------------------------------------------------- aggregation
def noise_study(name: str, seeds: Sequence[int], records: Sequence[NoiseRecord]) -> NoiseStudy | None:
    """Assemble the study from finished records (None before the first record)."""
    if not records:
        return None
    parts = [
        NoisePart(
            id="A", models=["qlassifier"], seeds=[int(s) for s in list(seeds)[:PART_A_SEEDS]],
            devices=["shots", *DEVICES], rows="all", n_train=None, n_test=None,
        )
    ]  # fmt: skip
    if name == PART_B_DATASET:
        parts.append(
            NoisePart(
                id="B", models=list(PART_B_MODELS), seeds=[int(s) for s in list(seeds)[:PART_B_SEEDS]],
                devices=list(DEVICES), rows="subset", n_train=PART_B_TRAIN, n_test=PART_B_TEST,
            )
        )  # fmt: skip
    ordered = sorted(
        records,
        key=lambda r: (r.part, r.model, (("shots", *DEVICES).index(r.device)), r.seed),
    )
    summary: list[NoiseSummary] = []
    groups: dict[tuple[str, str], list[NoiseRecord]] = {}
    for r in ordered:
        groups.setdefault((r.part, r.model), []).append(r)
    for (part, model), recs in groups.items():
        exact_by_seed = {r.seed: (r.exact_accuracy, r.exact_balanced_accuracy) for r in recs}
        summary.append(
            NoiseSummary(
                part=part,
                model=model,
                device="exact",
                accuracy_mean=float(np.mean([v[0] for v in exact_by_seed.values()])),
                balanced_accuracy_mean=float(np.mean([v[1] for v in exact_by_seed.values()])),
                drop_mean_pp=0.0,
                drop_worst_pp=0.0,
                kernel_r_mean=1.0,
                n_seeds=len(exact_by_seed),
            )
        )
        for dev in ("shots", *DEVICES):
            dr = [r for r in recs if r.device == dev]
            if not dr:
                continue
            rs = [r.kernel_r for r in dr if r.kernel_r is not None]
            summary.append(
                NoiseSummary(
                    part=part,
                    model=model,
                    device=dev,
                    accuracy_mean=float(np.mean([r.accuracy for r in dr])),
                    balanced_accuracy_mean=float(np.mean([r.balanced_accuracy for r in dr])),
                    drop_mean_pp=float(np.mean([r.drop_pp for r in dr])),
                    drop_worst_pp=float(np.max([r.drop_pp for r in dr])),
                    kernel_r_mean=float(np.mean(rs)) if rs else None,
                    n_seeds=len(dr),
                )
            )
    return NoiseStudy(shots=SHOTS, devices=["shots", *DEVICES], parts=parts, records=ordered, summary=summary)


def max_device_drop(noise: NoiseStudy | None) -> float | None:
    """Largest mean accuracy drop of Qlassifier over the 3 fake devices in part A (pp)."""
    if noise is None:
        return None
    drops = [s.drop_mean_pp for s in noise.summary if s.part == "A" and s.model == "qlassifier" and s.device in DEVICES]
    return float(max(drops)) if drops else None
