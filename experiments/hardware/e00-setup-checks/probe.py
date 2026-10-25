"""e00 setup checks: scaled FakeFez noise model, per-qubit measurement cost, fidelity-kernel cell cost, durations.

Up: ../README.md (protocol §2.5 noise scaling) · Log: ../learning-log.md
Run: OMP_NUM_THREADS=1 .venv/bin/python experiments/hardware/e00-setup-checks/probe.py
"""
from __future__ import annotations

import copy
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/hardware/common")
from qlassifier import backends  # noqa: E402
from qlassifier.encoding import Encoding  # noqa: E402

from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def scaled_fez(s: float):
    """FakeFez whose target has every gate / readout error x s and T1, T2 / s (s > 0).

    Qiskit 2.x Targets hand out copies of their properties, so every change goes back through
    ``update_instruction_properties`` and the ``qubit_properties`` setter.
    """
    from qiskit.providers import QubitProperties
    from qiskit.transpiler import InstructionProperties

    fake = FakeFez()
    t = copy.deepcopy(fake.target)
    n_err = 0
    for op in list(t.operation_names):
        props = t[op]
        if props is None:
            continue
        for q, p in list(props.items()):
            if p is None or p.error is None:
                continue
            t.update_instruction_properties(op, q, InstructionProperties(duration=p.duration, error=min(float(p.error) * s, 0.999)))
            n_err += 1
    if s > 0:
        t.qubit_properties = [
            QubitProperties(t1=None if qp.t1 is None else qp.t1 / s, t2=None if qp.t2 is None else qp.t2 / s, frequency=qp.frequency)
            for qp in t.qubit_properties
        ]
    attrs = [a for a in vars(fake) if "target" in a]
    fake._target = t  # noqa: SLF001
    return fake, n_err, attrs


print("python", sys.version.split()[0])
t0 = time.perf_counter()
base = NoiseModel.from_backend(FakeFez())
d1 = base.to_dict()["errors"]
print(f"base noise model: {len(d1)} errors, build {time.perf_counter() - t0:.1f}s")
for s in (0.5, 2.0):
    fake, n_err, attrs = scaled_fez(s)
    ref = FakeFez().target
    assert abs(fake.target.qubit_properties[0].t1 - ref.qubit_properties[0].t1 / s) < 1e-15
    assert abs(fake.target['sx'][(0,)].error - ref['sx'][(0,)].error * s) < 1e-12
    nm = NoiseModel.from_backend(fake)
    d = nm.to_dict()["errors"]
    ro1 = next(e for e in d1 if e["type"] == "roerror" and list(map(list, e["gate_qubits"])) == [[0]])["probabilities"]
    ro = next(e for e in d if e["type"] == "roerror" and list(map(list, e["gate_qubits"])) == [[0]])["probabilities"]
    sx1 = next(e for e in d1 if e["type"] == "qerror" and e["operations"] == ["sx"] and list(map(list, e["gate_qubits"])) == [[0]])
    sx = next(e for e in d if e["type"] == "qerror" and e["operations"] == ["sx"] and list(map(list, e["gate_qubits"])) == [[0]])
    print(f"s={s}: {len(d)} errors (attrs {attrs}); readout q0 {ro1[0][1]:.5f} -> {ro[0][1]:.5f} (x{ro[0][1]/ro1[0][1]:.2f});"
          f" sx q0 identity prob {sx1['probabilities'][0]:.6f} -> {sx['probabilities'][0]:.6f}"
          f" (non-identity x{(1-sx['probabilities'][0])/(1-sx1['probabilities'][0]):.2f}); kinds {sorted({i['name'] for inst in sx['instructions'] for i in inst})}")

# per-qubit Qlassifier cell on Breast Cancer (569 rows, 4 qubits) with a scaled simulator
from qlassifier.datasets import load_dataset  # noqa: E402

X, y = load_dataset("breast_cancer", return_X_y=True)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

Z = StandardScaler().fit_transform(PCA(8, random_state=0).fit_transform(StandardScaler().fit_transform(X.to_numpy())))
enc = Encoding.plan(8, None, None).fit(Z)
tmpl = backends.templates(enc)
t0 = time.perf_counter()
transpiled = backends.transpile_pinned(backends.bloch_circuits(tmpl), "fake_fez")
layout = backends.pinned_layout(transpiled, enc.n_qubits)
pq = backends.per_qubit_circuits(tmpl, "fake_fez", layout)
print(f"transpile + per-qubit circuits {time.perf_counter() - t0:.1f}s, layout {layout}, per_qubit_exact {backends.per_qubit_exact(transpiled, 'fake_fez')}")
for s in (0.0, 2.0):
    fake, _, _ = scaled_fez(s) if s > 0 else (FakeFez(), 0, [])
    nm = NoiseModel.from_backend(fake) if s > 0 else NoiseModel()
    sim = AerSimulator.from_backend(fake, noise_model=nm, method="density_matrix")
    sim.set_options(max_parallel_threads=1)
    flat = enc.angles(Z).reshape(len(Z), -1)
    ordered = [pq[q][st] for q in range(enc.n_qubits) for st in backends.SETTINGS]
    t0 = time.perf_counter()
    res = sim.run(ordered, shots=1024, seed_simulator=11, parameter_binds=[backends._binds(tc, flat) for tc in ordered], memory=False).result()
    R = np.empty((len(Z), enc.n_qubits, 3))
    i = 0
    for k, tc in enumerate(ordered):
        q, st = divmod(k, 3)
        for r in range(len(Z)):
            c = res.get_counts(i); i += 1
            p1 = sum(v for key, v in c.items() if key.replace(" ", "")[-1] == "1") / 1024
            R[r, q, st] = 1 - 2 * p1
    R = backends.renormalise(R)
    Re = enc.bloch(Z)
    print(f"s={s}: per-qubit cell {time.perf_counter() - t0:.1f}s, mean|R-exact| {np.abs(R-Re).mean():.4f}, RSS {rss_gb():.2f} GB")

# fidelity-kernel cell cost: FQK-bw linear and pinned ZZ full, 8 qubits, density matrix vs automatic
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit.circuit.library import zz_feature_map  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_aer.primitives import SamplerV2  # noqa: E402

rng = np.random.default_rng(0)
for ent in ("linear", "full"):
    pa, pb = ParameterVector("a", 8), ParameterVector("b", 8)
    fm = zz_feature_map(8, reps=2, entanglement=ent)
    qc = fm.assign_parameters(pa).compose(fm.assign_parameters(pb).inverse())
    qc.measure_all()
    tc = generate_preset_pass_manager(optimization_level=3, backend=FakeFez(), seed_transpiler=0).run(qc)
    two_q, depth = backends.circuit_stats(tc)
    try:
        dur = tc.estimate_duration(FakeFez().target, unit="s")
    except Exception as e:  # noqa: BLE001
        dur = f"n/a ({type(e).__name__})"
    n = 60
    vals = rng.uniform(0, np.pi, size=(n, len(tc.parameters)))
    for method in ("density_matrix", "automatic"):
        sim = AerSimulator.from_backend(FakeFez(), method=method)
        sim.set_options(max_parallel_threads=1)
        t0 = time.perf_counter()
        res = SamplerV2.from_backend(sim, seed=11).run([(tc, vals)], shots=1024).result()[0]
        dt = time.perf_counter() - t0
        p0 = (np.asarray(res.data.meas.bitcount()) == 0).mean(axis=-1)
        print(f"ZZ {ent}: 2Q {two_q}, depth {depth}, duration {dur}; {method}: {n} circuits {dt:.1f}s -> {dt/n*1000:.0f} ms/circuit, 4,170-circuit Gram ~ {dt/n*4170/60:.1f} min; mean p0 {p0.mean():.3f}; RSS {rss_gb():.2f} GB")
# Qlassifier circuit duration
for st, tc in transpiled.items():
    try:
        print("qlassifier", st, "duration s", tc.estimate_duration(FakeFez().target, unit="s"), "depth", tc.depth())
    except Exception as e:  # noqa: BLE001
        print("duration n/a", e)
print("done RSS", f"{rss_gb():.2f} GB")
