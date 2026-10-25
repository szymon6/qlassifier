"""e00: check every fast kernel in common/lab.py against Qiskit circuits (statevector).

Up: ../README.md. Writes results.json next to this file.
Run: .venv/bin/python docs/roadmap/01-fair-verdict/experiments/e00-kernel-checks/check_kernels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Pauli, Statevector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import lab  # noqa: E402

rng = np.random.default_rng(7)
out: dict[str, float] = {}


def bloch_from_sv(sv: Statevector, n: int) -> np.ndarray:
    r = np.zeros((n, 3))
    for q in range(n):
        for k, P in enumerate("XYZ"):
            label = ["I"] * n
            label[n - 1 - q] = P
            r[q, k] = np.real(sv.expectation_value(Pauli("".join(label))))
    return r


# 1) product map, 2 features per qubit: RX(c x_a), RY(c x_b) from |0> (shipped circuit)
d, c = 7, 0.8
X = rng.normal(size=(6, d))
R = lab.bloch_vectors(X, c)
n = (d + 1) // 2
svs = []
for x in X:
    qc = QuantumCircuit(n)
    z = np.zeros(2 * n)
    z[:d] = c * x
    for q in range(n):
        qc.rx(z[2 * q], q)
        qc.ry(z[2 * q + 1], q)
    svs.append(Statevector(qc))
Rq = np.array([bloch_from_sv(s, n) for s in svs])
out["bloch_vectors_max_abs_err"] = float(np.abs(R - Rq).max())
Kq = np.array([[abs(a.inner(b)) ** 2 for b in svs] for a in svs])
out["product_fidelity_max_abs_err"] = float(np.abs(lab.product_fidelity_from_bloch(R, R) - Kq).max())

# 2) ZZ feature map (qiskit 2.x function), full and linear, reps 2
for ent in ["full", "linear"]:
    k = 5
    Xz = rng.uniform(0, np.pi, size=(5, k))
    S = lab.zz_states(Xz, 2, ent)
    fm = zz_feature_map(k, reps=2, entanglement=ent)
    Sq = np.array([Statevector(fm.assign_parameters(x)).data for x in Xz])
    Kf = lab.fidelity_from_states(S, S)
    Kfq = lab.fidelity_from_states(Sq, Sq)
    out[f"zz_{ent}_kernel_max_abs_err"] = float(np.abs(Kf - Kfq).max())
    out[f"zz_{ent}_state_overlap_min"] = float(np.min(np.abs(np.sum(S.conj() * Sq, axis=1))))

# 3) native entangled encoding with CZ chain (layers 1 and 2)
for L in [1, 2]:
    d = 6
    Xn = rng.normal(size=(4, d))
    psi, n = lab.native_states(Xn, 0.7, L)
    Rl = lab.local_paulis(psi, n)
    errs = []
    for i, x in enumerate(Xn):
        qc = QuantumCircuit(n)
        z = 0.7 * x

        def enc():
            for q in range(n):
                qc.rx(z[2 * q], q)
                qc.ry(z[2 * q + 1], q)

        enc()
        for q in range(n - 1):
            qc.cz(q, q + 1)
        if L == 2:
            enc()
        errs.append(np.abs(bloch_from_sv(Statevector(qc), n) - Rl[i]).max())
    out[f"native_L{L}_local_pauli_max_abs_err"] = float(max(errs))

# 3b) native product (layers=0) gives the same kernel as the closed-form Bloch map
Xn = rng.normal(size=(8, 6))
psi0, n0 = lab.native_states(Xn, 0.9, 0)
K_native = lab.product_fidelity_from_bloch(lab.local_paulis(psi0, n0), lab.local_paulis(psi0, n0))
Rb = lab.bloch_vectors(Xn, 0.9)
out["native_L0_equals_bloch_kernel_max_abs_err"] = float(np.abs(K_native - lab.product_fidelity_from_bloch(Rb, Rb)).max())

# 4) Heisenberg PQK embedding (exp(-i th (XX+YY+ZZ)) = RXX(2th) RYY(2th) RZZ(2th))
d, t, T = 4, 0.9, 3
Xp = rng.normal(size=(3, d))
F = lab.heisenberg_pqk_features(Xp, t, T, seed=1234).reshape(3, d + 1, 3)
rng2 = np.random.default_rng(1234)
v = rng2.normal(size=(d + 1, 2)) + 1j * rng2.normal(size=(d + 1, 2))
v /= np.linalg.norm(v, axis=1, keepdims=True)
errs = []
for i, x in enumerate(Xp):
    qc = QuantumCircuit(d + 1)
    for q in range(d + 1):
        qc.initialize(v[q], q)
    for _ in range(T):
        for j in range(d):
            th = (t / T) * x[j]
            qc.rxx(2 * th, j, j + 1)
            qc.ryy(2 * th, j, j + 1)
            qc.rzz(2 * th, j, j + 1)
    errs.append(np.abs(bloch_from_sv(Statevector(qc), d + 1) - F[i]).max())
out["pqk_heisenberg_local_pauli_max_abs_err"] = float(max(errs))

# 5) separable kernel per feature: RX(pi/4) then RY(x) from |0>: k = 3/4 + cos(dx)/4
xs = rng.uniform(-2, 2, size=5)
sv1 = []
for x in xs:
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 4, 0)
    qc.ry(x, 0)
    sv1.append(Statevector(qc))
Kq = np.array([[abs(a.inner(b)) ** 2 for b in sv1] for a in sv1])
Kc = 0.75 + 0.25 * np.cos(xs[:, None] - xs[None, :])
out["separable_per_feature_max_abs_err"] = float(np.abs(Kq - Kc).max())

# 6) PSD by construction: product kernel from noisy, shrunk, non-unit vectors
Rn = R * rng.uniform(0.6, 1.1, size=R.shape[:2])[..., None] + rng.normal(scale=0.05, size=R.shape)
out["product_kernel_min_eig_noisy_vectors"] = float(np.linalg.eigvalsh(lab.product_fidelity_from_bloch(Rn, Rn)).min())

ok = all(v < 1e-9 for k, v in out.items() if k.endswith("_err")) and out["zz_full_state_overlap_min"] > 1 - 1e-9 \
    and out["product_kernel_min_eig_noisy_vectors"] > -1e-12
out["all_ok"] = bool(ok)
(HERE / "results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
