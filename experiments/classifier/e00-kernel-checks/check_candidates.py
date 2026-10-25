"""e00b: check iteration-1+ candidate encodings in common/cands.py against Qiskit circuits (statevector).

Up: README.md · Writes results_candidates.json next to this file.
Run: .venv/bin/python experiments/classifier/e00-kernel-checks/check_candidates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import cands  # noqa: E402
import lab  # noqa: E402

rng = np.random.default_rng(11)
out: dict[str, float] = {}


def bloch_from_sv(sv: Statevector, n: int) -> np.ndarray:
    r = np.zeros((n, 3))
    for q in range(n):
        for k, P in enumerate("XYZ"):
            label = ["I"] * n
            label[n - 1 - q] = P
            r[q, k] = np.real(sv.expectation_value(Pauli("".join(label))))
    return r


# 1) data re-uploading: [RX(c x_a) RY(c x_b)] x L per qubit
d, c = 5, 0.9
X = rng.normal(size=(4, d))
n = (d + 1) // 2
for L in (1, 2, 3):
    R = cands.reupload_bloch(X, c, L)
    errs = []
    for i, x in enumerate(X):
        z = np.zeros(2 * n)
        z[:d] = c * x
        qc = QuantumCircuit(n)
        for _ in range(L):
            for q in range(n):
                qc.rx(z[2 * q], q)
                qc.ry(z[2 * q + 1], q)
        errs.append(np.abs(bloch_from_sv(Statevector(qc), n) - R[i]).max())
    out[f"reupload_L{L}_bloch_max_abs_err"] = float(max(errs))
out["reupload_L1_equals_lab_bloch_max_abs_err"] = float(np.abs(cands.reupload_bloch(X, c, 1) - lab.bloch_vectors(X, c)).max())

# 2) Riemann-sphere encoding: |0> + w|1> (normalised) prepared as RY(2 arctan|w|) then P(arg w)
w = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
R = cands.riemann_bloch(w)
errs, ferr = [], []
svs = []
for i in range(len(w)):
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.ry(2 * np.arctan(abs(w[i, q])), q)
        qc.p(np.angle(w[i, q]), q)
    sv = Statevector(qc)
    svs.append(sv)
    errs.append(np.abs(bloch_from_sv(sv, 3) - R[i]).max())
    # amplitude check: state equals (|0> + w|1>)/norm per qubit (up to global phase)
Kq = np.array([[abs(a.inner(b)) ** 2 for b in svs] for a in svs])
out["riemann_bloch_max_abs_err"] = float(max(errs))
out["riemann_product_fidelity_max_abs_err"] = float(np.abs(lab.product_fidelity_from_bloch(R, R) - Kq).max())
# chordal identity: per-qubit fidelity = |1 + conj(z) w|^2 / ((1+|z|^2)(1+|w|^2))
z1, z2 = w[0, 0], w[1, 0]
f_closed = abs(1 + np.conj(z1) * z2) ** 2 / ((1 + abs(z1) ** 2) * (1 + abs(z2) ** 2))
f_bloch = (1 + R[0, 0] @ R[1, 0]) / 2
out["riemann_chordal_identity_abs_err"] = float(abs(f_closed - f_bloch))

# 3) geometry-aware encoding: periodic column -> RY(2 pi x / P); (lat, lon) -> RY(90 - lat) then RZ(lon); others Bloch pairs
Xg = np.c_[rng.uniform(-90, 90, 4), rng.uniform(-180, 180, 4), rng.uniform(0, 24, 4), rng.normal(size=(4, 3))]
Rg = cands.geo_bloch(Xg, Xg, {2: 24.0}, [(0, 1)])
mu, sd = Xg[:, 3:].mean(0), Xg[:, 3:].std(0)
errs = []
for i, x in enumerate(Xg):
    nq = Rg.shape[1]
    qc = QuantumCircuit(nq)
    qc.ry(2 * np.pi * x[2] / 24.0, 0)                       # periodic qubit
    qc.ry(np.deg2rad(90 - x[0]), 1)                          # sphere qubit
    qc.rz(np.deg2rad(x[1]), 1)
    z = np.zeros(4)
    z[:3] = (x[3:] - mu) / sd * (2 / np.sqrt(3))
    for q in range(2):                                       # Bloch pairs for the 3 other columns (2 qubits)
        qc.rx(z[2 * q], 2 + q)
        qc.ry(z[2 * q + 1], 2 + q)
    errs.append(np.abs(bloch_from_sv(Statevector(qc), nq) - Rg[i]).max())
out["geo_bloch_max_abs_err"] = float(max(errs))

ok = all(v < 1e-9 for k, v in out.items() if k.endswith("_err"))
out["all_ok"] = bool(ok)
(HERE / "results_candidates.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
