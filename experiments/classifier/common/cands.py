"""Candidate Qlassifier encodings for the stage-1 research loop (iteration 1+).

Up: ../README.md (protocol §2) · Log: ../learning-log.md · Base harness: lab.py (verified kernels; e00)
Every candidate is a product-state circuit (0 two-qubit gates) read out from local Pauli settings, so exact mode uses
closed forms. Each factory returns `extra_specs(d) -> list[Spec]` for `lab.run_seed`. Grids: 30 configs (protocol §2.4).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import lab


# ----------------------------------------------------------------------------- single-qubit algebra
def _rot(axis: str, th: np.ndarray) -> np.ndarray:
    """Batched SU(2) rotation matrices, shape (..., 2, 2)."""
    c, s = np.cos(th / 2), np.sin(th / 2)
    U = np.zeros(th.shape + (2, 2), dtype=complex)
    if axis == "x":
        U[..., 0, 0], U[..., 0, 1], U[..., 1, 0], U[..., 1, 1] = c, -1j * s, -1j * s, c
    elif axis == "y":
        U[..., 0, 0], U[..., 0, 1], U[..., 1, 0], U[..., 1, 1] = c, -s, s, c
    else:  # z
        U[..., 0, 0], U[..., 1, 1] = np.exp(-0.5j * th), np.exp(0.5j * th)
    return U


def bloch_of_states(psi: np.ndarray) -> np.ndarray:
    """psi: (..., 2) normalised single-qubit states -> Bloch vectors (..., 3) = (<X>, <Y>, <Z>)."""
    a, b = psi[..., 0], psi[..., 1]
    ab = np.conj(a) * b
    return np.stack([2 * ab.real, 2 * ab.imag, np.abs(a) ** 2 - np.abs(b) ** 2], axis=-1)


def reupload_bloch(X: np.ndarray, c: float, layers: int) -> np.ndarray:
    """Per qubit, from |0>: [RX(c x_a) RY(c x_b)] repeated `layers` times (data re-uploading, Fourier degree = layers).
    layers = 1 equals lab.bloch_vectors. Returns (N, n, 3)."""
    N, d = X.shape
    n = (d + 1) // 2
    Z = np.zeros((N, 2 * n))
    Z[:, :d] = c * X
    a, b = Z[:, 0::2], Z[:, 1::2]
    U1 = _rot("y", b) @ _rot("x", a)  # (N, n, 2, 2): RX first, then RY
    psi = np.zeros((N, n, 2), dtype=complex)
    psi[..., 0] = 1.0
    for _ in range(layers):
        psi = np.einsum("...ij,...j->...i", U1, psi)
    return bloch_of_states(psi)


def riemann_bloch(z: np.ndarray) -> np.ndarray:
    """Inverse stereographic projection: |psi(z)> ~ |0> + z|1>  ->  r = (2 Re z, 2 Im z, 1 - |z|^2) / (1 + |z|^2).
    z: complex array (N, m). Returns (N, m, 3). The Bloch sphere is the Riemann sphere of z."""
    den = 1.0 + np.abs(z) ** 2
    return np.stack([2 * z.real / den, 2 * z.imag / den, (1.0 - np.abs(z) ** 2) / den], axis=-1)


def _gauss(FA: np.ndarray, FB: np.ndarray, mult: float) -> tuple[np.ndarray, np.ndarray]:
    FA2, FB2 = FA.reshape(len(FA), -1), FB.reshape(len(FB), -1)
    g0 = lab.gamma0(FA2)
    return lab.gaussian_from_features(FA2, FA2, mult * g0), lab.gaussian_from_features(FB2, FA2, mult * g0)


def _g(square: bool = True) -> list[dict]:
    return lab.kgrid(1.0, square=square)


# ----------------------------------------------------------------------------- iteration 1 candidates
def spectrum_specs(d: int) -> list[lab.Spec]:
    """H3/H4: richer per-qubit Fourier spectra, still 0 two-qubit gates.
    - bloch_ms: Bloch features at angle scales {c0/2, c0, 2 c0} concatenated (9 circuits per sample).
    - bloch_reup2: [RX RY] uploaded twice per qubit (Fourier degree 2; 3 circuits per sample, depth x2)."""
    c0 = 2.0 / np.sqrt(d)
    out = []

    def k_ms(Xtr, Xte, bw, ctx):
        key = ("ms",)
        if key not in ctx:
            ctx[key] = (np.concatenate([lab.bloch_vectors(Xtr, f * c0).reshape(len(Xtr), -1) for f in (0.5, 1, 2)], 1),
                        np.concatenate([lab.bloch_vectors(Xte, f * c0).reshape(len(Xte), -1) for f in (0.5, 1, 2)], 1))
        return _gauss(*ctx[key], bw)

    out.append(lab.Spec("bloch_ms", "quantum", _g(), kernel=k_ms, notes="multi-scale Bloch features (c0/2, c0, 2c0)"))

    def k_reup(Xtr, Xte, bw, ctx):
        key = ("reup2",)
        if key not in ctx:
            ctx[key] = (reupload_bloch(Xtr, c0, 2), reupload_bloch(Xte, c0, 2))
        return _gauss(*ctx[key], bw)

    out.append(lab.Spec("bloch_reup2", "quantum", _g(), kernel=k_reup, notes="data re-uploading x2 per qubit"))
    return out


def balanced_specs(d: int) -> list[lab.Spec]:
    """H5: class-balanced SVM loss (targets BA). Quantum candidate + classical controls with the same option."""
    c0 = 2.0 / np.sqrt(d)

    def k_bg(Xtr, Xte, bw, ctx):
        return _gauss(lab.bloch_vectors(Xtr, c0), lab.bloch_vectors(Xte, c0), bw)

    out = [lab.Spec("bloch_gauss_bal", "quantum", _g(), kernel=k_bg, extra={"class_weight": "balanced"})]
    out.append(lab.Spec("rbf_svm_bal", "classical", lab.kgrid(1.0 / d, square=True),
                        make=lambda c: SVC(C=c["C"], gamma=c["bw"], kernel="rbf", class_weight="balanced")))
    out.append(lab.Spec("logreg_bal", "classical", [{"C": c} for c in np.logspace(-4, 4, lab.BUDGET)],
                        make=lambda c: LogisticRegression(C=c["C"], max_iter=5000, class_weight="balanced")))
    out.append(lab.Spec("random_forest_bal", "classical",
                        [{"max_features": mf, "min_samples_leaf": msl} for mf in ["sqrt", 0.1, 0.2, 0.4, 0.7, 1.0] for msl in [1, 2, 4, 8, 16]],
                        make=lambda c: RandomForestClassifier(n_estimators=200, max_features=c["max_features"], class_weight="balanced",
                                                              min_samples_leaf=c["min_samples_leaf"], random_state=0, n_jobs=1)))
    out.append(lab.Spec("grad_boost_bal", "classical",
                        [{"learning_rate": lr, "max_leaf_nodes": ml} for lr in [0.01, 0.03, 0.1, 0.3, 1.0] for ml in [2, 3, 7, 15, 31, 63]],
                        make=lambda c: HistGradientBoostingClassifier(learning_rate=c["learning_rate"], max_leaf_nodes=c["max_leaf_nodes"],
                                                                      max_iter=200, random_state=0, class_weight="balanced")))
    return out


# ----------------------------------------------------------------------------- complex-valued data (H2)
IONO_PAIRS = [(2 * k - 1, 2 * k) for k in range(1, 17)]  # (Re_k, Im_k) column pairs after dropping the constant column 1
IONO_SINGLES = [0]  # column 0 = Re of the zero-lag ACF (Im is the dropped constant 0)


def complex_view(X: np.ndarray, pairs=IONO_PAIRS, singles=IONO_SINGLES) -> np.ndarray:
    """Raw columns -> complex array (N, m): singles as real numbers, pairs as Re + i Im."""
    cols = [X[:, j] + 0j for j in singles] + [X[:, a] + 1j * X[:, b] for a, b in pairs]
    return np.stack(cols, axis=1)


def complex_specs(d: int) -> list[lab.Spec]:
    """H2 (Ionosphere, recipe `native`): one qubit per complex autocorrelation value (17 qubits).
    - cplx_riemann: |0> + w|1> (normalised), w = z / rms|z| (training rows); Gaussian readout on Bloch vectors.
    - cplx_pairs: the Bloch map RX(c x_re) RY(c x_im) on aligned (Re, Im) pairs, one shared scale (keeps phase geometry).
    Both read raw columns (no per-column standardisation: it would distort complex phases)."""
    out = []

    def k_riem(Xtr, Xte, bw, ctx):
        key = ("riem",)
        if key not in ctx:
            zA, zB = complex_view(Xtr), complex_view(Xte)
            s = 1.0 / np.sqrt(np.mean(np.abs(zA) ** 2))
            ctx[key] = (riemann_bloch(s * zA), riemann_bloch(s * zB))
        return _gauss(*ctx[key], bw)

    out.append(lab.Spec("cplx_riemann", "quantum", _g(), kernel=k_riem, extra={"raw": True},
                        notes="Riemann-sphere qubit per complex ACF value"))

    def k_pairs(Xtr, Xte, bw, ctx):
        key = ("pairs",)
        if key not in ctx:
            zA, zB = complex_view(Xtr), complex_view(Xte)
            sig = np.sqrt(np.mean(np.abs(zA - zA.mean(0)) ** 2))  # pooled spread, one scale for all components
            m = zA.shape[1]
            c = 2.0 / np.sqrt(2 * m)
            def enc(z):
                Z = np.zeros((len(z), 2 * m))
                Z[:, 0::2], Z[:, 1::2] = z.real / sig, z.imag / sig
                return lab.bloch_vectors(Z, c)
            ctx[key] = (enc(zA), enc(zB))
        return _gauss(*ctx[key], bw)

    out.append(lab.Spec("cplx_pairs", "quantum", _g(), kernel=k_pairs, extra={"raw": True},
                        notes="Bloch map on aligned (Re, Im) pairs, shared scale"))

    def k_bg_std(Xtr, Xte, bw, ctx):  # standard Qlassifier encoding on standardized columns (control)
        c0 = 2.0 / np.sqrt(Xtr.shape[1])
        return _gauss(lab.bloch_vectors(Xtr, c0), lab.bloch_vectors(Xte, c0), bw)

    out.append(lab.Spec("bloch_gauss_std", "quantum", _g(), kernel=k_bg_std, notes="control: standard encoding, standardized"))
    return out


# ----------------------------------------------------------------------------- geometry-aware encoding (BT-6 route)
def geo_bloch(X: np.ndarray, ref: np.ndarray, periods: dict[int, float] | None = None,
              sphere_pairs: list[tuple[int, int]] | None = None, c: float | None = None) -> np.ndarray:
    """Geometry-aware product encoding, 0 two-qubit gates. Returns Bloch vectors (N, n_qubits, 3).
    - periodic column j with period P (e.g. 360 deg, 24 h, 2*pi rad): own qubit, RY(2*pi*x/P) from |0>
      -> r = (sin t, 0, cos t), a great circle; exactly P-periodic (Gaussian readout = von Mises kernel).
    - sphere pair (lat_deg, lon_deg): one qubit, RY(90 deg - lat) then RZ(lon) -> r = the point's unit vector
      (the Bloch sphere is the globe; Gaussian readout = exp(-gamma * chord^2)).
    - every other column: z-scored with the training rows `ref`, then standard Bloch pairs RX(c x_a) RY(c x_b),
      c = 2/sqrt(d_other) unless given."""
    X = np.asarray(X, float)
    periods = periods or {}
    sphere_pairs = sphere_pairs or []
    used = set(periods) | {j for pr in sphere_pairs for j in pr}
    other = [j for j in range(X.shape[1]) if j not in used]
    parts = []
    for j, P in periods.items():
        t = 2 * np.pi * X[:, j] / P
        parts.append(np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], axis=-1)[:, None, :])
    for la, lo in sphere_pairs:
        pol, az = np.deg2rad(90.0 - X[:, la]), np.deg2rad(X[:, lo])
        parts.append(np.stack([np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)], axis=-1)[:, None, :])
    if other:
        mu, sd = ref[:, other].mean(0), ref[:, other].std(0)
        sd[sd == 0] = 1.0
        Z = (X[:, other] - mu) / sd
        cc = c if c is not None else 2.0 / np.sqrt(len(other))
        parts.append(lab.bloch_vectors(Z, cc))
    return np.concatenate(parts, axis=1)


def twin_features(X: np.ndarray, ref: np.ndarray, periods: dict[int, float] | None = None,
                  sphere_pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Classical twin features with the same geometry made by hand: sin/cos per periodic column, 3-D unit vector
    per sphere pair, z-scores for the rest."""
    X = np.asarray(X, float)
    periods = periods or {}
    sphere_pairs = sphere_pairs or []
    used = set(periods) | {j for pr in sphere_pairs for j in pr}
    other = [j for j in range(X.shape[1]) if j not in used]
    cols = []
    for j, P in periods.items():
        t = 2 * np.pi * X[:, j] / P
        cols += [np.sin(t), np.cos(t)]
    for la, lo in sphere_pairs:
        pol, az = np.deg2rad(90.0 - X[:, la]), np.deg2rad(X[:, lo])
        cols += [np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)]
    if other:
        mu, sd = ref[:, other].mean(0), ref[:, other].std(0)
        sd[sd == 0] = 1.0
        cols += list(((X[:, other] - mu) / sd).T)
    return np.stack(cols, axis=1)


def make_geo_specs(periods: dict[int, float] | None = None, sphere_pairs: list[tuple[int, int]] | None = None,
                   name: str = "qlass_geo", twin: bool = True):
    """Factory for recipe `native`: Qlassifier with declared geometry (+ the classical twin control).
    Both read raw columns and use the same 30-config ridge grid."""
    def _specs(d: int) -> list[lab.Spec]:
        def k_geo(Xtr, Xte, bw, ctx):
            key = ("geo",)
            if key not in ctx:
                ctx[key] = (geo_bloch(Xtr, Xtr, periods, sphere_pairs), geo_bloch(Xte, Xtr, periods, sphere_pairs))
            return _gauss(*ctx[key], bw)
        out = [lab.Spec(name, "quantum", _g(), kernel=k_geo, extra={"raw": True},
                        notes="periodic columns on great circles, lat/lon on the Bloch sphere, others as Bloch pairs")]
        if twin:
            def k_twin(Xtr, Xte, bw, ctx):
                FA, FB = twin_features(Xtr, Xtr, periods, sphere_pairs), twin_features(Xte, Xtr, periods, sphere_pairs)
                g0 = 1.0 / FA.var(axis=0).sum()
                return lab.gaussian_from_features(FA, FA, bw * g0), lab.gaussian_from_features(FB, FA, bw * g0)
            out.append(lab.Spec("twin_geo_rbf", "control", _g(), kernel=k_twin, extra={"raw": True},
                                notes="classical twin: RBF on hand-made sin/cos / unit-vector features"))
        return out
    return _specs


def quake_geo_specs(d: int) -> list[lab.Spec]:
    """Qlassifier geometry mode for `deep_quakes`: (lat, lon) on one qubit (Bloch vector = epicentre direction)."""
    return make_geo_specs(**lab.GEOMETRY["deep_quakes"], twin=False)(d)


def protein_geo_specs(d: int) -> list[lab.Spec]:
    """Qlassifier geometry mode for `protein_strands`: each dihedral angle on its own circle qubit (period 360 deg)."""
    return make_geo_specs(**lab.GEOMETRY["protein_strands"], twin=False)(d)


def quake_twin_specs(d: int) -> list[lab.Spec]:
    return [s for s in make_geo_specs(**lab.GEOMETRY["deep_quakes"], twin=True)(d) if s.name.startswith("twin")]


def protein_twin_specs(d: int) -> list[lab.Spec]:
    return [s for s in make_geo_specs(**lab.GEOMETRY["protein_strands"], twin=True)(d) if s.name.startswith("twin")]


# ----------------------------------------------------------------------------- iteration 3: shifted kernel grid (exploratory)
M_WIDE = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # protocol grid x 8: same 6 x 5 ridge shape, narrower kernels


def _wide(base: float, square: bool) -> list[dict]:
    return [{"bw": base * (m**2 if square else m), "C": ct / m**2, "m": m, "ct": ct} for m in M_WIDE for ct in lab.CT_GRID]


def quake_wide_specs(d: int) -> list[lab.Spec]:
    """H9 (exploratory, post hoc): every kernel model on `deep_quakes` gets the same x8-shifted 30-config ridge grid."""
    from sklearn.svm import SVC as _SVC
    geom = lab.GEOMETRY["deep_quakes"]

    def k_geo(Xtr, Xte, bw, ctx):
        key = ("geo",)
        if key not in ctx:
            ctx[key] = (geo_bloch(Xtr, Xtr, **geom), geo_bloch(Xte, Xtr, **geom))
        return _gauss(*ctx[key], bw)

    def k_bg(Xtr, Xte, bw, ctx):
        c0 = 2.0 / np.sqrt(Xtr.shape[1])
        return _gauss(lab.bloch_vectors(Xtr, c0), lab.bloch_vectors(Xte, c0), bw)

    return [lab.Spec("qlass_geo_wide", "quantum", _wide(1.0, True), kernel=k_geo, extra={"raw": True}),
            lab.Spec("bloch_gauss_wide", "quantum", _wide(1.0, True), kernel=k_bg),
            lab.Spec("rbf_svm_wide", "classical", _wide(1.0 / d, True), make=lambda c: _SVC(C=c["C"], gamma=c["bw"], kernel="rbf"))]


def div6_geo_specs(d: int) -> list[lab.Spec]:
    """Iteration 5 (H12): geometry mode on `protein_div6` (6 circle qubits)."""
    return make_geo_specs(**lab.GEOMETRY["protein_div6"], twin=False)(d)


def div10_geo_specs(d: int) -> list[lab.Spec]:
    """Iteration 5 (H13): geometry mode on `protein_div10` (10 circle qubits)."""
    return make_geo_specs(**lab.GEOMETRY["protein_div10"], twin=False)(d)
