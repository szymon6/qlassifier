"""Columns -> gate angles -> exact Bloch vectors (TDD §3.1-3.2, ADR 0001).

Core encoding: plain columns are z-scored on training rows; qubit q holds the pair (x_2q, x_2q+1):
|0> -RX(c0 x_2q)-RY(c0 x_2q+1)-, c0 = 2/sqrt(max(d_plain, 8)). Geometry mode (opt-in): a periodic column gets its own
qubit, RY(2 pi x / P) (a great circle); a (latitude, longitude) pair gets one qubit, RY(90 deg - lat)-RZ(lon)
(the Bloch vector is the direction). One closed form covers every kind: RX(a) -> RY(b) -> RZ(f) from |0>.
Every formula here is checked against Qiskit ``Statevector`` in the unit tests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qlassifier.errors import GeometryError

__all__ = ["Column", "Encoding", "Geometry", "QubitMap", "bloch_from_angles", "twin_features"]

Column = str | int
Geometry = Mapping[Column | tuple[Column, Column], float | Literal["sphere"]]
Kind = Literal["pair", "circle", "sphere"]
FloatArray = NDArray[np.float64]
# c0 = 2/sqrt(max(d_plain, 8)): below 8 plain columns the plain 2/sqrt(d) would push standardised values past
# pi/2 and fold distant points onto one Bloch vector (ADR 0001 amendment, experiments e13).
MIN_SCALE_DIM = 8


@dataclass(frozen=True)
class QubitMap:
    """One qubit of the encoding.

    Attributes:
        kind: ``"pair"`` (core: one or two plain columns), ``"circle"`` (one periodic column) or
            ``"sphere"`` (a latitude, longitude pair).
        columns: the columns on this qubit, as names (DataFrame input) or positions (array input).
        period: the circle's period in column units (``"circle"`` only).
    """

    kind: Kind
    columns: tuple[Column, ...]
    period: float | None = None


def _label(col: Column) -> str:
    return f"'{col}'" if isinstance(col, str) else str(col)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool | np.bool_)


@dataclass(frozen=True, eq=False)
class Encoding:
    """Qubit plan for a feature matrix: which columns sit on which qubit, and how they become angles.

    Build it with :meth:`plan`, fit the plain-column scaler with :meth:`fit`, then get angles or exact Bloch
    vectors for any rows. Qubit order: circles (declaration order), spheres, then plain pairs.
    """

    qubits: tuple[QubitMap, ...]
    positions: tuple[tuple[int, ...], ...]
    plain: tuple[int, ...]
    scale: float
    mean: FloatArray | None = field(default=None, repr=False)
    sd: FloatArray | None = field(default=None, repr=False)

    @property
    def n_qubits(self) -> int:
        """Number of qubits per circuit."""
        return len(self.qubits)

    @property
    def kinds(self) -> tuple[Kind, ...]:
        """Kind of every qubit, in qubit order."""
        return tuple(q.kind for q in self.qubits)

    @classmethod
    def plan(cls, n_features: int, geometry: Geometry | None, names: Sequence[str] | None) -> Encoding:
        """Validate a geometry declaration and lay columns onto qubits.

        Args:
            n_features: number of columns of X.
            geometry: ``None`` (core encoding for every column) or a mapping (TDD §3.2).
            names: column names when X is a DataFrame with string columns, else ``None``.

        Raises:
            GeometryError: invalid declaration (unknown column, bad period, pair mapped to a number, ...).
        """
        circles: list[tuple[int, float]] = []
        spheres: list[tuple[int, int]] = []
        used: set[int] = set()

        def ident(pos: int) -> Column:
            return names[pos] if names is not None else pos

        def resolve(key: Any) -> int:
            if isinstance(key, str):
                if names is None:
                    raise GeometryError(
                        f"Geometry names column '{key}', but X has no column names.",
                        "Pass a pandas DataFrame or use integer positions.",
                    )
                if key not in names:
                    shown = ", ".join(names[:2]) + (", …" if len(names) > 2 else "")
                    raise GeometryError(
                        f"Geometry names column '{key}', but X has no such column.",
                        f"Use one of: {shown} or integer positions for arrays.",
                    )
                pos = list(names).index(key)
            elif isinstance(key, int | np.integer) and not isinstance(key, bool | np.bool_):
                pos = int(key)
                if not 0 <= pos < n_features:
                    raise GeometryError(
                        f"Geometry names column {pos}, but X has only {n_features} columns.",
                        f"Use positions 0 to {n_features - 1}.",
                    )
            else:
                raise GeometryError(
                    f"Geometry key {key!r} is not a column or a (latitude, longitude) pair.",
                    "Use a column name, a position, or ('lat', 'lon'): 'sphere'.",
                )
            if pos in used:
                raise GeometryError(
                    f"Column {_label(ident(pos))} appears twice in geometry.",
                    "Each column sits on one qubit only.",
                )
            used.add(pos)
            return pos

        if geometry is not None and not isinstance(geometry, Mapping):
            raise GeometryError(
                f"geometry must be a dict of column -> period or (lat, lon) -> 'sphere'; "
                f"got {type(geometry).__name__}.",
                "Use e.g. geometry={'hour': 24} or geometry={('lat', 'lon'): 'sphere'}.",
            )
        for key, value in (geometry or {}).items():
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise GeometryError(
                        f"Geometry key {key!r} is not a column or a (latitude, longitude) pair.",
                        "Use a column name, a position, or ('lat', 'lon'): 'sphere'.",
                    )
                if value != "sphere":
                    raise GeometryError(
                        f"Geometry maps the pair {key!r} to {value!r}.",
                        f"Pairs mean (latitude, longitude): use {key!r}: 'sphere'; "
                        "declare circles one column at a time.",
                    )
                spheres.append((resolve(key[0]), resolve(key[1])))
                continue
            if isinstance(value, str) and value == "sphere":
                raise GeometryError(
                    f"Geometry maps column {_label(key)} to 'sphere'.", "Write {('lat', 'lon'): 'sphere'}."
                )
            if not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0:
                raise GeometryError(
                    f"Geometry period for column {_label(key)} must be a positive number; got {value!r}.",
                    "Use the column's full cycle, e.g. 360 for degrees, 24 for hours.",
                )
            circles.append((resolve(key), float(value)))

        plain = tuple(j for j in range(n_features) if j not in used)
        qubits: list[QubitMap] = []
        positions: list[tuple[int, ...]] = []
        for pos, period in circles:
            qubits.append(QubitMap("circle", (ident(pos),), period))
            positions.append((pos,))
        for lat, lon in spheres:
            qubits.append(QubitMap("sphere", (ident(lat), ident(lon))))
            positions.append((lat, lon))
        for k in range(0, len(plain), 2):
            cols = plain[k : k + 2]
            qubits.append(QubitMap("pair", tuple(ident(j) for j in cols)))
            positions.append(cols)
        scale = 2.0 / math.sqrt(max(len(plain), MIN_SCALE_DIM)) if plain else 1.0
        return cls(tuple(qubits), tuple(positions), plain, scale)

    def fit(self, X: FloatArray) -> Encoding:
        """Fit the plain-column z-score on training rows and check latitudes; return the fitted encoding.

        Raises:
            GeometryError: a latitude column has values outside [-90, 90].
        """
        for q, pos in zip(self.qubits, self.positions, strict=True):
            if q.kind == "sphere":
                lat = X[:, pos[0]]
                bad = lat[(lat < -90.0) | (lat > 90.0)]
                if bad.size:
                    raise GeometryError(
                        f"Latitude column {_label(q.columns[0])} has values outside [-90, 90] "
                        f"(e.g. {float(bad[0]):g}).",
                        "Use degrees, latitude first.",
                    )
        if not self.plain:
            return replace(self, mean=np.zeros(0), sd=np.ones(0))
        cols = X[:, list(self.plain)]
        mean, sd = cols.mean(axis=0), cols.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        return replace(self, mean=mean, sd=sd)

    def zscores(self, X: FloatArray) -> FloatArray:
        """Plain columns z-scored with the training mean and sd, shape (N, d_plain)."""
        if self.mean is None or self.sd is None:
            raise RuntimeError("Encoding.fit must run before zscores.")
        result: FloatArray = (X[:, list(self.plain)] - self.mean) / self.sd
        return result

    def angles(self, X: FloatArray) -> FloatArray:
        """Gate angles per row and qubit, shape (N, n_qubits, 3): (RX angle, RY angle, RZ angle) in radians."""
        N = X.shape[0]
        A = np.zeros((N, self.n_qubits, 3))
        Z = self.zscores(X) * self.scale if self.plain else np.zeros((N, 0))
        k = 0
        for i, (q, pos) in enumerate(zip(self.qubits, self.positions, strict=True)):
            if q.kind == "circle":
                assert q.period is not None
                A[:, i, 1] = 2 * np.pi * X[:, pos[0]] / q.period
            elif q.kind == "sphere":
                A[:, i, 1] = np.deg2rad(90.0 - X[:, pos[0]])
                A[:, i, 2] = np.deg2rad(X[:, pos[1]])
            else:
                A[:, i, 0] = Z[:, k]
                if len(pos) == 2:
                    A[:, i, 1] = Z[:, k + 1]
                k += len(pos)
        return A

    def bloch(self, X: FloatArray) -> FloatArray:
        """Exact Bloch vectors per row and qubit, shape (N, n_qubits, 3) = (<X>, <Y>, <Z>)."""
        return bloch_from_angles(self.angles(X))


def bloch_from_angles(A: FloatArray) -> FloatArray:
    """Bloch vectors of RX(a) -> RY(b) -> RZ(f) applied to |0>, for angles ``A[..., :] = (a, b, f)``.

    r = (cos a sin b cos f + sin a sin f, cos a sin b sin f - sin a cos f, cos a cos b). Pair: f = 0.
    Circle: a = f = 0. Sphere: a = 0.
    """
    a, b, f = A[..., 0], A[..., 1], A[..., 2]
    ca, sa, sb, cf, sf = np.cos(a), np.sin(a), np.sin(b), np.cos(f), np.sin(f)
    return np.stack([ca * sb * cf + sa * sf, ca * sb * sf - sa * cf, ca * np.cos(b)], axis=-1)


def twin_features(X: FloatArray, enc: Encoding) -> FloatArray:
    """Classical twin of geometry mode: the same geometry made by hand (benchmark control, TDD §3.2).

    sin/cos per circle, the 3-D unit vector per sphere, z-scores for plain columns. An RBF on these features has
    the same Gram matrix as geometry mode when every column is angular.
    """
    cols: list[FloatArray] = []
    for q, pos in zip(enc.qubits, enc.positions, strict=True):
        if q.kind == "circle":
            assert q.period is not None
            t = 2 * np.pi * X[:, pos[0]] / q.period
            cols += [np.sin(t), np.cos(t)]
        elif q.kind == "sphere":
            pol, az = np.deg2rad(90.0 - X[:, pos[0]]), np.deg2rad(X[:, pos[1]])
            cols += [np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)]
    if enc.plain:
        cols += list(enc.zscores(X).T)
    return np.stack(cols, axis=1)
