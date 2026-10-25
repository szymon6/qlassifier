"""The fair protocol as code (TDD §7, ADR 0004): splits, per-fold preprocessing, equal-budget tuning, tasks.

One task = (dataset, seed, model) -> ``RunRecord``. A task re-derives its split, inner folds and per-fold
preprocessing from (dataset, seed) alone, so numbers never depend on scheduling. Same split, same features and
the same 5-fold CV for every model; 30 configs per tuned model (pinned ZZ: 4); ties -> first config in grid order.
Phase-7 reference: ``experiments/classifier/common/lab.py`` (``run_seed``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from qlassifier.bench.models import CLASSICAL, MODEL_IDS, ModelSpec, describe, model_specs, on_edge
from qlassifier.bench.result import Published, RunRecord
from qlassifier.datasets import TEST_SIZE, SplitMode, load_dataset, outer_split, split_mode
from qlassifier.errors import UnknownModelError

__all__ = [
    "CI_LEVEL",
    "N_FOLDS",
    "PCA_COMPONENTS",
    "PUBLISHED",
    "Prep",
    "Task",
    "fingerprint",
    "fold_inputs",
    "inner_folds",
    "protocol_models",
    "run_model",
]

FloatArray = NDArray[np.float64]
PROTOCOL_VERSION = "1"
N_FOLDS = 5
PCA_COMPONENTS = 8
CI_LEVEL = 0.95
# Best published quantum result per UCI dataset (phase 2, cross-protocol context; BT-3 / BT-4).
PUBLISHED: dict[str, Published] = {
    "breast_cancer": Published(value=0.968, metric="balanced_accuracy", source="arXiv:2604.18837", method="QKT"),
    "ionosphere": Published(value=0.911, metric="balanced_accuracy", source="arXiv:2604.18837", method="QKT"),
    "sonar": Published(value=0.813, metric="balanced_accuracy", source="arXiv:2604.18837", method="Rot2DoF"),
    "heart": Published(value=0.804, metric="balanced_accuracy", source="arXiv:2604.18837", method="QKT"),
}


@dataclass(frozen=True)
class Task:
    """One unit of work: a (seed, model) run, the cost study, or one noise-study cell."""

    kind: Literal["run", "cost", "noise"]
    dataset: str
    seed: int | None
    model: str
    device: str | None = None
    part: Literal["A", "B"] | None = None
    weight: float = 1.0  # expected seconds; longest first


# ----------------------------------------------------------------------------- data and preprocessing
@dataclass(frozen=True)
class Arrays:
    """A dataset as arrays, with its recipe and declared geometry (column position -> period)."""

    X: FloatArray
    y: NDArray[np.int64]
    groups: NDArray[Any] | None
    recipe: Literal["pca8", "native"]
    geometry: dict[int, float] | None

    @property
    def d_used(self) -> int:
        """Model input features: 8 for ``pca8`` (fewer when d <= 8), every column for ``native``."""
        d = int(self.X.shape[1])
        return min(PCA_COMPONENTS, d) if self.recipe == "pca8" else d


@cache
def arrays(name: str) -> Arrays:
    """Load a bundled dataset once per process."""
    d = load_dataset(name)
    geometry = None if d.geometry is None else {d.X.columns.get_loc(c): float(p) for c, p in d.geometry.items()}
    return Arrays(
        X=d.X.to_numpy(dtype=np.float64),
        y=d.y.to_numpy(dtype=np.int64),
        groups=None if d.groups is None else d.groups.to_numpy(),
        recipe=d.info.recipe,
        geometry=geometry,
    )


class Prep:
    """Shared preprocessing, fit on training rows only (§7.3).

    ``pca8``: StandardScaler -> PCA(8, random_state=0) (skipped when d <= 8) -> StandardScaler.
    ``native``: identity (raw columns); every model scales internally.
    """

    def __init__(self, recipe: str) -> None:
        self.recipe = recipe

    def fit(self, X: FloatArray) -> Prep:
        """Fit on training rows."""
        if self.recipe == "native":
            return self
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.s1 = StandardScaler().fit(X)
        Z = self.s1.transform(X)
        self.pca = PCA(PCA_COMPONENTS, random_state=0).fit(Z) if Z.shape[1] > PCA_COMPONENTS else None
        if self.pca is not None:
            Z = self.pca.transform(Z)
        self.s2 = StandardScaler().fit(Z)
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        """Transform any rows with the fitted steps."""
        if self.recipe == "native":
            return np.asarray(X, dtype=np.float64)
        Z = self.s1.transform(X)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        out: FloatArray = self.s2.transform(Z)
        return out


def inner_folds(
    y_tr: NDArray[Any], groups_tr: NDArray[Any] | None, seed: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """5 inner folds of a training split: stratified, or stratified-grouped for grouped data (§7.2)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    X0 = np.zeros((len(y_tr), 1))
    if groups_tr is None:
        it = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X0, y_tr)
    else:
        it = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X0, y_tr, groups_tr)
    return [(np.asarray(a, dtype=np.intp), np.asarray(b, dtype=np.intp)) for a, b in it]


@dataclass(frozen=True)
class FoldInputs:
    """Everything a model needs for one (dataset, seed): prepped CV folds and the prepped outer split."""

    folds: list[tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64]]]
    X_train: FloatArray
    y_train: NDArray[np.int64]
    X_test: FloatArray
    y_test: NDArray[np.int64]
    raw_folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]]


@cache
def fold_inputs(name: str, seed: int, mode: SplitMode) -> FoldInputs:
    """Outer split, inner folds and per-fold preprocessing for one seed (cached per process)."""
    a = arrays(name)
    tr, te = outer_split(a.y, a.groups, seed, mode)
    Xtr_raw, Xte_raw, ytr, yte = a.X[tr], a.X[te], a.y[tr], a.y[te]
    raw_folds = inner_folds(ytr, None if a.groups is None else a.groups[tr], seed)
    folds = []
    for ia, ib in raw_folds:
        p = Prep(a.recipe).fit(Xtr_raw[ia])
        folds.append((p.transform(Xtr_raw[ia]), ytr[ia], p.transform(Xtr_raw[ib]), ytr[ib]))
    p = Prep(a.recipe).fit(Xtr_raw)
    return FoldInputs(folds, p.transform(Xtr_raw), ytr, p.transform(Xte_raw), yte, raw_folds)


def protocol_models(name: str, models: Sequence[str] | None = None) -> list[ModelSpec]:
    """Model specs of the protocol for a dataset, in leaderboard order (optionally a validated subset)."""
    a = arrays(name)
    specs = model_specs(a.d_used, a.geometry)
    ids = [s.id for s in specs]
    if models is None:
        return specs
    chosen = []
    for m in models:
        if m not in ids:
            raise UnknownModelError(m, ids)
    for s in specs:
        if s.id in models:
            chosen.append(s)
    return chosen


def _stable(value: Any) -> Any:
    """A grid value rendered for hashing: floats at 12 significant digits, so a 1-ulp difference between CPUs
    (numpy reductions, libm ``pow`` behind ``logspace``) does not change the fingerprint."""
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return f"{float(value):.12g}"
    if isinstance(value, tuple | list):
        return [_stable(v) for v in value]
    return repr(value)


def fingerprint(name: str, seeds: Sequence[int], models: Sequence[str], mode: SplitMode) -> str:
    """SHA-256 of everything that fixes the numbers: grids, models, seeds, recipe and split rule.

    Grid floats enter at 12 significant digits (``_stable``): the fingerprint identifies the protocol, not the
    CPU that evaluated it.
    """
    a = arrays(name)
    specs = {s.id: s for s in model_specs(a.d_used, a.geometry)}
    doc = {
        "protocol": PROTOCOL_VERSION,
        "dataset": name,
        "recipe": a.recipe,
        "split": mode,
        "test_size": TEST_SIZE,
        "cv_folds": N_FOLDS,
        "pca_components": PCA_COMPONENTS,
        "seeds": [int(s) for s in seeds],
        "models": {m: [{k: _stable(v) for k, v in sorted(cfg.items())} for cfg in specs[m].grid] for m in models},
        "geometry": None if a.geometry is None else sorted(a.geometry.items()),
        "noise": {"shots": 1024, "optimization_level": 3, "sampler_seed": 11, "transpiler_seed": 0},
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def test_train_ratio(name: str, seeds: Sequence[int], mode: SplitMode) -> float:
    """Mean realised n_test / n_train over the seeds' outer splits (labels and groups only)."""
    a = arrays(name)
    ratios = []
    for s in seeds:
        tr, te = outer_split(a.y, a.groups, int(s), mode)
        ratios.append(len(te) / len(tr))
    return float(np.mean(ratios))


# ----------------------------------------------------------------------------- one (seed, model) run
def _prepare(spec: ModelSpec, A: FloatArray, B: FloatArray, recipe: str) -> tuple[FloatArray, FloatArray]:
    """Model-internal scaling under ``native``: StandardScaler fit on training rows, unless the model reads raw."""
    if recipe != "native" or spec.kind == "trivial" or spec.raw:
        return A, B
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(A)
    return sc.transform(A), sc.transform(B)


def _grid_scores(
    spec: ModelSpec, A: FloatArray, ya: NDArray[Any], B: FloatArray, yb: NDArray[Any], recipe: str
) -> FloatArray:
    """Balanced accuracy of every config on one fold (kernels: one Gram per bandwidth, C swept on it)."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.svm import SVC

    A, B = _prepare(spec, A, B, recipe)
    scores = np.zeros(len(spec.grid))
    if spec.kind == "kernel":
        assert spec.kernel is not None
        ctx: dict[Any, Any] = {}
        current: tuple[Any, FloatArray, FloatArray] | None = (
            None  # one Gram pair at a time (bandwidths are consecutive)
        )
        for i, cfg in enumerate(spec.grid):
            bw = cfg["bw"]
            if current is None or current[0] != bw:
                current = (bw, *spec.kernel(A, B, bw, ctx))
            _, Kaa, Kba = current
            m = SVC(C=cfg["C"], kernel="precomputed").fit(Kaa, ya)
            scores[i] = balanced_accuracy_score(yb, m.predict(Kba))
    else:
        assert spec.make is not None
        for i, cfg in enumerate(spec.grid):
            est = spec.make(cfg).fit(A, ya)
            scores[i] = balanced_accuracy_score(yb, est.predict(B))
    return scores


def qlassifier_scores(
    folds: Sequence[tuple[FloatArray, NDArray[Any], FloatArray, NDArray[Any]]],
    d: int,
    geometry: dict[int, float] | None,
    grid: Sequence[tuple[float, float]],
) -> FloatArray:
    """Qlassifier's own tuning routine (same as ``fit``) on the shared per-fold features, (n_configs, n_folds)."""
    from qlassifier.encoding import Encoding, Geometry
    from qlassifier.model import fold_features
    from qlassifier.tuning import cv_kernel_scores

    plan = Encoding.plan(d, cast(Geometry | None, geometry), None)
    return cv_kernel_scores([fold_features(plan, A, ya, B, yb) for A, ya, B, yb in folds], grid)


def _metrics(y: NDArray[Any], pred: NDArray[Any], score: NDArray[Any]) -> dict[str, float | None]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["roc_auc"] = None
    return out


def run_model(name: str, seed: int, model: str, mode: SplitMode) -> RunRecord:
    """Tune one model on the inner folds of one seed, refit on the training split, evaluate once on the test split."""
    from sklearn.svm import SVC

    a = arrays(name)
    fi = fold_inputs(name, seed, mode)
    spec = next(s for s in protocol_models(name) if s.id == model)
    t0 = time.perf_counter()
    if len(spec.grid) > 1:
        if spec.kind == "qlassifier":
            grid = [(float(c["bw"]), float(c["C"])) for c in spec.grid]
            scores = qlassifier_scores(fi.folds, a.d_used, a.geometry, grid)
            cv = np.zeros(len(spec.grid))
            for f in range(scores.shape[1]):
                cv += scores[:, f]
        else:
            cv = np.zeros(len(spec.grid))
            for A, ya, B, yb in fi.folds:
                cv += _grid_scores(spec, A, ya, B, yb, a.recipe)
        cv /= len(fi.folds)
        best = int(np.argmax(cv))
        cv_score: float | None = float(cv[best])
    else:
        best, cv_score = 0, None
    t_tune = time.perf_counter() - t0
    cfg = spec.grid[best]
    Xtr, Xte = _prepare(spec, fi.X_train, fi.X_test, a.recipe)
    t1 = time.perf_counter()
    if spec.kind == "qlassifier":
        from qlassifier.encoding import Geometry
        from qlassifier.model import Qlassifier

        clf = Qlassifier(
            geometry=cast(Geometry | None, a.geometry), gamma=float(cfg["bw"]), C=float(cfg["C"]), random_state=seed
        )
        clf.fit(Xtr, fi.y_train)
        t2 = time.perf_counter()
        pred, score = clf.predict(Xte), clf.decision_function(Xte)
    elif spec.kind == "kernel":
        assert spec.kernel is not None
        Ktr, Kte = spec.kernel(Xtr, Xte, cfg["bw"], {})
        svc = SVC(C=cfg["C"], kernel="precomputed").fit(Ktr, fi.y_train)
        t2 = time.perf_counter()
        pred, score = svc.predict(Kte), svc.decision_function(Kte)
    else:
        assert spec.make is not None
        est = spec.make(cfg).fit(Xtr, fi.y_train)
        t2 = time.perf_counter()
        pred = est.predict(Xte)
        score = est.predict_proba(Xte)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Xte)
    t3 = time.perf_counter()
    m = _metrics(fi.y_test, pred, score)
    acc, ba, f1 = m["accuracy"], m["balanced_accuracy"], m["f1"]
    assert acc is not None
    assert ba is not None
    assert f1 is not None
    return RunRecord(
        seed=seed,
        model=model,
        config=describe(model, cfg),
        config_index=best,
        cv_score=cv_score,
        edge=bool(len(spec.grid) > 1 and on_edge(spec.grid, best)),
        balanced_accuracy=ba,
        accuracy=acc,
        roc_auc=m["roc_auc"],
        f1=f1,
        t_tune=t_tune,
        t_fit=t2 - t1,
        t_predict=t3 - t2,
        n_train=len(fi.y_train),
        n_test=len(fi.y_test),
    )


def decide_split(name: str, seeds: Sequence[int]) -> SplitMode:
    """The outer split rule for these seeds, from labels and groups only (drift rule, §7.2)."""
    a = arrays(name)
    return split_mode(a.y, a.groups, list(seeds))


def classical_ids() -> tuple[str, ...]:
    """Ids of the five classical baselines."""
    return CLASSICAL


def all_model_ids() -> tuple[str, ...]:
    """Every model id the registry knows, in leaderboard order."""
    return MODEL_IDS
