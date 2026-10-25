"""``Qlassifier``: the self-tuning, hardware-ready quantum kernel classifier (TDD §3, §5.2-5.3).

fit: checks -> encoding plan (core or geometry) -> exact features -> self-tuning (0 circuits) -> final features
(exact, or measured: 3 circuits per row) -> Gaussian Gram on Bloch vectors -> SVM + Platt -> ``resources_``.
predict: checks -> angles -> features -> K(X, train) -> decision -> labels / probabilities.
Decision records: ADR 0001 (method), ADR 0002 (tuning and noise handling), ADR 0006 (devices).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from qlassifier import backends
from qlassifier.encoding import Encoding, Geometry, QubitMap
from qlassifier.errors import (
    BACKENDS,
    MAX_TRAIN_ROWS,
    MIN_PER_CLASS,
    MIN_ROWS,
    InvalidParameterError,
    MissingValueError,
    MultiClassError,
    NonNumericError,
    ShotNoiseWarning,
    SingleClassError,
    TooFewSamplesError,
    TooManySamplesError,
    UnknownBackendError,
)
from qlassifier.tuning import (
    CT_GRID,
    M_GRID,
    RIDGE_GRID,
    FoldFeatures,
    cv_kernel_scores,
    gamma0,
    gaussian_gram,
    platt_fit,
    platt_predict,
    select,
)

__all__ = ["Backend", "Qlassifier", "Resources", "fold_features", "tuning_grid"]

Backend = Literal["exact", "shots", "fake_torino", "fake_fez", "fake_brisbane"]
FloatArray = NDArray[np.float64]
N_FOLDS = 5
CIRCUITS_PER_SAMPLE = 3
SHOT_NOISE_LIMIT_PP = 2.0


@dataclass(frozen=True)
class Resources:
    """What the model runs on a device, per circuit and in total (TDD §5.3).

    ``"exact"``: counts are what a device would run; nothing executes. ``"shots"``: sampled from the exact law;
    nothing executes. ``fake_*``: executed on Aer with the device noise model, depth and gates after transpilation.
    """

    backend: str
    qubits: int
    two_qubit_gates: int
    depth: int
    circuits_per_sample: int
    circuits_fit: int
    circuits_predict: int
    tuning_circuits: int
    shots: int | None

    def to_dict(self) -> dict[str, int | str | None]:
        """Plain dict of every field."""
        return dict(asdict(self))


# ----------------------------------------------------------------------------- input checks (typed errors)
def _first_bad(values: NDArray[Any]) -> Any:
    for v in values:
        try:
            float(v)
        except (TypeError, ValueError):
            return v
    return values[0] if len(values) else ""


def _typed_checks(X: Any, method: str) -> None:
    """Raise ``NonNumericError`` / ``MissingValueError`` for 2-D input; leave everything else to scikit-learn."""
    if isinstance(X, pd.DataFrame):
        for col in X.columns:
            dt = X[col].dtype
            if not pd.api.types.is_numeric_dtype(dt) or isinstance(dt, pd.CategoricalDtype):
                raise NonNumericError(col, _first_bad(X[col].dropna().to_numpy()))
        arr = X.to_numpy(dtype=np.float64, na_value=np.nan)
        labels: list[Any] = list(X.columns)
    else:
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            return  # sparse: scikit-learn's own message
        raw = np.asarray(X)
        if raw.ndim != 2 or np.iscomplexobj(raw):
            return
        if raw.dtype.kind in "OUSV":
            for j in range(raw.shape[1]):
                try:
                    raw[:, j].astype(np.float64)
                except ValueError:  # unparseable text; other objects keep numpy's TypeError (scikit-learn convention)
                    raise NonNumericError(j, _first_bad(raw[:, j])) from None
                except TypeError:
                    return
        try:
            arr = raw.astype(np.float64)
        except (TypeError, ValueError):
            return
        labels = list(range(arr.shape[1]))
    if arr.ndim != 2:
        return
    bad = ~np.isfinite(arr)
    if bad.any():
        per_col = bad.sum(axis=0)
        cols = [labels[j] for j in np.flatnonzero(per_col)]
        raise MissingValueError.in_x(int(per_col.sum()), cols, method)


def _check_labels(y: Any) -> None:
    """Raise ``MissingValueError`` for missing labels (1-D or column y); everything else is scikit-learn's."""
    if y is None:
        return
    arr = np.asarray(y.to_numpy() if isinstance(y, pd.Series | pd.DataFrame) else y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim == 1:
        missing = pd.isna(arr)
        if np.any(missing):
            raise MissingValueError.in_y(int(np.sum(missing)))


# ----------------------------------------------------------------------------- tuning helpers (shared with bench)
def tuning_grid(gamma: float | str, C: float | str) -> tuple[tuple[float, float], ...]:
    """Configs searched by self-tuning: the 30-config ridge grid, or one axis when the other is fixed."""
    if gamma == "auto" and C == "auto":
        return RIDGE_GRID
    if gamma == "auto":
        return tuple((m**2, float(C)) for m in M_GRID)
    if C == "auto":
        return tuple((float(gamma), ct / float(gamma)) for ct in CT_GRID)
    return ((float(gamma), float(C)),)


def fold_features(plan: Encoding, Xa: FloatArray, ya: NDArray[Any], Xb: FloatArray, yb: NDArray[Any]) -> FoldFeatures:
    """Exact Bloch features of one CV fold, the plain-column scaler fit on the fold-train rows only."""
    enc = plan.fit(Xa)
    return enc.bloch(Xa), ya, enc.bloch(Xb), yb


def _check_number(name: str, value: Any, example: str) -> None:
    if isinstance(value, str) and value == "auto":
        return
    ok = isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool | np.bool_)
    if not ok or not math.isfinite(float(value)) or float(value) <= 0:
        raise InvalidParameterError(
            f"{name} must be 'auto' or a positive number; got {value!r}.",
            f"Use {name}='auto' to tune it, or e.g. {name}={example}.",
        )


class Qlassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Quantum kernel classifier on Bloch vectors: good by default, hardware-ready, noise-tested.

    Two features per qubit, 0 two-qubit gates, 3 circuits per sample; a Gaussian kernel on the measured Bloch
    vectors and an SVM, tuned on training data only (0 tuning circuits). ``geometry`` puts angle-native columns
    on their own qubit (a periodic column on a circle, a latitude/longitude pair on the sphere).

    Args:
        geometry: angle-native columns, e.g. ``{"hour": 24}`` or ``{("lat", "lon"): "sphere"}``; ``None`` = core
            encoding for every column.
        backend: ``"exact"`` (closed form), ``"shots"`` (ideal sampling), ``"fake_torino"``, ``"fake_fez"`` or
            ``"fake_brisbane"`` (IBM device noise models on Aer).
        shots: shots per circuit; ignored by ``"exact"``.
        gamma: kernel sharpness relative to the data scale (gamma * gamma0_); ``"auto"`` = tuned.
        C: SVM regularisation; ``"auto"`` = tuned on the ridge C = C~ / gamma.
        random_state: int for shuffled CV folds and seeded shots / sampler; ``None`` = unshuffled folds, fresh
            shot randomness.

    Example:
        >>> from qlassifier import Qlassifier, load_dataset
        >>> X_train, X_test, y_train, y_test = load_dataset("breast_cancer").split(seed=0)
        >>> clf = Qlassifier().fit(X_train, y_train)
        >>> round(clf.score(X_test, y_test), 2) > 0.9
        True
    """

    def __init__(
        self,
        geometry: Geometry | None = None,
        backend: Backend = "exact",
        shots: int = 1024,
        gamma: float | Literal["auto"] = "auto",
        C: float | Literal["auto"] = "auto",
        random_state: int | None = None,
    ) -> None:
        self.geometry = geometry
        self.backend = backend
        self.shots = shots
        self.gamma = gamma
        self.C = C
        self.random_state = random_state

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = False
        tags.non_deterministic = self.backend != "exact" and self.random_state is None
        return tags

    # ------------------------------------------------------------------------- parameters and seeds
    def _check_params(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in BACKENDS:
            raise UnknownBackendError(self.backend)
        shots_ok = isinstance(self.shots, int | np.integer) and not isinstance(self.shots, bool | np.bool_)
        if not shots_ok or int(self.shots) <= 0:
            raise InvalidParameterError(
                f"shots must be a positive integer; got {self.shots!r}.", "Use e.g. shots=1024."
            )
        _check_number("gamma", self.gamma, "1.0")
        _check_number("C", self.C, "10.0")
        rs = self.random_state
        if rs is not None and (not isinstance(rs, int | np.integer) or isinstance(rs, bool | np.bool_)):
            raise InvalidParameterError(
                f"random_state must be an integer or None; got {rs!r}.", "Use e.g. random_state=0."
            )

    def _streams(self) -> list[np.random.SeedSequence]:
        """Separate fixed streams for fit, predict and the shot-noise guard (fresh when random_state is None)."""
        root = np.random.SeedSequence(None if self.random_state is None else int(self.random_state))
        return root.spawn(3)

    def _folds(self, X: FloatArray, y01: NDArray[np.int64]) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        if self.random_state is None:
            skf = StratifiedKFold(N_FOLDS)
        else:
            skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=int(self.random_state))
        return [(np.asarray(a), np.asarray(b)) for a, b in skf.split(X, y01)]

    # ------------------------------------------------------------------------- fit
    def fit(self, X: Any, y: Any) -> Qlassifier:
        """Self-tune on the training data (0 circuits), then fit the SVM on the final Bloch features.

        Args:
            X: numeric array-like or DataFrame, shape (n_samples, n_features).
            y: two class labels of any type.

        Returns:
            self.

        Raises:
            MissingValueError, NonNumericError, SingleClassError, MultiClassError, TooFewSamplesError,
            TooManySamplesError, GeometryError, UnknownBackendError, TooManyQubitsError, InvalidParameterError.
        """
        self._check_params()
        _check_labels(y)
        _typed_checks(X, "fit")
        Xa, y_checked = validate_data(self, X, y, reset=True, dtype=np.float64)
        check_classification_targets(y_checked)
        classes, y01 = np.unique(y_checked, return_inverse=True)
        if len(classes) == 1:
            raise SingleClassError(classes[0])
        if len(classes) > 2:
            raise MultiClassError(list(classes))  # "Only binary classification is supported." (sklearn convention)
        y01 = y01.astype(np.int64)
        counts = np.bincount(y01, minlength=2)
        if len(y01) < MIN_ROWS or counts.min() < MIN_PER_CLASS:
            raise TooFewSamplesError(len(y01), {classes[k]: int(counts[k]) for k in range(2)})
        if len(y01) > MAX_TRAIN_ROWS:
            raise TooManySamplesError(len(y01))
        names = [str(c) for c in self.feature_names_in_] if hasattr(self, "feature_names_in_") else None
        plan = Encoding.plan(Xa.shape[1], self.geometry, names)
        enc = plan.fit(Xa)
        tmpl = backends.templates(enc)
        if self.backend in backends.DEVICES:
            backends.check_qubits(enc.n_qubits, self.backend)

        folds = self._folds(Xa, y01)
        grid = tuning_grid(self.gamma, self.C)
        if len(grid) > 1:
            cv_folds = [fold_features(plan, Xa[a], y01[a], Xa[b], y01[b]) for a, b in folds]
            scores = cv_kernel_scores(cv_folds, grid)
            best = select(scores)
            mean = scores.mean(axis=1)
            order = np.argsort(-mean, kind="stable")
            rank = np.empty(len(grid), dtype=np.int64)
            rank[order] = np.arange(1, len(grid) + 1)
            for i in range(1, len(grid)):  # equal means share the better rank (scikit-learn "min" ranking)
                if mean[order[i]] == mean[order[i - 1]]:
                    rank[order[i]] = rank[order[i - 1]]
            self.cv_results_: dict[str, Any] = {
                "params": [{"gamma": g, "C": c} for g, c in grid],
                "mean_test_score": mean,
                "std_test_score": scores.std(axis=1),
                "rank_test_score": rank,
            }
        else:
            best = 0
            self.cv_results_ = {}
        g, C = grid[best]
        fit_seq, _, guard_seq = self._streams()

        F_exact = enc.bloch(Xa)
        self._transpiled: dict[str, Any] | None = None
        self._per_qubit: list[dict[str, Any]] | None = None
        two_q, depth = 0, backends.logical_depth(tmpl)
        if self.backend == "exact":
            F = F_exact
        elif self.backend == "shots":
            F = backends.sample_shots(F_exact, int(self.shots), np.random.default_rng(fit_seq))
        else:
            transpiled = backends.transpile_pinned(backends.bloch_circuits(tmpl), self.backend)
            self._transpiled = transpiled  # whole-register circuits: what a device runs (resources_)
            if backends.per_qubit_exact(transpiled, self.backend):  # same law at linear cost (ADR 0006)
                layout = backends.pinned_layout(transpiled, enc.n_qubits)
                self._per_qubit = backends.per_qubit_circuits(tmpl, self.backend, layout)
            stats = [backends.circuit_stats(tc) for tc in transpiled.values()]
            two_q, depth = max(s[0] for s in stats), max(s[1] for s in stats)
            seed = int(np.random.default_rng(fit_seq).integers(2**31 - 1))
            F = self._measure(enc.angles(Xa), seed)

        self.gamma0_ = gamma0(F)
        K = gaussian_gram(F, F, g * self.gamma0_)
        self._svc = SVC(C=C, kernel="precomputed").fit(K, y01)
        oof = np.zeros(len(y01))
        for a, b in folds:
            m = SVC(C=C, kernel="precomputed").fit(K[np.ix_(a, a)], y01[a])
            oof[b] = m.decision_function(K[np.ix_(b, a)])
        self._platt = platt_fit(oof, y01)

        if self.backend != "exact":
            self._shot_noise_guard(F_exact, y01, folds, g, C, np.random.default_rng(guard_seq))

        self.classes_ = classes
        self._enc = enc
        self._F = F
        self.encoding_: tuple[QubitMap, ...] = enc.qubits
        self.n_qubits_ = enc.n_qubits
        self.best_params_ = {"gamma": g, "C": C}
        self._resources = Resources(
            backend=self.backend,
            qubits=enc.n_qubits,
            two_qubit_gates=two_q,
            depth=depth,
            circuits_per_sample=CIRCUITS_PER_SAMPLE,
            circuits_fit=CIRCUITS_PER_SAMPLE * len(y01),
            circuits_predict=0,
            tuning_circuits=0,
            shots=None if self.backend == "exact" else int(self.shots),
        )
        self._predicted = [0]  # rows predicted so far (mutated in place: predict leaves __dict__ unchanged)
        return self

    def _shot_noise_guard(
        self,
        F_exact: FloatArray,
        y01: NDArray[np.int64],
        folds: Sequence[tuple[NDArray[np.intp], NDArray[np.intp]]],
        g: float,
        C: float,
        rng: np.random.Generator,
    ) -> None:
        """Simulate shot noise on the exact training features (0 circuits); warn when CV BA drops > 2 pp."""
        F_noisy = backends.sample_shots(F_exact, int(self.shots), rng)
        grid = ((g, C),)
        exact = cv_kernel_scores([(F_exact[a], y01[a], F_exact[b], y01[b]) for a, b in folds], grid).mean()
        noisy = cv_kernel_scores([(F_noisy[a], y01[a], F_noisy[b], y01[b]) for a, b in folds], grid).mean()
        loss_pp = 100.0 * float(exact - noisy)
        if loss_pp > SHOT_NOISE_LIMIT_PP:
            warnings.warn(ShotNoiseWarning(int(self.shots), loss_pp), stacklevel=3)

    # ------------------------------------------------------------------------- predict
    @property
    def resources_(self) -> Resources:
        """Qubits, two-qubit gates, depth, circuits run (fit and cumulative predict) and shots."""
        try:
            base: Resources = self.__dict__["_resources"]
        except KeyError:
            raise AttributeError("resources_ is set by fit.") from None
        return replace(base, circuits_predict=CIRCUITS_PER_SAMPLE * self._predicted[0])

    def _features(self, X: Any, method: str) -> FloatArray:
        check_is_fitted(self)
        _typed_checks(X, method)
        Xa = validate_data(self, X, reset=False, dtype=np.float64)
        enc: Encoding = self._enc
        _, predict_seq, _ = self._streams()
        if self.backend == "exact":
            F = enc.bloch(Xa)
        elif self.backend == "shots":
            F = backends.sample_shots(enc.bloch(Xa), int(self.shots), np.random.default_rng(predict_seq))
        else:
            seed = int(np.random.default_rng(predict_seq).integers(2**31 - 1))
            F = self._measure(enc.angles(Xa), seed)
        self._predicted[0] += len(Xa)
        return F

    def _measure(self, angles: FloatArray, seed: int) -> FloatArray:
        """Measured Bloch vectors on the fake device: qubit by qubit when that is exact, else the whole register."""
        if self._per_qubit is not None:
            return backends.measure_bloch_per_qubit(self._per_qubit, angles, self.backend, int(self.shots), seed)
        assert self._transpiled is not None
        return backends.measure_bloch(self._transpiled, angles, self.backend, int(self.shots), seed)

    def _kernel(self, F: FloatArray) -> FloatArray:
        return gaussian_gram(F, self._F, self.best_params_["gamma"] * self.gamma0_)

    def decision_function(self, X: Any) -> FloatArray:
        """SVM margin per row; > 0 means ``classes_[1]``. Used for ROC-AUC."""
        K = self._kernel(self._features(X, "decision_function"))
        out: FloatArray = self._svc.decision_function(K)
        return out

    def predict(self, X: Any) -> NDArray[Any]:
        """Predicted labels (same type as the training labels)."""
        K = self._kernel(self._features(X, "predict"))
        pred = self._svc.predict(K)
        out: NDArray[Any] = self.classes_[np.asarray(pred, dtype=np.int64)]
        return out

    def predict_proba(self, X: Any) -> FloatArray:
        """Class probabilities, shape (n, 2), columns in ``classes_`` order (Platt sigmoid on the margin)."""
        K = self._kernel(self._features(X, "predict_proba"))
        d = self._svc.decision_function(K)
        p1 = platt_predict(d, *self._platt)
        return np.column_stack([1.0 - p1, p1])
