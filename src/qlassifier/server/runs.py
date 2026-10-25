"""Run lifecycle for the dashboard (TDD §6.2-6.4, ADR 0007).

One active run at a time, executed on one worker thread (``benchmark(progress=..., cancel=...)``; joblib processes
inside). Every run keeps an in-memory event log (incrementing ids) so a reconnecting client replays missed events
with ``Last-Event-ID``. The log of the latest run stays until the next run starts; older runs keep their terminal
event and final result. Completed results are the session cache per dataset; nothing is written to disk.
"""

from __future__ import annotations

import secrets
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from qlassifier.bench import ProgressEvent
from qlassifier.bench.result import BenchmarkResult
from qlassifier.errors import RunInProgressError, RunNotFoundError, error_body
from qlassifier.server.models import ErrorBody, FailedData, ProgressData, Run, RunStatus, TerminalData

__all__ = ["RunManager", "Runner"]

Runner = Callable[..., BenchmarkResult]
PARTIAL_INTERVAL_S = 0.3
TERMINAL = ("complete", "cancelled", "failed")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _RunState:
    id: str
    dataset: str
    seeds: list[int]
    total: int
    n_models: int
    started_at: str = field(default_factory=_now)
    status: RunStatus = "running"
    done: int = 0
    eta_s: float | None = None
    finished_at: str | None = None
    error: ErrorBody | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    events: list[tuple[int, str, str]] = field(default_factory=list)
    next_id: int = 1
    result_json: str | None = None
    last_partial: float = 0.0
    per_seed: dict[int, int] = field(default_factory=dict)
    thread: threading.Thread | None = None

    def view(self) -> Run:
        return Run(
            id=self.id,
            dataset=self.dataset,
            seeds=self.seeds,
            status=self.status,
            done=self.done,
            total=self.total,
            eta_s=self.eta_s,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
        )


class RunManager:
    """Start, follow, cancel and fetch dashboard runs; thread-safe."""

    def __init__(self, run_seeds: int = 10, n_jobs: int | None = None, runner: Runner | None = None) -> None:
        self.run_seeds = run_seeds
        self._n_jobs = n_jobs
        self._runner = runner
        self._cond = threading.Condition()
        self._runs: dict[str, _RunState] = {}
        self._latest: _RunState | None = None
        self._session: dict[str, str] = {}

    # ------------------------------------------------------------------------- queries
    def active(self) -> Run | None:
        """The running (or cancelling) run, if any."""
        with self._cond:
            st = self._latest
            return st.view() if st is not None and st.status not in TERMINAL else None

    def get(self, run_id: str) -> Run:
        """The run's current state."""
        with self._cond:
            return self._state(run_id).view()

    def result_json(self, run_id: str) -> str:
        """The latest (partial or final) result of a run, as JSON."""
        with self._cond:
            st = self._state(run_id)
            if st.result_json is None:
                st.result_json = self._empty_result(st).model_dump_json()
            return st.result_json

    def session_result(self, dataset: str) -> str | None:
        """The latest completed run result of this session for a dataset, as JSON."""
        with self._cond:
            return self._session.get(dataset)

    def events_after(self, run_id: str, last_id: int) -> tuple[list[tuple[int, str, str]], bool]:
        """Events with id > ``last_id`` and whether the run has ended (its terminal event is in the log)."""
        with self._cond:
            st = self._state(run_id)
            evs = [e for e in st.events if e[0] > last_id]
            return evs, st.status in TERMINAL and any(e[1] in TERMINAL for e in st.events)

    def wait(self, timeout: float) -> None:
        """Block until any run changes (or ``timeout`` passes)."""
        with self._cond:
            self._cond.wait(timeout)

    def _state(self, run_id: str) -> _RunState:
        st = self._runs.get(run_id)
        if st is None:
            raise RunNotFoundError(run_id)
        return st

    # ------------------------------------------------------------------------- commands
    def start(self, dataset: str, seeds: int | None = None) -> Run:
        """Start a run (raises ``RunInProgressError`` while another one is active)."""
        from qlassifier.bench.protocol import protocol_models
        from qlassifier.errors import DATASET_IDS, UnknownDatasetError

        if dataset not in DATASET_IDS:
            raise UnknownDatasetError(dataset)
        n = seeds if seeds is not None else self.run_seeds
        n_models = len(protocol_models(dataset))
        with self._cond:
            cur = self._latest
            if cur is not None and cur.status not in TERMINAL:
                pct = round(100 * cur.done / cur.total) if cur.total else 0
                raise RunInProgressError(cur.dataset, pct)
            if cur is not None:  # the previous log is dropped; its terminal event and result stay
                cur.events = [e for e in cur.events if e[1] in TERMINAL]
            st = _RunState(
                id=f"r-{secrets.token_hex(3)}",
                dataset=dataset,
                seeds=list(range(n)),
                total=n * n_models + 1,
                n_models=n_models,
            )
            self._runs[st.id] = st
            self._latest = st
            st.thread = threading.Thread(target=self._work, args=(st,), name=f"qlassifier-run-{st.id}", daemon=True)
            st.thread.start()
            return st.view()

    def cancel(self, run_id: str) -> tuple[Run, bool]:
        """Ask a run to stop after its running tasks; returns (run, changed)."""
        with self._cond:
            st = self._state(run_id)
            if st.status != "running":
                return st.view(), False
            st.status = "cancelling"
            st.cancel.set()
            self._cond.notify_all()
            return st.view(), True

    def shutdown(self, timeout: float = 30.0) -> None:
        """Cancel the active run and wait for its thread (server stop / Ctrl-C)."""
        with self._cond:
            st = self._latest
            if st is None or st.status in TERMINAL:
                return
            st.status = "cancelling"
            st.cancel.set()
            thread = st.thread
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------------- worker
    def _emit(self, st: _RunState, name: str, data: str) -> None:
        st.events.append((st.next_id, name, data))
        st.next_id += 1
        self._cond.notify_all()

    def _empty_result(self, st: _RunState) -> BenchmarkResult:
        from qlassifier.bench.aggregate import build_result
        from qlassifier.bench.protocol import arrays, decide_split, fingerprint, protocol_models, test_train_ratio
        from qlassifier.datasets import list_datasets

        specs = protocol_models(st.dataset)
        mode = decide_split(st.dataset, st.seeds)
        return build_result(
            info=next(i for i in list_datasets() if i.id == st.dataset),
            seeds=st.seeds,
            specs=specs,
            mode=mode,
            ratio=test_train_ratio(st.dataset, st.seeds, mode),
            fingerprint=fingerprint(st.dataset, st.seeds, [s.id for s in specs], mode),
            n_features_used=arrays(st.dataset).d_used,
            runs=[],
            cost=None,
            noise_records=[],
            status="partial",
            wall_s=0.0,
            n_jobs=0,
        )

    def _on_progress(self, st: _RunState, ev: ProgressEvent) -> None:
        now = time.monotonic()
        with self._cond:
            st.done, st.total, st.eta_s = ev.done, ev.total, ev.eta_s
            st.result_json = ev.result.model_dump_json()
            seed_done = False
            if ev.seed is not None and ev.model != "cost":
                st.per_seed[ev.seed] = st.per_seed.get(ev.seed, 0) + 1
                seed_done = st.per_seed[ev.seed] == st.n_models
            progress = ProgressData(
                run_id=st.id, done=ev.done, total=ev.total, seed=ev.seed, model=ev.model, eta_s=ev.eta_s
            )
            self._emit(st, "progress", progress.model_dump_json())
            if now - st.last_partial >= PARTIAL_INTERVAL_S or seed_done or ev.done == ev.total:
                partial = f'{{"run_id":"{st.id}","result":{st.result_json}}}'
                self._emit(st, "partial", partial)
                st.last_partial = now

    def _work(self, st: _RunState) -> None:
        runner = self._runner
        if runner is None:
            from qlassifier.bench import benchmark

            runner = benchmark
        try:
            result = runner(
                st.dataset,
                seeds=st.seeds,
                n_jobs=self._n_jobs,
                progress=lambda ev: self._on_progress(st, ev),
                cancel=st.cancel,
            )
        except Exception as exc:  # any failure becomes a typed `failed` event; the trace goes to the server log
            traceback.print_exc(file=sys.stderr)
            with self._cond:
                body = ErrorBody(**error_body(exc))
                st.status, st.error, st.finished_at = "failed", body, _now()
                self._emit(st, "failed", FailedData(run_id=st.id, error=body).model_dump_json())
            return
        with self._cond:
            st.result_json = result.model_dump_json()
            st.done = st.done if result.status == "cancelled" else st.total
            st.status = "complete" if result.status == "complete" else "cancelled"
            st.finished_at = _now()
            st.eta_s = 0.0 if st.status == "complete" else None
            if st.status == "complete":
                self._session[st.dataset] = st.result_json
            url = f"/api/v1/runs/{st.id}/result"
            self._emit(st, st.status, TerminalData(run_id=st.id, result_url=url).model_dump_json())
