"""Command line: ``qlassifier ui | benchmark | --version`` (TDD §5.7, §10.2).

Exit codes: 0 ok; 1 run error (typed message on stderr); 2 usage error; 130 interrupted (partial results written,
status "cancelled"). Heavy imports load lazily so ``--version`` and ``ui`` start fast.
"""

from __future__ import annotations

import argparse
import platform
import signal
import socket
import sys
import threading
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from types import FrameType
from typing import Any

from qlassifier.errors import DATASET_IDS, QlassifierError

__all__ = ["main"]

DEFAULT_PORT = 8765
PORT_SPAN = 20
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


def version_line() -> str:
    """``qlassifier 0.1.0 (python ..., qiskit ..., ...)``."""
    parts = [f"python {platform.python_version()}"]
    for dist in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "scikit-learn", "numpy"):
        parts.append(f"{dist} {version(dist)}")
    return f"qlassifier {version('qlassifier')} ({', '.join(parts)})"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="qlassifier", description="Qlassifier: the quantum classifier built to win, with its proof inside."
    )
    ap.add_argument("--version", action="store_true", help="print versions and exit")
    sub = ap.add_subparsers(dest="command", metavar="{ui,benchmark}")
    ui = sub.add_parser("ui", help="open the dashboard (local server on 127.0.0.1)")
    ui.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (next free one if busy; default 8765)")
    ui.add_argument("--seeds", type=int, default=10, help="seeds of dashboard runs (2-20, default 10)")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ui.add_argument(
        "--results",
        type=Path,
        default=Path("qlassifier-benchmark"),
        help="folder of results saved from code to show too (default: qlassifier-benchmark, as benchmark --out)",
    )
    bench = sub.add_parser("benchmark", help="rerun the leaderboard under the fair protocol")
    bench.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help=f"bundled ids (default: all: {', '.join(DATASET_IDS)}) or CSV files",
    )
    bench.add_argument("--target", default=None, help="target column of a CSV dataset (default: its last column)")
    bench.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 (default 10)")
    bench.add_argument("--noise", action="store_true", help="add the noise study (slow)")
    bench.add_argument("--out", type=Path, default=Path("qlassifier-benchmark"), help="output folder")
    bench.add_argument(
        "--jobs", type=int, default=None, help="worker processes (default: all cores; with --noise capped by RAM)"
    )
    return ap


# ----------------------------------------------------------------------------- ui
def _free_socket(port: int) -> socket.socket:
    """Bind 127.0.0.1 on ``port``, else the next free port up to +20, else an OS-assigned one."""
    for p in [*range(port, port + PORT_SPAN + 1), 0]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", p))
        except OSError:
            sock.close()
            continue
        return sock
    raise OSError("no free port on 127.0.0.1")  # unreachable: port 0 always binds


def _ui(args: argparse.Namespace) -> int:
    if not 2 <= args.seeds <= 20:
        print("qlassifier ui: --seeds must be between 2 and 20.", file=sys.stderr)
        return EXIT_USAGE
    import uvicorn

    from qlassifier.server.app import create_app

    sock = _free_socket(args.port)
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    results: Path = args.results.resolve()
    app = create_app(run_seeds=args.seeds, user_results=results)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on"))
    print(f"Qlassifier dashboard: {url}  (Ctrl-C stops)", flush=True)
    if results.is_dir():
        print(f"Your saved results: {results}", flush=True)
    if not args.no_browser:
        import webbrowser

        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:  # uvicorn re-raises Ctrl-C after a graceful shutdown (runs cancelled, sockets closed)
        return EXIT_INTERRUPTED
    finally:
        sock.close()
    return EXIT_OK


# ----------------------------------------------------------------------------- benchmark
class _Progress:
    """One updating progress line on stderr."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.tty = sys.stderr.isatty()
        self.last = 0.0

    def __call__(self, ev: Any) -> None:
        now = time.monotonic()
        if not self.tty and ev.done != ev.total and now - self.last < 10:
            return
        self.last = now
        eta = f"ETA {ev.eta_s:,.0f} s" if ev.eta_s is not None else "ETA —"
        seed = f"seed {ev.seed}" if ev.seed is not None else "cost"
        line = f"  {self.dataset}: {ev.done}/{ev.total} · {seed} · {ev.model} · {eta}"
        if self.tty:
            print(f"\r{line:<90}", end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.tty:
            print(file=sys.stderr, flush=True)


def _plan_eta(datasets: Sequence[str], seeds: int, noise: bool, jobs: int) -> float | None:
    """Rough wall-time estimate from the shipped results (sum of task seconds / jobs)."""
    from qlassifier.bench.result import BenchmarkResult
    from qlassifier.errors import NoResultError, UnknownDatasetError

    total = 0.0
    for ds in datasets:
        try:
            r = BenchmarkResult.cached(ds)
        except (NoResultError, UnknownDatasetError):
            return None
        per_seed = sum(m.time_s.tune + m.time_s.fit + m.time_s.predict for m in r.models)
        total += per_seed * seeds
        if noise and r.noise is not None:
            total += sum(x.seconds for x in r.noise.records)
    return total / max(jobs, 1)


def _jobs_note(requested: int | None, noise: bool, jobs: int) -> str:
    """Why the plan runs on fewer workers than cores: the noise study's RAM cap (TDD §7.11)."""
    import os

    if requested is None and noise and jobs < (os.cpu_count() or 1):
        return " (capped by RAM for the noise study; --jobs overrides)"
    return ""


def _benchmark(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from qlassifier.datasets import csv_dataset, load_dataset, looks_like_csv
    from qlassifier.errors import UnknownDatasetError

    datasets = []
    for ds in dict.fromkeys(args.datasets) or DATASET_IDS:
        if ds in DATASET_IDS:
            datasets.append(ds)
        elif looks_like_csv(ds):
            try:
                datasets.append(csv_dataset(ds, target=args.target))
            except QlassifierError as exc:
                print(f"qlassifier benchmark: {exc}", file=sys.stderr)
                return EXIT_USAGE
        else:
            print(str(UnknownDatasetError(ds)), file=sys.stderr)
            return EXIT_USAGE
    labels = {ds: load_dataset(ds).info.id for ds in datasets}  # CSV names -> file stems (folder names)
    if args.seeds < 2:
        print("qlassifier benchmark: --seeds must be at least 2.", file=sys.stderr)
        return EXIT_USAGE
    if args.jobs is not None and args.jobs < 1:
        print("qlassifier benchmark: --jobs must be at least 1.", file=sys.stderr)
        return EXIT_USAGE
    import os

    from qlassifier.bench import benchmark, default_jobs
    from qlassifier.bench.protocol import protocol_models
    from qlassifier.bench.report import bt10_from, saved_results, write_report

    load_avg = os.getloadavg()  # timing context (BT-10 is measured on an idle machine)
    jobs = args.jobs or default_jobs(args.noise)
    n_models = {ds: len(protocol_models(ds)) for ds in datasets}
    eta = _plan_eta(datasets, args.seeds, args.noise, jobs)
    plan = ", ".join(f"{labels[ds]} ({n_models[ds]} models)" for ds in datasets)
    print(
        f"Plan: {plan} × {args.seeds} seeds{' + noise study' if args.noise else ''} on {jobs} workers"
        + _jobs_note(args.jobs, args.noise, jobs)
        + (f" · ETA ≈ {eta / 60:.0f} min" if eta is not None and eta >= 90 else f" · ETA ≈ {eta:.0f} s" if eta else ""),
        flush=True,
    )
    cancel = threading.Event()

    def on_sigint(signum: int, frame: FrameType | None) -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        print("\nStopping after the running tasks (Ctrl-C again to abort)…", file=sys.stderr, flush=True)

    in_main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, on_sigint) if in_main else None
    out: Path = args.out
    results: dict[str, Any] = {}
    walls: dict[str, float] = {}
    try:
        for ds in datasets:
            label = labels[ds]
            progress = _Progress(label)
            t0 = time.perf_counter()
            try:
                result = benchmark(
                    ds, seeds=args.seeds, noise=args.noise, n_jobs=jobs, progress=progress, cancel=cancel
                )
            finally:
                progress.close()
            walls[label] = time.perf_counter() - t0
            results[label] = result
            result.save(out / label)
            if ds not in DATASET_IDS:
                print(f"Saved: {out / label} (result.json, runs.csv, plots/)", flush=True)
            print(result.summary(), flush=True)
            print(flush=True)
            if result.status == "cancelled":
                break
    except QlassifierError as exc:
        print(f"qlassifier benchmark: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if not results:
        return EXIT_INTERRUPTED if cancel.is_set() else EXIT_ERROR
    defaults = args.seeds == 10 and not args.noise and not cancel.is_set()
    command = "qlassifier " + " ".join(argv)
    bt10 = bt10_from(out, results, walls, defaults, load_avg)
    saved, saved_walls, provenance = saved_results(out, exclude=results)  # earlier runs in the folder stay in
    everything = {ds: (results | saved)[ds] for ds in DATASET_IDS if ds in results or ds in saved}
    if everything:  # the target report covers the bundled datasets; CSV results stand on their own
        report = write_report(out, everything, command, walls | saved_walls, jobs, bt10, load_avg, provenance)
        if saved:
            print(f"Report also covers earlier results in {out}: {', '.join(saved)}", flush=True)
        print(f"Report: {report}", flush=True)
    return EXIT_INTERRUPTED if cancel.is_set() else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``qlassifier`` command."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    try:
        args = ap.parse_args(args_list)
    except SystemExit as exc:  # argparse: usage error (2) or --help (0)
        return int(exc.code or 0)
    if args.version:
        print(version_line())
        return EXIT_OK
    if args.command == "ui":
        return _ui(args)
    if args.command == "benchmark":
        return _benchmark(args, args_list)
    ap.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
