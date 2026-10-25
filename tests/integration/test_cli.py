"""Command line (TDD §5.7, §10.2, FR-8): --version, benchmark outputs and exit codes, Ctrl-C, ui."""

from __future__ import annotations

import functools
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from qlassifier import UnknownModelError
from qlassifier.cli import _free_socket, main, version_line

FAST = ["trivial", "logistic_regression", "zz_qsvc", "qlassifier"]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == version_line()
    assert out.startswith("qlassifier 0.1.0 (python 3.12.")
    assert "qiskit 2.6.1, qiskit-aer 0.18.0, qiskit-ibm-runtime 0.51.0, scikit-learn 1.9.1, numpy 2.5.4)" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["benchmark", "iris"],
            "No dataset 'iris'. Choose one of: breast_cancer, ionosphere, sonar, heart, protein_angles.",
        ),
        (["benchmark", "heart", "--seeds", "1"], "--seeds must be at least 2."),
        (["benchmark", "heart", "--jobs", "0"], "--jobs must be at least 1."),
        (["ui", "--seeds", "1"], "--seeds must be between 2 and 20."),
    ],
)
def test_usage_errors_exit_2(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_argparse_usage_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "{ui,benchmark}" in capsys.readouterr().out
    assert main(["bogus"]) == 2
    assert main(["--help"]) == 0


def test_benchmark_writes_the_output_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import qlassifier.bench as bench

    monkeypatch.setattr(bench, "benchmark", functools.partial(bench.benchmark, models=FAST))
    out = tmp_path / "bench"
    assert main(["benchmark", "heart", "sonar", "--seeds", "2", "--out", str(out), "--jobs", "1"]) == 0
    printed = capsys.readouterr()
    assert printed.out.startswith("Plan: heart (12 models), sonar (12 models) × 2 seeds on 1 workers")
    assert "Heart Disease (Cleveland) · 4 models × 2 seeds" in printed.out
    assert f"Report: {out / 'REPORT.md'}" in printed.out
    assert "heart: 9/9" in printed.err
    for name in ("REPORT.md", "targets.json", "summary.md", "environment.json", "heart/result.json", "heart/runs.csv",
                 "heart/plots/leaderboard.png", "sonar/result.json"):  # fmt: skip
        assert (out / name).is_file(), name
    env = json.loads((out / "environment.json").read_text())
    assert env["datasets"] == ["sonar", "heart"]  # protocol order
    assert len(env["load_avg_at_start"]) == 3
    assert env["command"] == f"qlassifier benchmark heart sonar --seeds 2 --out {out} --jobs 1"
    targets = {t["id"]: t["status"] for t in json.loads((out / "targets.json").read_text())}
    assert targets["BT-10"] == "not-measured"


def test_datasets_run_one_at_a_time_share_one_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import qlassifier.bench as bench

    monkeypatch.setattr(bench, "benchmark", functools.partial(bench.benchmark, models=FAST))
    out = tmp_path / "bench"
    assert main(["benchmark", "sonar", "--seeds", "2", "--out", str(out), "--jobs", "1"]) == 0
    first = json.loads((out / "environment.json").read_text())
    assert main(["benchmark", "heart", "--seeds", "2", "--out", str(out), "--jobs", "1"]) == 0
    env = json.loads((out / "environment.json").read_text())
    assert env["datasets"] == ["sonar", "heart"]  # protocol order; sonar's wall time kept from its own run
    assert env["wall_s"]["sonar"] == first["wall_s"]["sonar"]
    assert env["command"].startswith("qlassifier benchmark heart")
    assert env["per_dataset"]["sonar"] == first["per_dataset"]["sonar"]  # wall, workers, command of its own run
    assert env["per_dataset"]["sonar"]["command"].startswith("qlassifier benchmark sonar")
    assert env["per_dataset"]["heart"]["n_jobs"] == 1
    assert "sonar" in (out / "REPORT.md").read_text()
    summary = (out / "summary.md").read_text()
    assert "## Sonar (mines vs rocks) (`sonar`)" in summary
    assert "## Heart Disease (Cleveland) (`heart`)" in summary
    assert (out / "sonar" / "result.json").is_file()


def test_run_errors_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import qlassifier.bench as bench

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise UnknownModelError("svm", ["trivial"])

    monkeypatch.setattr(bench, "benchmark", broken)
    assert main(["benchmark", "heart", "--out", str(tmp_path)]) == 1
    assert "qlassifier benchmark: No model 'svm'. Choose from: trivial." in capsys.readouterr().err


def test_ctrl_c_writes_partial_results_and_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import qlassifier.bench as bench

    real = bench.benchmark

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        progress = kwargs.pop("progress")

        def ctrl_c_after_first_task(ev: Any) -> None:
            progress(ev)
            if ev.done == 1:
                os.kill(os.getpid(), signal.SIGINT)  # the CLI's handler asks the run to stop

        return real(*args, models=FAST, progress=ctrl_c_after_first_task, **kwargs)

    monkeypatch.setattr(bench, "benchmark", interrupted)
    out = tmp_path / "partial"
    assert main(["benchmark", "heart", "sonar", "--seeds", "2", "--out", str(out), "--jobs", "1"]) == 130
    assert "Stopping after the running tasks" in capsys.readouterr().err
    result = json.loads((out / "heart" / "result.json").read_text())
    assert result["status"] == "cancelled"
    assert not (out / "sonar").exists()  # later datasets are not started
    assert (out / "REPORT.md").is_file()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # the handler is restored


def test_second_ctrl_c_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import qlassifier.bench as bench

    def twice(*args: Any, **kwargs: Any) -> Any:
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        raise AssertionError("unreachable")

    monkeypatch.setattr(bench, "benchmark", twice)
    with pytest.raises(KeyboardInterrupt):
        main(["benchmark", "heart", "--out", str(tmp_path)])


def test_ui_serves_the_dashboard_and_stops_on_ctrl_c() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = {**os.environ, "BROWSER": "true"}
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-m", "qlassifier.cli", "ui", "--port", str(port), "--seeds", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("Qlassifier dashboard: http://127.0.0.1:")
        url = line.split()[2]
        deadline = time.monotonic() + 30
        while True:
            try:
                meta = httpx.get(url + "api/v1/meta", timeout=5)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline
                time.sleep(0.1)
        assert time.monotonic() - t0 < 3.0  # TDD §10.4: first response < 3 s
        assert meta.json()["run_seeds"] == 2
        page = httpx.get(url, timeout=5)
        assert page.status_code == 200
        assert "<html" in page.text.lower()
    finally:
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=60)
    assert proc.returncode == 130  # interrupted (TDD §5.7), after a graceful shutdown
    assert "Traceback" not in err


def test_ui_in_process(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import threading

    import uvicorn

    started: dict[str, Any] = {}
    opened = threading.Event()

    def fake_run(self: uvicorn.Server, sockets: list[socket.socket]) -> None:
        started["port"] = sockets[0].getsockname()[1]
        started["app"] = self.config.app

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    monkeypatch.setattr("webbrowser.open", lambda url: opened.set())
    assert main(["ui", "--port", "0", "--seeds", "3"]) == 0
    assert opened.wait(5)
    assert f"http://127.0.0.1:{started['port']}/" in capsys.readouterr().out
    assert started["app"].state.runs.run_seeds == 3


def test_ui_lists_results_saved_from_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``qlassifier ui --results DIR`` names the folder it reads; without the folder it stays quiet."""
    import uvicorn

    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: None)
    (tmp_path / "mine").mkdir()
    assert main(["ui", "--port", "0", "--no-browser", "--results", str(tmp_path / "mine")]) == 0
    assert f"Your saved results: {tmp_path / 'mine'}" in capsys.readouterr().out
    monkeypatch.chdir(tmp_path)
    assert main(["ui", "--port", "0", "--no-browser"]) == 0  # default ./qlassifier-benchmark does not exist
    assert "Your saved results" not in capsys.readouterr().out


def test_free_socket_skips_busy_ports() -> None:
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    try:
        sock = _free_socket(port)
        assert sock.getsockname()[1] != port
        sock.close()
    finally:
        busy.close()


def test_benchmark_runs_a_csv_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """US-1 (stage 2, CLI part): ``qlassifier benchmark my.csv --target col`` saves a leaderboard folder."""
    import pandas as pd

    from qlassifier import load_dataset

    X_train, _, y_train, _ = load_dataset("heart").split(seed=0)
    csv = tmp_path / "My Data.csv"
    pd.DataFrame(X_train).assign(target=y_train).to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["benchmark", str(csv), "--target", "target", "--seeds", "2", "--out", str(out), "--jobs", "1"]) == 0
    printed = capsys.readouterr()
    assert printed.out.startswith("Plan: my_data (12 models) × 2 seeds on 1 workers")
    assert f"Saved: {out / 'my_data'}" in printed.out
    assert (out / "my_data" / "result.json").is_file()
    assert (out / "my_data" / "runs.csv").is_file()
    assert not (out / "REPORT.md").exists()  # the target report covers the bundled datasets only
    assert main(["benchmark", str(csv), "--target", "nope", "--out", str(out)]) == 2
    assert "No column 'nope'" in capsys.readouterr().err
    assert main(["benchmark", str(tmp_path / "none.csv"), "--out", str(out)]) == 2
    assert "No CSV file" in capsys.readouterr().err
