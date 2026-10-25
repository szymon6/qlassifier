"""README code runs in a fresh environment.

The ``docs`` CI job installs the package exactly as the README says (``pip install .``) into a fresh venv and runs
these tests with that venv's Python. Python blocks run as written. Shell lines run as written, with two exceptions:
``qlassifier ui`` gets ``--no-browser`` (CI has no browser), and benchmark lines run at 2 seeds (the protocol minimum)
on 2 workers into a temporary folder. The full-protocol lines (``--noise``) take hours: the CLI parser checks them
here, and their output is the saved run in ``benchmarks/``.
"""

from __future__ import annotations

import json
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from importlib.metadata import version
from pathlib import Path

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.docs

README = ROOT / "README.md"
BIN = Path(sys.executable).parent
HOURS = ("--noise",)


def blocks(text: str, lang: str) -> list[str]:
    """Fenced code blocks of one language, in order."""
    return re.findall(rf"```{lang}\n(.*?)```", text, flags=re.S)


def readme(lang: str) -> list[str]:
    return blocks(README.read_text(encoding="utf-8"), lang)


def commands() -> list[list[str]]:
    """Every shell command line of the README (comments removed)."""
    out = []
    for block in readme("bash"):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(shlex.split(line))
    return out


def test_every_command_is_tested_here() -> None:
    known = (["pip", "install"], ["qlassifier", "ui"], ["qlassifier", "benchmark"])
    assert commands()
    for cmd in commands():
        assert cmd[:2] in known, f"README command without a test: {shlex.join(cmd)}"


def test_install_is_one_command_and_installed_this_package() -> None:
    assert ["pip", "install", "."] in commands()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert version("qlassifier") == project["version"]
    assert project["requires-python"].startswith(">=3.12")


def test_quickstart_is_at_most_five_lines() -> None:
    python = readme("python")
    first = [line for line in python[0].splitlines() if line.strip()]
    assert len(first) <= 5


def test_python_blocks_run(tmp_path: Path) -> None:
    code = "\n".join(readme("python"))
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=1500, check=False
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = proc.stdout.strip().splitlines()
    accuracy, resources = lines[0].split(" ", 1)
    assert 0.9 < float(accuracy) <= 1.0
    assert resources.startswith("Resources(backend='exact', qubits=15, two_qubit_gates=0")
    head = next(i for i, line in enumerate(lines) if line.startswith("my_data.csv"))  # the own-CSV leaderboard
    assert 0.8 < float(lines[head - 1]) <= 1.0
    assert "12 models × 3 seeds" in lines[head]
    assert any(line.startswith("Qlassifier ") for line in lines[head:])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _get(url: str) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return response.status, response.read()


def test_ui_command_serves_the_dashboard() -> None:
    assert ["qlassifier", "ui"] in commands()
    port = _free_port()
    proc = subprocess.Popen(
        [str(BIN / "qlassifier"), "ui", "--no-browser", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                status, body = _get(f"http://127.0.0.1:{port}/api/v1/meta")
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"qlassifier ui did not answer: {proc.stdout.read() if proc.stdout else ''}")
                time.sleep(0.5)
        assert status == 200
        assert json.loads(body)["api_version"] == "1"
        status, page = _get(f"http://127.0.0.1:{port}/")
        assert status == 200
        assert b"<html" in page.lower()
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)


def test_benchmark_commands(tmp_path: Path) -> None:
    from qlassifier.cli import _parser

    lines = [cmd for cmd in commands() if cmd[:2] == ["qlassifier", "benchmark"]]
    assert lines
    ran = 0
    for cmd in lines:
        args = _parser().parse_args(cmd[1:])
        assert args.command == "benchmark"
        if any(flag in cmd for flag in HOURS):
            continue
        out = tmp_path / "-".join(cmd[2:] or ["all"])
        proc = subprocess.run(
            [str(BIN / "qlassifier"), *cmd[1:], "--seeds", "2", "--jobs", "2", "--out", str(out)],
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        assert (out / "REPORT.md").is_file()
        for dataset in args.datasets:
            assert (out / dataset / "result.json").is_file()
        ran += 1
    assert ran >= 1


def test_relative_links_resolve() -> None:
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    for target in targets:
        if "://" in target:
            continue
        path = target.split("#", 1)[0]
        assert (ROOT / path).exists(), f"README links to a missing file: {target}"
