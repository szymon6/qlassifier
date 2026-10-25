"""Integration fixtures: a live loopback server, a real shipped-style result with a noise study, schema validation."""

from __future__ import annotations

import functools
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

SCHEMAS = Path(__file__).resolve().parents[2] / "src" / "qlassifier" / "schemas"
FAST_MODELS = ["trivial", "logistic_regression", "zz_qsvc", "qlassifier"]


def validate(schema: str, payload: Any) -> None:
    """Validate a JSON payload against a committed contract schema (``src/qlassifier/schemas``)."""
    import jsonschema

    jsonschema.validate(payload, json.loads((SCHEMAS / f"{schema}.json").read_text()))


class LiveServer:
    """A FastAPI app served by uvicorn on 127.0.0.1 in a background thread."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    def __enter__(self) -> LiveServer:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(30)
        self.sock.close()

    def client(self, **kw: Any) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=60, **kw)


@pytest.fixture
def live() -> Callable[[FastAPI], LiveServer]:
    """Factory: ``with live(app) as srv: srv.client().get(...)``."""
    return LiveServer


@pytest.fixture(scope="session")
def shipped_dir(tmp_path_factory: pytest.TempPathFactory, heart_result: Any) -> Path:
    """A results folder with a real Heart result (2 seeds, 5 models, noise part A + cost), marked as shipped."""
    folder = tmp_path_factory.mktemp("results")
    heart_result.as_shipped().to_json(folder / "heart.json")
    return folder


def fast_runner(**fixed: Any) -> Callable[..., Any]:
    """The real benchmark engine on a small model subset (dashboard runs in seconds)."""
    from qlassifier import benchmark

    return functools.partial(benchmark, models=FAST_MODELS, **fixed)


def read_events(client: httpx.Client, run_id: str, last_id: int | None = None) -> list[tuple[int, str, Any]]:
    """Read an SSE stream to its end: [(id, event, data)]; comments (pings) are kept as (0, ': ping', None)."""
    headers = {"Last-Event-ID": str(last_id)} if last_id is not None else {}
    events: list[tuple[int, str, Any]] = []
    with client.stream("GET", f"/api/v1/runs/{run_id}/events", headers=headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        eid, name, data = 0, "", None
        for line in resp.iter_lines():
            if line.startswith(": ping"):
                events.append((0, ": ping", None))
            elif line.startswith("id: "):
                eid = int(line[4:])
            elif line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif line == "" and name:
                events.append((eid, name, data))
                eid, name, data = 0, "", None
    return events


@pytest.fixture
def wait_for() -> Iterator[Callable[[Callable[[], bool], float], None]]:
    """Poll a condition (no fixed sleeps): raise after ``timeout`` seconds."""

    def _wait(cond: Callable[[], bool], timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > deadline:
                raise TimeoutError("condition not met")
            time.sleep(0.05)

    return _wait
