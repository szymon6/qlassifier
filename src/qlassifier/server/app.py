"""FastAPI app of the local dashboard server (TDD §6, ADR 0007).

HTTP on 127.0.0.1 only; JSON API under ``/api/v1``; Server-Sent Events for run progress; the prebuilt static
dashboard (``src/qlassifier/_ui``) mounted at ``/`` when present (API only otherwise). Results the user saved from
code (``qlassifier benchmark my.csv`` or ``result.save(dir)``) are listed after the bundled datasets. Host allow-list
against DNS rebinding, ``application/json`` required on POST, no CORS (dev only: ``QLASSIFIER_DEV_CORS=<origin>``),
and a CSP that keeps the page offline. Every error is ``{"error": {code, message, hint, status}}`` with the
library's text.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from contextlib import asynccontextmanager
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from qlassifier.errors import (
    DATASET_IDS,
    InvalidParameterError,
    NoResultError,
    QlassifierError,
    UnknownDatasetError,
    error_body,
)
from qlassifier.server.models import (
    API_VERSION,
    DatasetCard,
    DatasetList,
    ErrorBody,
    ErrorResponse,
    Glyph,
    Meta,
    MetaVersions,
    RunRequest,
)
from qlassifier.server.runs import RunManager, Runner

__all__ = ["UI_DIR", "create_app"]

UI_DIR = Path(__file__).resolve().parent.parent / "_ui"
JSON = "application/json; charset=utf-8"
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")
CSP = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data:; font-src 'self'; "
    "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
)
PING_S = 15.0
ASGIApp = Callable[
    [MutableMapping[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]], Awaitable[None]
]


def _json(body: Any, status: int = 200) -> Response:
    text = body if isinstance(body, str | bytes) else json.dumps(body, ensure_ascii=False)
    return Response(content=text, status_code=status, media_type=JSON)


def _error(code: str, message: str, hint: str, status: int) -> Response:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, hint=hint, status=status))
    return _json(body.model_dump_json(), status)


def _host(raw: str) -> str:
    """Host without port; IPv6 literals keep their brackets."""
    raw = raw.strip().lower()
    if raw.startswith("["):
        return raw[: raw.find("]") + 1] if "]" in raw else raw
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


class _Guard:
    """Pure ASGI middleware: Host allow-list (400 JSON error) and security headers on every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if _host(headers.get("host", "")) not in ALLOWED_HOSTS:
            resp = _error(
                "bad_host", "Host header is not a loopback name.", "Open the dashboard at http://127.0.0.1.", 400
            )
            await resp(scope, receive, send)
            return

        async def send_with_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers", []))
                hdrs += [
                    (b"content-security-policy", CSP.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, send_with_headers)


@cache
def _versions() -> MetaVersions:
    return MetaVersions(
        python=platform.python_version(),
        qiskit=version("qiskit"),
        qiskit_aer=version("qiskit-aer"),
        qiskit_ibm_runtime=version("qiskit-ibm-runtime"),
        scikit_learn=version("scikit-learn"),
        numpy=version("numpy"),
    )


@cache
def _datasets_json() -> str:
    from qlassifier.bench.protocol import PUBLISHED
    from qlassifier.datasets import glyph, list_datasets, load_dataset

    cards = []
    for info in list_datasets():
        g = glyph(info.id)
        cards.append(
            DatasetCard(
                id=info.id,
                name=info.name,
                n_samples=info.n_samples,
                n_features=info.n_features,
                class_counts={str(k): v for k, v in info.class_counts.items()},
                positive_class=info.positive_class,
                licence=info.licence,
                source=info.source,
                citation=info.citation,
                recipe=info.recipe,
                grouped=info.grouped,
                geometry=load_dataset(info.id).geometry,
                glyph=Glyph(kind=g["kind"], points=g["points"]),
                published=PUBLISHED.get(info.id),
            )
        )
    return DatasetList(datasets=cards).model_dump_json()


def _user_results(directory: Path | None) -> dict[str, tuple[Path, DatasetCard]]:
    """Complete results of the user's own data saved under ``directory/<name>/result.json``, by dataset id.

    ``qlassifier benchmark my.csv`` and ``benchmark(...).save(dir)`` write them. Read on every request, so a run
    from code shows up on reload. Bundled ids, unreadable files and unfinished runs are skipped.
    """
    from qlassifier.bench.result import BenchmarkResult

    found: dict[str, tuple[Path, DatasetCard]] = {}
    if directory is None or not directory.is_dir():
        return found
    for path in sorted(directory.glob("*/result.json")):
        try:
            result = BenchmarkResult.from_json(path)
        except (OSError, ValueError):  # a broken or foreign file must not stop the dashboard
            continue
        info = result.dataset
        if info.id in DATASET_IDS or info.id in found or result.status != "complete":
            continue
        card = DatasetCard(
            id=info.id,
            name=info.name,
            n_samples=info.n_samples,
            n_features=info.n_features,
            class_counts={str(k): v for k, v in info.class_counts.items()},
            positive_class=info.positive_class,
            licence=info.licence,
            source=info.source,
            citation=info.citation,
            recipe=info.recipe,
            grouped=info.grouped,
            geometry=None,
            glyph=Glyph(kind="pca", points=[]),
            published=None,
        )
        found[info.id] = (path, card)
    return found


class _Results:
    """Shipped results (read once, served as pre-serialised bytes) and session results (TDD §6.6).

    Session runs measure no noise study; when ``GET /results/{dataset}`` serves one, the shipped study is attached
    unchanged, marked ``noise.source = "shipped"`` (with its own seeds, devices and ``created_at``), and
    ``highlights.noise_drop_max_pp`` is recomputed from it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._shipped: dict[str, bytes] = {}
        self._merged: dict[str, tuple[str, str]] = {}

    def shipped(self, dataset: str) -> bytes:
        """The shipped result file of a dataset (``NoResultError`` when missing)."""
        with self._lock:
            if dataset not in self._shipped:
                path = self.directory / f"{dataset}.json"
                if not path.is_file():
                    raise NoResultError(dataset)
                self._shipped[dataset] = path.read_bytes()
            return self._shipped[dataset]

    def session(self, dataset: str, session_json: str) -> str:
        """A completed session result with the shipped noise study attached (cached per session result)."""
        with self._lock:
            cached = self._merged.get(dataset)
        if cached is not None and cached[0] == session_json:
            return cached[1]
        merged = self._attach_noise(dataset, session_json)
        with self._lock:
            self._merged[dataset] = (session_json, merged)
        return merged

    def _attach_noise(self, dataset: str, session_json: str) -> str:
        from qlassifier.bench.noise import max_device_drop
        from qlassifier.bench.result import BenchmarkResult

        result = BenchmarkResult.model_validate_json(session_json)
        if result.noise is not None:
            return session_json
        try:
            shipped = BenchmarkResult.model_validate_json(self.shipped(dataset))
        except NoResultError:
            return session_json
        if shipped.noise is None:
            return session_json
        noise = shipped.noise.model_copy(
            update={"source": "shipped", "created_at": shipped.noise.created_at or shipped.created_at}
        )
        highlights = result.highlights.model_copy(update={"noise_drop_max_pp": max_device_drop(noise)})
        return result.model_copy(update={"noise": noise, "highlights": highlights}).model_dump_json()


def _validation_error(errors: Sequence[Any]) -> Response:
    """422 ``invalid_request`` with the first validation error as the message."""
    first = errors[0] if errors else {"loc": (), "msg": "Invalid request"}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = str(first.get("msg", "Invalid request")).rstrip(".")
    return _error(
        "invalid_request", f"{where}: {msg}." if where else f"{msg}.", "Check the request against the API schema.", 422
    )


def _require_json(request: Request) -> Response | None:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        return _error(
            "unsupported_media_type",
            "POST requests must send Content-Type: application/json.",
            "Send a JSON body with that header.",
            415,
        )
    return None


def create_app(
    *,
    run_seeds: int = 10,
    n_jobs: int | None = None,
    ui_dir: Path | None = None,
    results_dir: Path | None = None,
    runner: Runner | None = None,
    user_results: Path | None = None,
) -> FastAPI:
    """Build the dashboard app.

    Args:
        run_seeds: seeds of dashboard runs (``qlassifier ui --seeds``).
        n_jobs: worker processes of dashboard runs (None -> all cores).
        ui_dir: static dashboard folder (default: the packaged ``_ui``; served only when it holds ``index.html``).
        results_dir: shipped results folder (default: the packaged ``_results``).
        runner: benchmark callable (default ``qlassifier.benchmark``; tests pass a model subset).
        user_results: folder of results saved from code (``qlassifier ui --results``; None -> none listed).
    """
    from qlassifier.bench.result import RESULTS_DIR

    manager = RunManager(run_seeds=run_seeds, n_jobs=n_jobs, runner=runner)
    stored = _Results(RESULTS_DIR if results_dir is None else results_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(title="Qlassifier", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runs = manager

    dev_origin = os.environ.get("QLASSIFIER_DEV_CORS")
    if dev_origin:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=[dev_origin], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(_Guard)

    @app.exception_handler(QlassifierError)
    async def _typed(_: Request, exc: QlassifierError) -> Response:
        return _json(ErrorResponse(error=ErrorBody(**exc.to_dict())).model_dump_json(), exc.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> Response:
        return _validation_error(list(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 404:
            return _error("not_found", f"No resource at {request.url.path}.", "See GET /api/v1/meta.", 404)
        if exc.status_code == 405:
            return _error("method_not_allowed", f"{request.method} is not allowed here.", "See TDD §6.2.", 405)
        return _error("http_error", str(exc.detail), "See the terminal running `qlassifier ui`.", exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> Response:
        traceback.print_exception(exc, file=sys.stderr)
        return _json(ErrorResponse(error=ErrorBody(**error_body(exc))).model_dump_json(), 500)

    @app.get("/api/v1/meta")
    async def meta() -> Response:
        from qlassifier.bench.result import SCHEMA_VERSION

        body = Meta(
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            qlassifier=version("qlassifier"),
            versions=_versions(),
            default_dataset=DATASET_IDS[0],
            run_seeds=manager.run_seeds,
            active_run=manager.active(),
        )
        return _json(body.model_dump_json())

    @app.get("/api/v1/datasets")
    async def datasets() -> Response:
        bundled = await asyncio.to_thread(_datasets_json)
        user = await asyncio.to_thread(_user_results, user_results)
        if not user:
            return _json(bundled)
        cards = [*DatasetList.model_validate_json(bundled).datasets, *(card for _, card in user.values())]
        return _json(DatasetList(datasets=cards).model_dump_json())

    @app.get("/api/v1/results/{dataset}")
    async def results(dataset: str) -> Response:
        if dataset not in DATASET_IDS:
            user = await asyncio.to_thread(_user_results, user_results)
            if dataset not in user:
                raise UnknownDatasetError(dataset)
            return _json(await asyncio.to_thread(user[dataset][0].read_bytes))
        session = manager.session_result(dataset)
        if session is None:
            return _json(stored.shipped(dataset))
        return _json(await asyncio.to_thread(stored.session, dataset, session))

    @app.post("/api/v1/runs", status_code=202)
    async def start_run(request: Request) -> Response:
        if (bad := _require_json(request)) is not None:
            return bad
        try:
            payload = await request.json()
        except ValueError:
            return _error("invalid_request", "Body is not valid JSON.", 'Send e.g. {"dataset": "sonar"}.', 422)
        try:
            body = RunRequest.model_validate(payload if isinstance(payload, dict) else {})
        except ValidationError as exc:
            return _validation_error(exc.errors())
        if body.dataset not in DATASET_IDS:
            user = await asyncio.to_thread(_user_results, user_results)
            if body.dataset in user:
                raise InvalidParameterError(
                    f"'{body.dataset}' holds results you saved from code; the dashboard reruns bundled datasets.",
                    f"Rerun it with `qlassifier benchmark {user[body.dataset][1].source} --target <column>`, then "
                    "reload this page.",
                )
        run = await asyncio.to_thread(manager.start, body.dataset, body.seeds)
        return _json(run.model_dump_json(), 202)

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str) -> Response:
        return _json(manager.get(run_id).model_dump_json())

    @app.get("/api/v1/runs/{run_id}/result")
    async def run_result(run_id: str) -> Response:
        return _json(await asyncio.to_thread(manager.result_json, run_id))

    @app.post("/api/v1/runs/{run_id}/cancel")
    async def cancel_run(request: Request, run_id: str) -> Response:
        if (bad := _require_json(request)) is not None:
            return bad
        run, changed = manager.cancel(run_id)
        return _json(run.model_dump_json(), 202 if changed else 200)

    @app.get("/api/v1/runs/{run_id}/events")
    async def run_events(request: Request, run_id: str) -> StreamingResponse:
        manager.get(run_id)  # 404 before the stream starts
        try:
            last = int(request.headers.get("last-event-id", "0") or 0)
        except ValueError:
            last = 0

        async def stream() -> AsyncIterator[str]:
            nonlocal last
            last_ping = time.monotonic()
            while True:
                events, finished = manager.events_after(run_id, last)
                for eid, name, data in events:
                    yield f"id: {eid}\nevent: {name}\ndata: {data}\n\n"
                    last = eid
                if finished:
                    return
                if await request.is_disconnected():
                    return
                if time.monotonic() - last_ping >= PING_S:
                    yield ": ping\n\n"
                    last_ping = time.monotonic()
                await asyncio.to_thread(manager.wait, 0.25)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def api_not_found(request: Request, rest: str) -> Response:
        return _error("not_found", f"No resource at {request.url.path}.", "See GET /api/v1/meta.", 404)

    folder = UI_DIR if ui_dir is None else ui_dir
    if (folder / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=folder, html=True), name="ui")

    return app
