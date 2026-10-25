"""Local server (TDD §6, ADR 0007): every endpoint and status, SSE, replay, errors, static dashboard, shipped noise."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qlassifier import UnknownModelError
from qlassifier.server.app import CSP, create_app
from tests.integration.conftest import LiveServer, fast_runner, read_events, validate

JSON_POST = {"content-type": "application/json"}


def test_meta(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    with live(create_app(run_seeds=3, results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/meta")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/json; charset=utf-8"
        assert r.headers["content-security-policy"] == CSP
        body = r.json()
        validate("meta", body)
        assert body["api_version"] == "1"
        assert body["schema_version"] == "1.0"
        assert body["default_dataset"] == "breast_cancer"
        assert body["run_seeds"] == 3
        assert body["active_run"] is None
        assert set(body["versions"]) == {
            "python",
            "qiskit",
            "qiskit_aer",
            "qiskit_ibm_runtime",
            "scikit_learn",
            "numpy",
        }


def test_datasets(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        body = c.get("/api/v1/datasets").json()
        validate("dataset-list", body)
        cards = {d["id"]: d for d in body["datasets"]}
        assert list(cards) == ["breast_cancer", "ionosphere", "sonar", "heart", "protein_angles"]
        assert cards["breast_cancer"]["published"] == {
            "value": 0.968, "metric": "balanced_accuracy", "source": "arXiv:2604.18837", "method": "QKT"
        }  # fmt: skip
        assert cards["protein_angles"]["published"] is None
        assert cards["protein_angles"]["glyph"]["kind"] == "torus"
        assert cards["protein_angles"]["geometry"]["phi_0"] == 360.0
        assert cards["heart"]["glyph"]["kind"] == "pca"
        assert cards["heart"]["class_counts"] == {"0": 160, "1": 137}
        assert len(cards["sonar"]["glyph"]["points"]) == 120


def test_results_shipped_unknown_and_missing(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/results/heart")
        assert r.status_code == 200
        assert r.content == (shipped_dir / "heart.json").read_bytes()  # pre-serialised bytes, served unchanged
        t0 = time.perf_counter()
        c.get("/api/v1/results/heart")
        assert time.perf_counter() - t0 < 0.2  # TDD §10.4: < 50 ms target, < 200 ms asserted
        body = r.json()
        validate("benchmark-result", body)
        assert body["source"] == "shipped"
        assert body["noise"]["source"] == "shipped"
        r = c.get("/api/v1/results/iris")
        assert r.status_code == 404
        validate("error-response", r.json())
        assert r.json()["error"] == {
            "code": "unknown_dataset",
            "message": "No dataset 'iris'.",
            "hint": "Choose one of: breast_cancer, ionosphere, sonar, heart, protein_angles.",
            "status": 404,
        }
        r = c.get("/api/v1/results/sonar")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "no_result"


def _user_folder(root: Path, heart_result: Any) -> Path:
    """A results folder as ``qlassifier benchmark my_data.csv`` leaves it, plus files the dashboard must skip."""
    from dataclasses import replace

    mine = replace(heart_result.dataset, id="my_data", name="my_data.csv", source="/data/my_data.csv",
                   licence="user data", citation="user data (not redistributed)")  # fmt: skip
    heart_result.model_copy(update={"dataset": mine}).save(root / "my_data")
    skipped = {
        "heart": heart_result,  # a bundled id
        "half": heart_result.model_copy(update={"dataset": replace(mine, id="half_done"), "status": "partial"}),
    }
    for name, result in skipped.items():
        (root / name).mkdir()
        result.to_json(root / name / "result.json")
    (root / "broken").mkdir()
    (root / "broken" / "result.json").write_text("{not json", encoding="utf-8")
    return root


def test_results_saved_from_code(
    live: Callable[..., LiveServer], shipped_dir: Path, heart_result: Any, tmp_path: Path
) -> None:
    """``qlassifier ui --results``: results of the user's own data, saved from code, show next to the bundled ones."""
    folder = _user_folder(tmp_path / "qlassifier-benchmark", heart_result)
    with live(create_app(results_dir=shipped_dir, user_results=folder)) as srv, srv.client() as c:
        body = c.get("/api/v1/datasets").json()
        validate("dataset-list", body)
        cards = {d["id"]: d for d in body["datasets"]}
        assert list(cards) == ["breast_cancer", "ionosphere", "sonar", "heart", "protein_angles", "my_data"]
        mine = cards["my_data"]
        assert (mine["name"], mine["licence"], mine["published"]) == ("my_data.csv", "user data", None)
        assert mine["class_counts"] == {"0": 160, "1": 137}
        assert mine["glyph"] == {"kind": "pca", "points": []}
        r = c.get("/api/v1/results/my_data")
        assert r.status_code == 200
        assert r.content == (folder / "my_data" / "result.json").read_bytes()
        validate("benchmark-result", r.json())
        assert c.get("/api/v1/results/half_done").json()["error"]["code"] == "unknown_dataset"
        assert c.get("/api/v1/results/heart").json()["source"] == "shipped"  # a bundled id is never replaced
        r = c.post("/api/v1/runs", json={"dataset": "my_data"})
        assert r.status_code == 422
        error = r.json()["error"]
        assert error["code"] == "invalid_parameter"
        assert "qlassifier benchmark /data/my_data.csv --target <column>" in error["hint"]
        assert c.get("/api/v1/meta").json()["active_run"] is None
    with live(create_app(results_dir=shipped_dir, user_results=tmp_path / "missing")) as srv, srv.client() as c:
        ids = [d["id"] for d in c.get("/api/v1/datasets").json()["datasets"]]
        assert ids == ["breast_cancer", "ionosphere", "sonar", "heart", "protein_angles"]


def test_run_lifecycle_events_replay_and_session_result(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    app = create_app(run_seeds=2, n_jobs=1, results_dir=shipped_dir, runner=fast_runner())
    with live(app) as srv, srv.client() as c:
        r = c.post("/api/v1/runs", json={"dataset": "heart"})
        assert r.status_code == 202
        run = r.json()
        validate("run", run)
        assert run["status"] == "running"
        assert run["seeds"] == [0, 1]
        assert run["total"] == 2 * 12 + 1  # seeds x protocol models + cost; progress events carry the run's own total

        busy = c.post("/api/v1/runs", json={"dataset": "sonar"})
        assert busy.status_code == 409
        validate("error-response", busy.json())
        assert busy.json()["error"]["code"] == "run_in_progress"
        assert busy.json()["error"]["message"].startswith("A run is already in progress (heart, ")

        events = read_events(c, run["id"])
        names = [e[1] for e in events if e[0]]
        assert names[-1] == "complete"
        assert names.count("complete") == 1
        assert "partial" in names
        ids = [e[0] for e in events if e[0]]
        assert ids == list(range(1, len(ids) + 1))
        progress = [e[2] for e in events if e[1] == "progress"]
        assert [p["done"] for p in progress] == list(range(1, progress[-1]["total"] + 1))
        for p in progress:
            validate("progress-data", p)
        cost = [p for p in progress if p["model"] == "cost"]
        assert len(cost) == 1
        assert cost[0]["seed"] is None
        partials = [e[2] for e in events if e[1] == "partial"]
        for p in partials:
            validate("partial-data", p)
        seeds_done = [len({x["seed"] for x in p["result"]["runs"]}) for p in partials]
        assert seeds_done == sorted(seeds_done)
        assert partials[-1]["result"]["status"] == "partial"
        complete = events[-1][2]
        validate("terminal-data", complete)
        assert complete == {"run_id": run["id"], "result_url": f"/api/v1/runs/{run['id']}/result"}

        replay = read_events(c, run["id"], last_id=ids[-3])
        assert [e[0] for e in replay] == ids[-2:]
        assert replay[-1][1] == "complete"

        state = c.get(f"/api/v1/runs/{run['id']}").json()
        assert state["status"] == "complete"
        assert state["done"] == state["total"]
        assert state["finished_at"] is not None
        result = c.get(complete["result_url"]).json()
        validate("benchmark-result", result)
        assert result["status"] == "complete"
        assert result["noise"] is None  # session runs measure no noise
        assert {m["id"] for m in result["models"]} == {"trivial", "logistic_regression", "zz_qsvc", "qlassifier"}

        session = c.get("/api/v1/results/heart").json()  # the session result, with the shipped noise attached
        validate("benchmark-result", session)
        assert session["source"] == "run"
        assert session["created_at"] == result["created_at"]
        shipped = (shipped_dir / "heart.json").read_text()
        from qlassifier.bench.result import BenchmarkResult

        ref = BenchmarkResult.from_json(shipped)
        assert session["noise"]["source"] == "shipped"
        assert session["noise"]["created_at"] == ref.noise.created_at  # type: ignore[union-attr]
        assert session["noise"]["records"] == [r.model_dump() for r in ref.noise.records]  # type: ignore[union-attr]
        assert session["highlights"]["noise_drop_max_pp"] == ref.highlights.noise_drop_max_pp
        assert session["runs"] == result["runs"]
        assert c.get("/api/v1/results/heart").json() == session  # cached merge

        again = c.post(f"/api/v1/runs/{run['id']}/cancel", headers=JSON_POST)
        assert again.status_code == 200  # finished runs are unchanged
        assert again.json()["status"] == "complete"
        assert c.get("/api/v1/meta").json()["active_run"] is None


def test_cancel_mid_run_keeps_the_partial_result(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    from qlassifier import benchmark

    first_task = threading.Event()

    def runner(dataset: str, **kw: Any) -> Any:
        progress, cancel = kw.pop("progress"), kw["cancel"]

        def hold(ev: Any) -> None:  # after the first task, wait until the client asked to stop
            progress(ev)
            first_task.set()
            cancel.wait(30)

        return benchmark(dataset, models=["trivial", "logistic_regression", "qlassifier"], progress=hold, **kw)

    with live(create_app(run_seeds=3, n_jobs=1, results_dir=shipped_dir, runner=runner)) as srv, srv.client() as c:
        run = c.post("/api/v1/runs", json={"dataset": "heart", "seeds": 3}).json()
        assert first_task.wait(30)
        active = c.get("/api/v1/meta").json()["active_run"]
        assert active["id"] == run["id"]
        r = c.post(f"/api/v1/runs/{run['id']}/cancel", headers=JSON_POST)
        assert r.status_code == 202
        assert r.json()["status"] == "cancelling"
        events = read_events(c, run["id"])
        assert events[-1][1] == "cancelled"
        validate("terminal-data", events[-1][2])
        state = c.get(f"/api/v1/runs/{run['id']}").json()
        assert state["status"] == "cancelled"
        assert 0 < state["done"] < state["total"]
        result = c.get(f"/api/v1/runs/{run['id']}/result").json()
        assert result["status"] == "cancelled"
        assert len(result["runs"]) + (result["cost"] is not None) == state["done"]  # finished tasks are kept
        assert c.get("/api/v1/results/heart").json()["source"] == "shipped"  # cancelled runs never replace it


def test_failed_run_reports_the_typed_error(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    def broken(dataset: str, **kw: Any) -> Any:
        raise UnknownModelError("svm", ["trivial", "qlassifier"])

    with live(create_app(n_jobs=1, results_dir=shipped_dir, runner=broken)) as srv, srv.client() as c:
        run = c.post("/api/v1/runs", json={"dataset": "heart", "seeds": 2}).json()
        events = read_events(c, run["id"])
        assert [e[1] for e in events] == ["failed"]
        failed = events[0][2]
        validate("failed-data", failed)
        assert failed["error"] == {
            "code": "unknown_model", "message": "No model 'svm'.", "hint": "Choose from: trivial, qlassifier.",
            "status": 422,
        }  # fmt: skip
        state = c.get(f"/api/v1/runs/{run['id']}").json()
        assert state["status"] == "failed"
        assert state["error"] == failed["error"]
        partial = c.get(f"/api/v1/runs/{run['id']}/result").json()  # nothing finished: an empty partial result
        validate("benchmark-result", partial)
        assert partial["runs"] == []
        assert partial["models"] == []


def test_request_errors(live: Callable[..., LiveServer], shipped_dir: Path, tmp_path: Path) -> None:
    app = create_app(results_dir=shipped_dir, ui_dir=tmp_path, runner=fast_runner())  # empty ui_dir: API only
    with live(app) as srv, srv.client() as c:
        r = c.post("/api/v1/runs", content="dataset=heart")
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "unsupported_media_type"
        r = c.post("/api/v1/runs", content="{not json", headers=JSON_POST)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"
        for body, message in [
            ({"dataset": "heart", "seeds": 50}, "seeds: Input should be less than or equal to 20."),
            ({"dataset": "heart", "seeds": 1}, "seeds: Input should be greater than or equal to 2."),
            ({"seeds": 3}, "dataset: Field required."),
            ({"dataset": "heart", "noise": True}, "noise: Extra inputs are not permitted."),
            ([1, 2], "dataset: Field required."),
        ]:
            r = c.post("/api/v1/runs", json=body)
            assert r.status_code == 422
            validate("error-response", r.json())
            assert r.json()["error"]["message"] == message
        r = c.post("/api/v1/runs", json={"dataset": "iris"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_dataset"
        for path in ("/api/v1/runs/r-000000", "/api/v1/runs/r-000000/result", "/api/v1/runs/r-000000/events"):
            r = c.get(path)
            assert r.status_code == 404
            assert r.json()["error"] == {
                "code": "run_not_found", "message": "No run 'r-000000' in this session.", "hint": "Start a new run.",
                "status": 404,
            }  # fmt: skip
        assert c.post("/api/v1/runs/r-000000/cancel", headers=JSON_POST).status_code == 404
        assert c.post("/api/v1/runs/r-000000/cancel").status_code == 415
        r = c.get("/api/v1/nothing")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"
        assert c.delete("/api/v1/meta").status_code == 404
        assert c.get("/").status_code == 404  # no dashboard build: API only


def test_host_allow_list(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/meta", headers={"host": "evil.example"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "bad_host"
        for host in ("localhost", f"localhost:{srv.port}", "[::1]", f"[::1]:{srv.port}", "127.0.0.1"):
            assert c.get("/api/v1/meta", headers={"host": host}).status_code == 200


def test_static_dashboard_is_served(live: Callable[..., LiveServer], shipped_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>qlassifier</title>")
    (tmp_path / "_next").mkdir()
    (tmp_path / "_next" / "app.js").write_text("console.log(1)")
    with live(create_app(ui_dir=tmp_path, results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "qlassifier" in r.text
        assert r.headers["content-security-policy"] == CSP
        assert c.get("/_next/app.js").status_code == 200
        assert c.get("/api/v1/meta").status_code == 200
        r = c.get("/api/v1/unknown")
        assert r.headers["content-type"].startswith("application/json")
        assert c.get("/missing.png").status_code == 404


def test_unexpected_errors_become_internal(
    live: Callable[..., LiveServer], shipped_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qlassifier.server.app as app_mod

    def boom() -> str:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(app_mod, "_datasets_json", boom)
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/datasets")
        assert r.status_code == 500
        assert r.json()["error"] == {
            "code": "internal", "message": "Unexpected error (RuntimeError).",
            "hint": "See the terminal running `qlassifier ui`.", "status": 500,
        }  # fmt: skip


def test_dev_cors_origin(live: Callable[..., LiveServer], shipped_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLASSIFIER_DEV_CORS", "http://localhost:3000")
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/meta", headers={"origin": "http://localhost:3000"})
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    monkeypatch.delenv("QLASSIFIER_DEV_CORS")
    with live(create_app(results_dir=shipped_dir)) as srv, srv.client() as c:
        r = c.get("/api/v1/meta", headers={"origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in r.headers


def test_stream_pings_while_waiting(
    live: Callable[..., LiveServer], shipped_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qlassifier.server.app as app_mod
    from qlassifier import benchmark

    monkeypatch.setattr(app_mod, "PING_S", 0.2)
    gate = threading.Event()

    def slow(dataset: str, **kw: Any) -> Any:
        gate.wait(1.0)  # the run starts late: the open stream sends keep-alive comments meanwhile
        return benchmark(dataset, models=["trivial", "logistic_regression"], **kw)

    with live(create_app(n_jobs=1, results_dir=shipped_dir, runner=slow)) as srv, srv.client() as c:
        run = c.post("/api/v1/runs", json={"dataset": "heart", "seeds": 2}).json()
        events = read_events(c, run["id"])
        assert (0, ": ping", None) in events
        assert events[-1][1] == "complete"


def test_shutdown_cancels_the_active_run(live: Callable[..., LiveServer], shipped_dir: Path) -> None:
    from qlassifier import benchmark

    started = threading.Event()
    seen: dict[str, Any] = {}

    def runner(dataset: str, **kw: Any) -> Any:
        progress, cancel = kw.pop("progress"), kw["cancel"]

        def hold(ev: Any) -> None:
            progress(ev)
            started.set()
            cancel.wait(30)

        result = benchmark(dataset, models=["trivial", "logistic_regression"], progress=hold, **kw)
        seen["status"] = result.status
        return result

    app = create_app(n_jobs=1, results_dir=shipped_dir, runner=runner)
    with live(app) as srv, srv.client() as c:
        c.post("/api/v1/runs", json={"dataset": "heart", "seeds": 3})
        assert started.wait(30)
    assert seen["status"] == "cancelled"  # server stop (Ctrl-C) cancels the run and waits for it
