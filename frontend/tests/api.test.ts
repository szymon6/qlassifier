import { describe, expect, it, vi } from "vitest";
import {
  ApiError,
  createApi,
  errorFromResponse,
  isCompatible,
  isErrorBody,
  SERVER_STOPPED,
} from "@/lib/api";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("api client", () => {
  it("calls the v1 endpoints", async () => {
    const fetchImpl = vi.fn(async (url: string) => {
      if (url.endsWith("/datasets")) return json({ datasets: [{ id: "sonar" }] });
      return json({ ok: url });
    });
    const api = createApi("http://127.0.0.1:8765", fetchImpl);
    await api.meta();
    expect(await api.datasets()).toEqual([{ id: "sonar" }]);
    await api.result("sonar");
    await api.run("r-1");
    await api.runResult("r-1");
    expect(fetchImpl.mock.calls.map((c) => c[0])).toEqual([
      "http://127.0.0.1:8765/api/v1/meta",
      "http://127.0.0.1:8765/api/v1/datasets",
      "http://127.0.0.1:8765/api/v1/results/sonar",
      "http://127.0.0.1:8765/api/v1/runs/r-1",
      "http://127.0.0.1:8765/api/v1/runs/r-1/result",
    ]);
    expect(api.eventsUrl("r 1")).toBe("http://127.0.0.1:8765/api/v1/runs/r%201/events");
  });

  it("posts JSON to start and cancel runs", async () => {
    const fetchImpl = vi.fn(async () => json({ id: "r-1" }, 202));
    const api = createApi("", fetchImpl);
    await api.startRun("sonar");
    await api.startRun("sonar", 3);
    await api.cancelRun("r-1");
    const calls = fetchImpl.mock.calls as unknown as [string, RequestInit][];
    expect(calls[0][0]).toBe("/api/v1/runs");
    expect(calls[0][1].method).toBe("POST");
    expect((calls[0][1].headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json",
    );
    expect(calls[0][1].body).toBe('{"dataset":"sonar"}');
    expect(calls[1][1].body).toBe('{"dataset":"sonar","seeds":3}');
    expect(calls[2][0]).toBe("/api/v1/runs/r-1/cancel");
  });

  it("turns error bodies into ApiError with the exact text", async () => {
    const body = {
      error: {
        code: "unknown_dataset",
        message: "No dataset 'iris'.",
        hint: "Choose one of: breast_cancer, ionosphere, sonar, heart, protein_angles.",
        status: 404,
      },
    };
    const api = createApi(
      "",
      vi.fn(async () => json(body, 404)),
    );
    const error = await api.result("iris").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).body).toEqual(body.error);
  });

  it("maps a non-JSON failure to a status error", async () => {
    const error = await errorFromResponse(new Response("boom", { status: 502 }));
    expect(error.status).toBe(502);
    expect(error.message).toBe("Request failed (502).");
  });

  it("maps a network failure to 'Server stopped.'", async () => {
    const api = createApi(
      "",
      vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))),
    );
    const error = (await api.meta().catch((e: unknown) => e)) as ApiError;
    expect(error.message).toBe(SERVER_STOPPED.message);
    expect(error.hint).toBe(SERVER_STOPPED.hint);
  });

  it("checks API and schema versions", () => {
    expect(isCompatible({ api_version: "1", schema_version: "1.3" })).toBe(true);
    expect(isCompatible({ api_version: "1", schema_version: "2.0" })).toBe(false);
    expect(isCompatible({ api_version: "2" as "1", schema_version: "1.0" })).toBe(false);
  });

  it("recognises error bodies", () => {
    expect(isErrorBody({ code: "x", message: "m", hint: "h", status: 1 })).toBe(true);
    expect(isErrorBody({ code: "x" })).toBe(false);
    expect(isErrorBody(null)).toBe(false);
  });
});
