// Controller + screen wiring against an in-memory API (mechanism test; TDD §8.4 states).
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { toast } from "sonner";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeResult } from "./helpers";
import { Dashboard } from "@/components/dashboard";
import type { BenchmarkResult, DatasetCard, Run } from "@/lib/types";

const card = (id: string, name: string): DatasetCard => ({
  id,
  name,
  n_samples: 100,
  n_features: 8,
  class_counts: { "0": 60, "1": 40 },
  positive_class: "one",
  licence: "CC0 1.0",
  source: "test",
  citation: "test",
  recipe: "pca8",
  grouped: false,
  geometry: null,
  glyph: { kind: "pca", points: [[0.1, 0.2, 1]] },
  published: null,
});

const shipped = (status: BenchmarkResult["status"] = "complete", seeds = [0.9, 0.92, 0.91]) =>
  makeResult(
    [
      { id: "trivial", values: [0.5, 0.5, 0.5] },
      { id: "mlp", name: "MLP", values: [0.93, 0.94, 0.92], tied: true },
      { id: "qlassifier", name: "Qlassifier", values: seeds, tied: true },
      { id: "zz_qsvc", name: "ZZ QSVC", values: [0.57, 0.6, 0.55] },
    ],
    { status },
  );

const run = (patch: Partial<Run> = {}): Run => ({
  id: "r-1",
  dataset: "sonar",
  seeds: [0, 1],
  status: "running",
  done: 0,
  total: 9,
  eta_s: 9,
  started_at: "2026-10-24T00:00:00Z",
  finished_at: null,
  error: null,
  ...patch,
});

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const sse = (events: [string, unknown][]) =>
  new Response(
    events
      .map(([event, data], i) => `id: ${i + 1}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
      .join(""),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );

interface Server {
  activeRun: Run | null;
  startStatus: number;
  posts: string[];
  events: [string, unknown][];
}

function installServer(server: Server) {
  const results: Record<string, BenchmarkResult> = { breast_cancer: shipped(), sonar: shipped() };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    if (init?.method === "POST") server.posts.push(path);
    if (path === "/api/v1/meta")
      return json({
        api_version: "1",
        schema_version: "1.0",
        qlassifier: "0.1.0",
        versions: {},
        default_dataset: "breast_cancer",
        run_seeds: 2,
        active_run: server.activeRun,
      });
    if (path === "/api/v1/datasets")
      return json({ datasets: [card("breast_cancer", "Breast Cancer"), card("sonar", "Sonar")] });
    const result = path.match(/^\/api\/v1\/results\/(\w+)$/);
    if (result) {
      const r = results[result[1]];
      return r
        ? json(r)
        : json(
            {
              error: {
                code: "unknown_dataset",
                message: `No dataset '${result[1]}'.`,
                hint: "Choose one of: breast_cancer, sonar.",
                status: 404,
              },
            },
            404,
          );
    }
    if (path === "/api/v1/runs" && init?.method === "POST") {
      if (server.startStatus === 409)
        return json(
          {
            error: {
              code: "run_in_progress",
              message: "A run is already in progress (sonar, 40%).",
              hint: "Wait for it or stop it first.",
              status: 409,
            },
          },
          409,
        );
      server.activeRun = run();
      return json(server.activeRun, 202);
    }
    if (path === "/api/v1/runs/r-1/events") return sse(server.events);
    if (path === "/api/v1/runs/r-1/result") return json(shipped("complete", [0.8, 0.82, 0.81]));
    if (path === "/api/v1/runs/r-1/cancel") return json(run({ status: "cancelling" }), 202);
    return json(
      { error: { code: "not_found", message: "Not found.", hint: "", status: 404 } },
      404,
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("Dashboard", () => {
  let server: Server;

  beforeEach(() => {
    server = { activeRun: null, startStatus: 202, posts: [], events: [] };
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    // Sonner keeps toasts in a module store: clear them between tests.
    act(() => {
      toast.dismiss();
    });
  });

  it("first paint: cached Breast Cancer leaderboard, three controls, chips", async () => {
    installServer(server);
    render(<Dashboard />);
    await waitFor(() =>
      expect(screen.getByTestId("leaderboard")).toHaveAttribute("data-ready", "true"),
    );
    expect(screen.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "breast_cancer");
    expect(screen.getAllByRole("radiogroup")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: "Run benchmark" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Copy link" })).toBeNull();
    expect(screen.getByRole("button", { name: "Dark theme" })).toBeInTheDocument();
    expect(screen.getByTestId("footer")).toHaveTextContent("3 seeds");
    expect(screen.getByTestId("footer")).toHaveTextContent("v0.1.0");
  });

  it("switching dataset and metric writes the URL; Esc clears the pin", async () => {
    installServer(server);
    render(<Dashboard />);
    await waitFor(() =>
      expect(screen.getByTestId("leaderboard")).toHaveAttribute("data-ready", "true"),
    );
    fireEvent.click(screen.getByRole("radio", { name: "Sonar" }));
    await waitFor(() => expect(window.location.search).toBe("?d=sonar"));
    fireEvent.click(screen.getByRole("radio", { name: "F1" }));
    await waitFor(() => expect(window.location.search).toBe("?d=sonar&m=f1"));
    fireEvent.click(screen.getByTestId("row-mlp"));
    await waitFor(() => expect(window.location.search).toBe("?d=sonar&m=f1&h=mlp"));
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(window.location.search).toBe("?d=sonar&m=f1"));
  });

  it("an unknown dataset shows the API text and falls back to the default", async () => {
    window.history.replaceState(null, "", "/?d=iris&h=nope");
    installServer(server);
    render(<Dashboard />);
    expect(await screen.findByTestId("toast-message")).toHaveTextContent("No dataset 'iris'.");
    expect(screen.getByTestId("toast-hint")).toHaveTextContent(
      "Choose one of: breast_cancer, sonar.",
    );
    await waitFor(() =>
      expect(screen.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "breast_cancer"),
    );
    await waitFor(() => expect(window.location.search).toBe(""));
  });

  it("a stopped server shows 'Server stopped.'", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))),
    );
    render(<Dashboard />);
    expect(await screen.findByTestId("toast-message")).toHaveTextContent("Server stopped.");
    expect(screen.getByTestId("toast-hint")).toHaveTextContent("Run `qlassifier ui` again.");
  });
});
