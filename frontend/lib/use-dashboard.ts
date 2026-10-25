// Dashboard controller: load order, URL sync, prefetch, run lifecycle (TDD §8.4, §8.8).
"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { showError } from "@/components/toaster";
import { ApiError, VERSION_MISMATCH, createApi, isCompatible, type ApiClient } from "./api";
import { subscribe, type Subscription } from "./events";
import { activeResult, initialState, reducer, type State } from "./state";
import type { ApiErrorBody, BenchmarkResult, Run } from "./types";
import { parseUrlView, writeUrlView, type MetricCode } from "./url-state";

export interface DashboardController {
  state: State;
  ready: boolean;
  settled: boolean;
  announcement: string;
  setDataset(id: string): void;
  setMetric(metric: MetricCode): void;
  hover(model: string | null): void;
  togglePin(model: string): void;
  clearFocus(): void;
}

function asErrorBody(error: unknown): ApiErrorBody {
  if (error instanceof ApiError) return error.body;
  return {
    code: "internal",
    message: error instanceof Error ? error.message : "Unexpected error.",
    hint: "See the terminal running `qlassifier ui`.",
    status: 0,
  };
}

function parseData<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

const whenIdle = (fn: () => void): void => {
  const w = window as Window & {
    requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number;
  };
  if (w.requestIdleCallback) w.requestIdleCallback(fn, { timeout: 2000 });
  else window.setTimeout(fn, 200);
};

export function useDashboard(apiOverride?: ApiClient): DashboardController {
  const api = useMemo(() => apiOverride ?? createApi(), [apiOverride]);
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState());
  const [ready, setReady] = useState(false);
  // First load finished, with or without data (stops the loading skeleton).
  const [settled, setSettled] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);
  const subscription = useRef<Subscription | null>(null);
  const loading = useRef(new Set<string>());
  const quarter = useRef(0);

  const announce = useCallback((text: string) => setAnnouncement(text), []);

  const loadResult = useCallback(
    async (dataset: string): Promise<BenchmarkResult> => {
      const result = await api.result(dataset);
      dispatch({ type: "result", dataset, result });
      return result;
    },
    [api],
  );

  // --- run lifecycle ---------------------------------------------------------------------
  const finish = useCallback(
    async (runId: string, dataset: string, status: "complete" | "cancelled") => {
      try {
        const result = await api.runResult(runId);
        dispatch({ type: "run/result", runId, result });
      } catch (error) {
        showError(asErrorBody(error));
      }
      announce(status === "complete" ? `Run complete: ${dataset}.` : `Run stopped: ${dataset}.`);
    },
    [api, announce],
  );

  const attach = useCallback(
    (run: Run) => {
      dispatch({ type: "run/start", run });
      subscription.current?.close();
      quarter.current = run.total > 0 ? Math.floor((4 * run.done) / run.total) : 0;
      if (run.status !== "running" && run.status !== "cancelling") return;
      subscription.current = subscribe(api.eventsUrl(run.id), {
        onMessage(message, eventId) {
          const runId = run.id;
          switch (message.event) {
            case "progress": {
              const data = parseData<{ done: number; total: number; eta_s: number | null }>(
                message.data,
              );
              if (!data) return;
              dispatch({
                type: "run/progress",
                runId,
                eventId,
                done: data.done,
                total: data.total,
                etaS: data.eta_s,
              });
              const q = data.total > 0 ? Math.floor((4 * data.done) / data.total) : 0;
              if (q > quarter.current && q < 4) announce(`${q * 25}%`);
              quarter.current = Math.max(quarter.current, q);
              return;
            }
            case "partial": {
              const data = parseData<{ result: BenchmarkResult }>(message.data);
              if (data?.result)
                dispatch({ type: "run/partial", runId, eventId, result: data.result });
              return;
            }
            case "complete":
            case "cancelled": {
              dispatch({ type: "run/end", runId, eventId, status: message.event });
              void finish(runId, run.dataset, message.event);
              return;
            }
            case "failed": {
              dispatch({ type: "run/end", runId, eventId, status: "failed" });
              const data = parseData<{ error: ApiErrorBody }>(message.data);
              if (data?.error) showError(data.error);
              announce(`Run failed: ${run.dataset}.`);
              return;
            }
            default:
              return;
          }
        },
        onError(error) {
          showError(error.body);
        },
        onLost(error) {
          showError(error.body);
        },
      });
    },
    [api, announce, finish],
  );

  // --- first load: meta → datasets + result → paint → prefetch -------------------------------
  useEffect(() => {
    let cancelled = false;
    const url = parseUrlView(window.location.search);
    dispatch({ type: "view", patch: { metric: url.metric, pinned: url.pinned } });
    (async () => {
      let meta;
      try {
        meta = await api.meta();
      } catch (error) {
        showError(asErrorBody(error));
        if (!cancelled) setSettled(true);
        return;
      }
      if (cancelled) return;
      dispatch({ type: "meta", meta });
      if (!isCompatible(meta)) showError(VERSION_MISMATCH);
      const requested = url.dataset ?? meta.default_dataset;
      dispatch({ type: "view", patch: { dataset: requested } });
      const datasets = api.datasets().then(
        (list) => !cancelled && dispatch({ type: "datasets", datasets: list }),
        (error: unknown) => showError(asErrorBody(error)),
      );
      try {
        await loadResult(requested);
      } catch (error) {
        showError(asErrorBody(error));
        // Unknown `d` in the URL: toast + the default dataset (TDD §8.7).
        const unknown = error instanceof ApiError && error.code === "unknown_dataset";
        if (unknown && requested !== meta.default_dataset) {
          dispatch({ type: "view", patch: { dataset: meta.default_dataset } });
          try {
            await loadResult(meta.default_dataset);
          } catch (inner) {
            showError(asErrorBody(inner));
          }
        }
      }
      await datasets;
      if (cancelled) return;
      setReady(true);
      setSettled(true);
      if (
        meta.active_run &&
        (meta.active_run.status === "running" || meta.active_run.status === "cancelling")
      ) {
        attach(meta.active_run);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api, attach, loadResult]);

  useEffect(() => () => subscription.current?.close(), []);

  // Prefetch the other datasets when idle (instant switching).
  useEffect(() => {
    if (!ready) return;
    whenIdle(() => {
      const { datasets, shipped } = stateRef.current;
      for (const card of datasets) {
        if (shipped[card.id] || loading.current.has(card.id)) continue;
        loading.current.add(card.id);
        void loadResult(card.id)
          .catch(() => undefined)
          .finally(() => loading.current.delete(card.id));
      }
    });
  }, [ready, loadResult]);

  // Load on switch when the prefetch has not landed yet.
  const dataset = state.view.dataset;
  useEffect(() => {
    if (!ready || stateRef.current.shipped[dataset] || loading.current.has(dataset)) return;
    loading.current.add(dataset);
    loadResult(dataset)
      .catch((error: unknown) => showError(asErrorBody(error)))
      .finally(() => loading.current.delete(dataset));
  }, [ready, dataset, loadResult]);

  // URL state (replaceState; defaults omitted).
  const { metric, pinned } = state.view;
  const defaultDataset = state.meta?.default_dataset ?? null;
  useEffect(() => {
    if (!ready || !defaultDataset) return;
    writeUrlView({ dataset, metric, pinned }, defaultDataset);
  }, [ready, dataset, metric, pinned, defaultDataset]);

  // Drop a pin that names no model of the result on screen (unknown `h` is ignored).
  const result = activeResult(state);
  useEffect(() => {
    if (!result || !pinned) return;
    if (!result.models.some((m) => m.id === pinned)) dispatch({ type: "pin", model: null });
  }, [result, pinned]);

  return {
    state,
    ready,
    settled,
    announcement,
    setDataset: useCallback((id: string) => dispatch({ type: "view", patch: { dataset: id } }), []),
    setMetric: useCallback((m: MetricCode) => dispatch({ type: "view", patch: { metric: m } }), []),
    hover: useCallback((model: string | null) => dispatch({ type: "hover", model }), []),
    togglePin: useCallback(
      (model: string) =>
        dispatch({ type: "pin", model: stateRef.current.view.pinned === model ? null : model }),
      [],
    ),
    clearFocus: useCallback(() => {
      dispatch({ type: "pin", model: null });
      dispatch({ type: "hover", model: null });
    }, []),
  };
}
