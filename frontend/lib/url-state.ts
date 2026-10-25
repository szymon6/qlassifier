// View state in the URL: ?d=<dataset>&m=<ba|acc|auc|f1>&h=<model> (TDD §8.7).
import type { MetricKey } from "./types";

export type MetricCode = "ba" | "acc" | "auc" | "f1";

export const METRICS: readonly { code: MetricCode; key: MetricKey; label: string; name: string }[] =
  [
    { code: "ba", key: "balanced_accuracy", label: "BA", name: "balanced accuracy" },
    { code: "acc", key: "accuracy", label: "ACC", name: "accuracy" },
    { code: "auc", key: "roc_auc", label: "AUC", name: "ROC-AUC" },
    { code: "f1", key: "f1", label: "F1", name: "F1" },
  ];

export const DEFAULT_METRIC: MetricCode = "ba";

export function metricKey(code: MetricCode): MetricKey {
  return METRICS.find((m) => m.code === code)?.key ?? "balanced_accuracy";
}

export function metricName(code: MetricCode): string {
  return METRICS.find((m) => m.code === code)?.name ?? "balanced accuracy";
}

export function isMetricCode(value: unknown): value is MetricCode {
  return METRICS.some((m) => m.code === value);
}

export interface UrlView {
  /** Requested dataset id; `null` = default. Validated by the API (unknown → toast). */
  dataset: string | null;
  metric: MetricCode;
  /** Pinned model id; validated against the result's models by the caller. */
  pinned: string | null;
}

const ID = /^[a-z0-9_]{1,64}$/;

/** Reads the view from a query string. Unknown `m` / malformed `h` are ignored. */
export function parseUrlView(search: string): UrlView {
  const params = new URLSearchParams(search);
  const d = params.get("d");
  const m = params.get("m");
  const h = params.get("h");
  return {
    dataset: d && d.trim() ? d.trim() : null,
    metric: isMetricCode(m) ? m : DEFAULT_METRIC,
    pinned: h && ID.test(h) ? h : null,
  };
}

/** Writes the view as a query string ("" when every value is a default). */
export function serializeUrlView(
  view: { dataset: string; metric: MetricCode; pinned: string | null },
  defaultDataset: string,
): string {
  const params = new URLSearchParams();
  if (view.dataset && view.dataset !== defaultDataset) params.set("d", view.dataset);
  if (view.metric !== DEFAULT_METRIC) params.set("m", view.metric);
  if (view.pinned) params.set("h", view.pinned);
  const text = params.toString();
  return text ? `?${text}` : "";
}

/** Replaces the current URL without a navigation (history.replaceState). */
export function writeUrlView(
  view: { dataset: string; metric: MetricCode; pinned: string | null },
  defaultDataset: string,
): void {
  if (typeof window === "undefined") return;
  const search = serializeUrlView(view, defaultDataset);
  const next = `${window.location.pathname}${search}${window.location.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) window.history.replaceState(window.history.state, "", next);
}
