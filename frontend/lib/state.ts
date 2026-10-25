// Dashboard state: one reducer, pure selectors (TDD §8.8).
// Run events are idempotent: an event id at or below the last applied one is ignored, so a
// replay after `Last-Event-ID` resume or a page reload never moves the view backwards.
import type { BenchmarkResult, DatasetCard, Meta, NoiseStudy, Run, RunStatus } from "./types";
import { DEFAULT_METRIC, type MetricCode } from "./url-state";

export interface View {
  dataset: string;
  metric: MetricCode;
  pinned: string | null;
  hovered: string | null;
}

export interface RunState {
  id: string;
  dataset: string;
  status: RunStatus;
  done: number;
  total: number;
  etaS: number | null;
  lastEventId: number;
  result: BenchmarkResult | null;
}

export interface State {
  view: View;
  meta: Meta | null;
  datasets: DatasetCard[];
  shipped: Record<string, BenchmarkResult>;
  /** Last noise study seen per dataset: session runs carry none, the view keeps the shipped one. */
  noise: Record<string, NoiseStudy>;
  run: RunState | null;
}

export type Action =
  | { type: "meta"; meta: Meta }
  | { type: "datasets"; datasets: DatasetCard[] }
  | { type: "result"; dataset: string; result: BenchmarkResult }
  | { type: "view"; patch: Partial<Omit<View, "hovered">> }
  | { type: "hover"; model: string | null }
  | { type: "pin"; model: string | null }
  | { type: "run/start"; run: Run }
  | {
      type: "run/progress";
      runId: string;
      eventId: number | null;
      done: number;
      total: number;
      etaS: number | null;
    }
  | { type: "run/partial"; runId: string; eventId: number | null; result: BenchmarkResult }
  | { type: "run/status"; runId: string; status: RunStatus }
  | {
      type: "run/end";
      runId: string;
      eventId: number | null;
      status: "complete" | "cancelled" | "failed";
    }
  | { type: "run/result"; runId: string; result: BenchmarkResult };

export function initialState(view?: Partial<View>): State {
  return {
    view: {
      dataset: view?.dataset ?? "breast_cancer",
      metric: view?.metric ?? DEFAULT_METRIC,
      pinned: view?.pinned ?? null,
      hovered: null,
    },
    meta: null,
    datasets: [],
    shipped: {},
    noise: {},
    run: null,
  };
}

const ACTIVE: ReadonlySet<RunStatus> = new Set(["running", "cancelling"]);

function rememberNoise(
  state: State,
  dataset: string,
  result: BenchmarkResult,
): Record<string, NoiseStudy> {
  if (!result.noise || state.noise[dataset] === result.noise) return state.noise;
  return { ...state.noise, [dataset]: result.noise };
}

/** True when the event is new for the current run (ids only grow). */
function fresh(run: RunState | null, runId: string, eventId: number | null): run is RunState {
  if (!run || run.id !== runId) return false;
  return eventId == null || eventId > run.lastEventId;
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "meta":
      return { ...state, meta: action.meta };
    case "datasets":
      return { ...state, datasets: action.datasets };
    case "result": {
      if (state.shipped[action.dataset] === action.result) return state;
      return {
        ...state,
        shipped: { ...state.shipped, [action.dataset]: action.result },
        noise: rememberNoise(state, action.dataset, action.result),
      };
    }
    case "view": {
      const next = { ...state.view, ...action.patch };
      if (
        next.dataset === state.view.dataset &&
        next.metric === state.view.metric &&
        next.pinned === state.view.pinned
      )
        return state;
      return { ...state, view: next };
    }
    case "hover":
      if (state.view.hovered === action.model) return state;
      return { ...state, view: { ...state.view, hovered: action.model } };
    case "pin":
      if (state.view.pinned === action.model) return state;
      return { ...state, view: { ...state.view, pinned: action.model } };
    case "run/start": {
      const r = action.run;
      if (state.run && state.run.id === r.id) {
        // Same run seen again (reattach): only adopt newer counters.
        if (r.done <= state.run.done && r.status === state.run.status) return state;
        return {
          ...state,
          run: {
            ...state.run,
            status: r.status,
            done: Math.max(r.done, state.run.done),
            total: r.total,
          },
        };
      }
      return {
        ...state,
        run: {
          id: r.id,
          dataset: r.dataset,
          status: r.status,
          done: r.done,
          total: r.total,
          etaS: r.eta_s,
          lastEventId: 0,
          result: null,
        },
      };
    }
    case "run/progress": {
      if (!fresh(state.run, action.runId, action.eventId)) return state;
      const run = state.run;
      return {
        ...state,
        run: {
          ...run,
          done: action.done,
          total: action.total,
          etaS: action.etaS,
          lastEventId: action.eventId ?? run.lastEventId,
        },
      };
    }
    case "run/partial": {
      if (!fresh(state.run, action.runId, action.eventId)) return state;
      const run = state.run;
      return {
        ...state,
        run: { ...run, result: action.result, lastEventId: action.eventId ?? run.lastEventId },
      };
    }
    case "run/status": {
      const run = state.run;
      if (!run || run.id !== action.runId || run.status === action.status) return state;
      if (!ACTIVE.has(run.status)) return state;
      return { ...state, run: { ...run, status: action.status } };
    }
    case "run/end": {
      if (!fresh(state.run, action.runId, action.eventId)) return state;
      const run = state.run;
      if (!ACTIVE.has(run.status) && run.status === action.status) return state;
      return {
        ...state,
        run: {
          ...run,
          status: action.status,
          etaS: null,
          done: action.status === "complete" ? run.total : run.done,
          lastEventId: action.eventId ?? run.lastEventId,
        },
      };
    }
    case "run/result": {
      const run = state.run;
      if (!run || run.id !== action.runId || run.result === action.result) return state;
      return { ...state, run: { ...run, result: action.result } };
    }
    default:
      return state;
  }
}

// --- selectors ---------------------------------------------------------------------------

export function isRunActive(run: RunState | null): boolean {
  return run != null && ACTIVE.has(run.status);
}

/** Result on screen: the run's (same dataset, has a result) or the cached one. */
export function activeResult(state: State): BenchmarkResult | null {
  const { run, view } = state;
  if (run && run.dataset === view.dataset && run.result) return run.result;
  return state.shipped[view.dataset] ?? null;
}

/** True when the result on screen comes from this session's run. */
export function showsRun(state: State): boolean {
  const { run, view } = state;
  return run != null && run.dataset === view.dataset && run.result != null;
}

export function activeNoise(state: State): NoiseStudy | null {
  return activeResult(state)?.noise ?? state.noise[state.view.dataset] ?? null;
}

/** Model in focus: hover wins over the pin. */
export function focusModel(view: View): string | null {
  return view.hovered ?? view.pinned;
}

export function datasetCard(state: State, id: string): DatasetCard | null {
  return state.datasets.find((d) => d.id === id) ?? null;
}
