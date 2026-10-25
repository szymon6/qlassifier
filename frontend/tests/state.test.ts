import { describe, expect, it } from "vitest";
import { makeResult } from "./helpers";
import {
  activeNoise,
  activeResult,
  datasetCard,
  focusModel,
  initialState,
  isRunActive,
  reducer,
  showsRun,
  type Action,
  type State,
} from "@/lib/state";
import type { DatasetCard, Meta, NoiseStudy, Run } from "@/lib/types";

const run = (patch: Partial<Run> = {}): Run => ({
  id: "r-1",
  dataset: "sonar",
  seeds: [0, 1],
  status: "running",
  done: 0,
  total: 23,
  eta_s: 20,
  started_at: "2026-10-24T00:00:00Z",
  finished_at: null,
  error: null,
  ...patch,
});

const noise: NoiseStudy = {
  shots: 1024,
  devices: ["fake_torino"],
  parts: [],
  records: [],
  summary: [],
};
const shipped = makeResult([{ id: "qlassifier", values: [0.9, 0.92] }], { noise });
const partial1 = makeResult([{ id: "qlassifier", values: [0.8] }], {
  status: "partial",
  source: "run",
});
const partial2 = makeResult([{ id: "qlassifier", values: [0.8, 0.85] }], {
  status: "partial",
  source: "run",
});

function apply(state: State, ...actions: Action[]): State {
  return actions.reduce(reducer, state);
}

describe("reducer", () => {
  it("starts from the URL view with defaults", () => {
    const s = initialState({ dataset: "sonar", metric: "f1" });
    expect(s.view).toEqual({ dataset: "sonar", metric: "f1", pinned: null, hovered: null });
    expect(initialState().view.dataset).toBe("breast_cancer");
  });

  it("stores meta, datasets and results; remembers the shipped noise study", () => {
    const meta = { default_dataset: "breast_cancer" } as Meta;
    const cards = [{ id: "sonar" } as DatasetCard];
    let s = apply(initialState(), { type: "meta", meta }, { type: "datasets", datasets: cards });
    s = reducer(s, { type: "result", dataset: "sonar", result: shipped });
    expect(s.meta).toBe(meta);
    expect(datasetCard(s, "sonar")).toBe(cards[0]);
    expect(datasetCard(s, "nope")).toBeNull();
    expect(s.noise.sonar).toBe(noise);
    // The same result again changes nothing (reference-equal state).
    expect(reducer(s, { type: "result", dataset: "sonar", result: shipped })).toBe(s);
    // A session result without noise keeps the remembered study.
    const s2 = reducer(s, { type: "result", dataset: "sonar", result: partial2 });
    expect(s2.noise.sonar).toBe(noise);
  });

  it("updates the view and ignores no-op patches", () => {
    const s = initialState();
    expect(reducer(s, { type: "view", patch: { dataset: "breast_cancer" } })).toBe(s);
    const s2 = reducer(s, { type: "view", patch: { metric: "auc", pinned: "mlp" } });
    expect(s2.view.metric).toBe("auc");
    expect(s2.view.pinned).toBe("mlp");
  });

  it("tracks hover and pin; hover wins the focus", () => {
    let s = reducer(initialState(), { type: "pin", model: "mlp" });
    expect(focusModel(s.view)).toBe("mlp");
    s = reducer(s, { type: "hover", model: "qlassifier" });
    expect(focusModel(s.view)).toBe("qlassifier");
    expect(reducer(s, { type: "hover", model: "qlassifier" })).toBe(s);
    expect(reducer(s, { type: "pin", model: "mlp" })).toBe(s);
    s = apply(s, { type: "hover", model: null }, { type: "pin", model: null });
    expect(focusModel(s.view)).toBeNull();
  });

  it("applies progress and partial events in order", () => {
    let s = apply(initialState({ dataset: "sonar" }), {
      type: "result",
      dataset: "sonar",
      result: shipped,
    });
    s = reducer(s, { type: "run/start", run: run() });
    expect(isRunActive(s.run)).toBe(true);
    expect(activeResult(s)).toBe(shipped);
    s = reducer(s, {
      type: "run/progress",
      runId: "r-1",
      eventId: 1,
      done: 1,
      total: 23,
      etaS: 19,
    });
    s = reducer(s, { type: "run/partial", runId: "r-1", eventId: 2, result: partial1 });
    expect(s.run?.done).toBe(1);
    expect(s.run?.lastEventId).toBe(2);
    expect(activeResult(s)).toBe(partial1);
    expect(showsRun(s)).toBe(true);
    // Session runs carry no noise: the view keeps the shipped study.
    expect(activeNoise(s)).toBe(noise);
  });

  it("is idempotent for replayed and stale events", () => {
    let s = apply(
      initialState({ dataset: "sonar" }),
      { type: "run/start", run: run() },
      { type: "run/partial", runId: "r-1", eventId: 5, result: partial2 },
    );
    const same = reducer(s, { type: "run/partial", runId: "r-1", eventId: 5, result: partial2 });
    expect(same).toBe(s);
    const stale = reducer(s, { type: "run/partial", runId: "r-1", eventId: 3, result: partial1 });
    expect(stale).toBe(s);
    const staleProgress = reducer(s, {
      type: "run/progress",
      runId: "r-1",
      eventId: 4,
      done: 1,
      total: 23,
      etaS: 1,
    });
    expect(staleProgress).toBe(s);
    const otherRun = reducer(s, {
      type: "run/partial",
      runId: "r-2",
      eventId: 9,
      result: partial1,
    });
    expect(otherRun).toBe(s);
    s = reducer(s, { type: "run/partial", runId: "r-1", eventId: 6, result: partial1 });
    expect(activeResult(s)).toBe(partial1);
  });

  it("replays a full event log to the same state as the live stream (reattach)", () => {
    const events: Action[] = [
      { type: "run/progress", runId: "r-1", eventId: 1, done: 1, total: 23, etaS: 9 },
      { type: "run/partial", runId: "r-1", eventId: 2, result: partial1 },
      { type: "run/progress", runId: "r-1", eventId: 3, done: 2, total: 23, etaS: 8 },
      { type: "run/partial", runId: "r-1", eventId: 4, result: partial2 },
    ];
    const live = apply(
      initialState({ dataset: "sonar" }),
      { type: "run/start", run: run() },
      ...events,
    );
    const replayed = apply(live, ...events);
    expect(replayed).toBe(live);
    // Reattach after reload: the same run seen again only adopts newer counters.
    expect(reducer(live, { type: "run/start", run: run({ done: 1 }) })).toBe(live);
    const ahead = reducer(live, { type: "run/start", run: run({ done: 5 }) });
    expect(ahead.run?.done).toBe(5);
    expect(ahead.run?.result).toBe(partial2);
  });

  it("handles cancel, end and the final result", () => {
    let s = apply(
      initialState({ dataset: "sonar" }),
      { type: "run/start", run: run({ done: 3 }) },
      { type: "run/status", runId: "r-1", status: "cancelling" },
    );
    expect(s.run?.status).toBe("cancelling");
    expect(reducer(s, { type: "run/status", runId: "r-1", status: "cancelling" })).toBe(s);
    s = reducer(s, { type: "run/end", runId: "r-1", eventId: 7, status: "cancelled" });
    expect(s.run?.status).toBe("cancelled");
    expect(s.run?.done).toBe(3);
    expect(isRunActive(s.run)).toBe(false);
    expect(reducer(s, { type: "run/status", runId: "r-1", status: "running" })).toBe(s);
    expect(reducer(s, { type: "run/end", runId: "r-1", eventId: 7, status: "cancelled" })).toBe(s);
    s = reducer(s, { type: "run/result", runId: "r-1", result: partial1 });
    expect(activeResult(s)).toBe(partial1);
    expect(reducer(s, { type: "run/result", runId: "r-1", result: partial1 })).toBe(s);
    expect(reducer(s, { type: "run/result", runId: "r-x", result: partial2 })).toBe(s);
  });

  it("fills the ring on complete and resets for a new run", () => {
    let s = apply(
      initialState({ dataset: "sonar" }),
      { type: "run/start", run: run({ done: 20 }) },
      { type: "run/end", runId: "r-1", eventId: 30, status: "complete" },
    );
    expect(s.run?.done).toBe(23);
    s = reducer(s, { type: "run/start", run: run({ id: "r-2", done: 0 }) });
    expect(s.run?.id).toBe("r-2");
    expect(s.run?.lastEventId).toBe(0);
    expect(s.run?.result).toBeNull();
  });

  it("shows the cached result for another dataset while a run is on", () => {
    const s = apply(
      initialState({ dataset: "heart" }),
      { type: "result", dataset: "heart", result: shipped },
      { type: "run/start", run: run() },
      { type: "run/partial", runId: "r-1", eventId: 1, result: partial1 },
    );
    expect(activeResult(s)).toBe(shipped);
    expect(showsRun(s)).toBe(false);
  });

  it("ignores unknown actions", () => {
    const s = initialState();
    expect(reducer(s, { type: "nope" } as unknown as Action)).toBe(s);
  });
});
