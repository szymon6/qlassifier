// Leaderboard model (TDD §8.3): rows sorted by mean of the chosen metric, Qlassifier in its
// sorted place, the classical twin right under Qlassifier, the majority class as a rule,
// rows `tied_with_best` grouped into soft bands. Statistics come from the server only.
import { groupOf, SHAPES, type Group, type Shape } from "./palette";
import type {
  BenchmarkResult,
  DatasetCard,
  MetricKey,
  ModelSummary,
  PairStat,
  Published,
} from "./types";

export interface Dot {
  seed: number;
  value: number;
}

export interface LeaderRow {
  id: string;
  name: string;
  group: Group;
  shape: Shape;
  /** null: no finished seed yet (live run). */
  mean: number | null;
  std: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  n: number;
  tied: boolean;
  dots: Dot[];
  /** Qlassifier − this model, for the chosen metric. */
  vsOurs: PairStat | null;
  /** Median seconds per seed: tune + fit + predict. */
  seconds: number | null;
}

export interface Band {
  start: number;
  end: number;
}

export interface LeaderboardModel {
  rows: LeaderRow[];
  majority: number | null;
  published: Published | null;
  best: LeaderRow | null;
  ours: LeaderRow | null;
  bands: Band[];
  domain: [number, number];
}

export const OURS_ID = "qlassifier";
export const TWIN_ID = "classical_twin";
export const TRIVIAL_ID = "trivial";

type ModelRef = Pick<ModelSummary, "id" | "name" | "family" | "role">;

function placeholder(id: string, reference: BenchmarkResult | null): ModelRef {
  const known = reference?.models.find((m) => m.id === id);
  if (known) return known;
  if (id === OURS_ID) return { id, name: "Qlassifier", family: "quantum", role: "ours" };
  return { id, name: id, family: "classical", role: "classical" };
}

/**
 * One row per model of the protocol. Models without a finished seed (live run) get an empty
 * row so the layout does not jump; their names come from `reference` (the cached result).
 */
export function buildRows(
  result: BenchmarkResult,
  metric: MetricKey,
  reference: BenchmarkResult | null = null,
): LeaderRow[] {
  const comparisons = new Map(
    result.comparisons.map((c) => [c.model, c.metrics?.[metric] ?? null]),
  );
  const dotsByModel = new Map<string, Dot[]>();
  for (const record of result.runs) {
    const value = record[metric];
    if (value == null || !Number.isFinite(value)) continue;
    const list = dotsByModel.get(record.model) ?? [];
    list.push({ seed: record.seed, value });
    dotsByModel.set(record.model, list);
  }
  const rows: LeaderRow[] = result.models.map((model) => {
    const group = groupOf(model);
    const summary = model.metrics?.[metric];
    const dots = (dotsByModel.get(model.id) ?? []).sort((a, b) => a.seed - b.seed);
    const t = model.time_s;
    return {
      id: model.id,
      name: model.name,
      group,
      shape: SHAPES[group],
      mean: summary?.mean ?? null,
      std: summary?.std ?? null,
      ciLow: summary?.ci_low ?? null,
      ciHigh: summary?.ci_high ?? null,
      n: summary?.n ?? dots.length,
      tied: Boolean(summary?.tied_with_best),
      dots,
      vsOurs: model.id === OURS_ID ? null : (comparisons.get(model.id) ?? null),
      seconds: t ? t.tune + t.fit + t.predict : null,
    };
  });
  const present = new Set(rows.map((r) => r.id));
  for (const id of result.protocol?.models ?? []) {
    if (present.has(id)) continue;
    const ref = placeholder(id, reference);
    const group = groupOf(ref);
    rows.push({
      id,
      name: ref.name,
      group,
      shape: SHAPES[group],
      mean: null,
      std: null,
      ciLow: null,
      ciHigh: null,
      n: 0,
      tied: false,
      dots: [],
      vsOurs: null,
      seconds: null,
    });
  }
  return rows;
}

/** Display order: mean desc (rows without data last, protocol order), twin under Qlassifier. */
export function orderRows(rows: LeaderRow[], protocolOrder: string[] = []): LeaderRow[] {
  const index = new Map<string, number>();
  protocolOrder.forEach((id, i) => index.set(id, i));
  rows.forEach((r, i) => {
    if (!index.has(r.id)) index.set(r.id, protocolOrder.length + i);
  });
  const visible = rows.filter((r) => r.group !== "trivial");
  const twin = visible.find((r) => r.id === TWIN_ID) ?? null;
  const rest = visible.filter((r) => r.id !== TWIN_ID);
  rest.sort((a, b) => {
    if (a.mean == null && b.mean == null) return index.get(a.id)! - index.get(b.id)!;
    if (a.mean == null) return 1;
    if (b.mean == null) return -1;
    if (b.mean !== a.mean) return b.mean - a.mean;
    return index.get(a.id)! - index.get(b.id)!;
  });
  if (!twin) return rest;
  const at = rest.findIndex((r) => r.id === OURS_ID);
  if (at === -1) return [...rest, twin];
  return [...rest.slice(0, at + 1), twin, ...rest.slice(at + 1)];
}

/** Contiguous runs of tied rows (indices into the ordered rows, inclusive). */
export function tieBands(rows: LeaderRow[]): Band[] {
  const bands: Band[] = [];
  rows.forEach((row, i) => {
    if (!row.tied || row.mean == null) return;
    const last = bands[bands.length - 1];
    if (last && last.end === i - 1) last.end = i;
    else bands.push({ start: i, end: i });
  });
  return bands;
}

/** x-domain on a 0.05 grid covering every mean, seed dot, the majority rule and ◇ published. */
export function domainOf(values: (number | null | undefined)[]): [number, number] {
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (finite.length === 0) return [0.5, 1];
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const step = 0.05;
  let low = Math.floor((lo - 0.005) / step) * step;
  let high = Math.ceil((hi + 0.005) / step) * step;
  low = Math.max(0, Math.round(low * 100) / 100);
  high = Math.min(1, Math.round(high * 100) / 100);
  if (high - low < 0.1) {
    if (high >= 1) low = Math.max(0, high - 0.1);
    else high = Math.min(1, low + 0.1);
  }
  return [low, high];
}

function domainValues(result: BenchmarkResult, metric: MetricKey): (number | null)[] {
  // CI ends are left out on purpose: with 2–3 seeds a corrected CI spans most of [0, 1] and
  // would make the axis jump during a live run. Bars are clipped to the domain instead.
  const values: (number | null)[] = [];
  for (const m of result.models) values.push(m.metrics?.[metric]?.mean ?? null);
  for (const r of result.runs) values.push(r[metric] ?? null);
  return values;
}

/**
 * @param reference the cached result of the same dataset: names for rows still empty in a live
 * run, and extra domain values so the axis holds still while seeds arrive.
 */
export function buildLeaderboard(
  result: BenchmarkResult,
  metric: MetricKey,
  card: DatasetCard | null,
  reference: BenchmarkResult | null = null,
): LeaderboardModel {
  const all = buildRows(result, metric, reference);
  const trivial = all.find((r) => r.group === "trivial") ?? null;
  const rows = orderRows(all, result.protocol?.models ?? []);
  const published =
    metric === "balanced_accuracy"
      ? (result.highlights?.published ?? card?.published ?? null)
      : null;
  const majority = trivial?.mean ?? null;
  const withData = rows.filter((r) => r.mean != null);
  const best = withData.reduce<LeaderRow | null>(
    (top, r) => (top == null || r.mean! > top.mean! ? r : top),
    null,
  );
  const values = [majority, published?.value ?? null, ...domainValues(result, metric)];
  if (reference && reference !== result) values.push(...domainValues(reference, metric));
  return {
    rows,
    majority,
    published,
    best,
    ours: rows.find((r) => r.id === OURS_ID) ?? null,
    bands: tieBands(rows),
    domain: domainOf(values),
  };
}
