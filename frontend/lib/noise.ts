// Noise view model (TDD §8.3): x = exact → shots → Torino → Fez → Brisbane,
// y = accuracy lost vs exact (pp, 0 on top), one line per (model, part).
import { DEVICE_LABELS } from "./chips";
import { groupOf, type Group } from "./palette";
import type { ModelSummary, NoiseStudy } from "./types";

export const MODES = ["exact", "shots", "fake_torino", "fake_fez", "fake_brisbane"] as const;

export interface NoisePoint {
  mode: string;
  x: number;
  drop: number;
  worst: number | null;
  kernelR: number | null;
}

export interface NoiseSeries {
  key: string;
  model: string;
  name: string;
  part: string;
  group: Group;
  /** Part B runs on a paired 60/40 subset of Breast Cancer (TDD §7.8). */
  subset: boolean;
  points: NoisePoint[];
}

export interface NoiseModel {
  series: NoiseSeries[];
  modes: { mode: string; label: string }[];
  maxDrop: number;
  minDrop: number;
}

const FALLBACK_NAMES: Record<string, string> = {
  qlassifier: "Qlassifier",
  fqk_bandwidth_linear: "FQK-bw linear",
  fqk_bandwidth_full: "FQK-bw full",
  pqk: "PQK",
  separable: "Separable",
  zz_qsvc: "ZZ QSVC",
};

export function buildNoise(noise: NoiseStudy | null, models: ModelSummary[]): NoiseModel {
  const modes = MODES.map((mode) => ({ mode, label: DEVICE_LABELS[mode] ?? mode }));
  if (!noise) return { series: [], modes, maxDrop: 0, minDrop: 0 };
  const byKey = new Map<string, NoiseSeries>();
  for (const row of noise.summary) {
    const x = (MODES as readonly string[]).indexOf(row.device);
    if (x === -1) continue;
    const part = row.part.toUpperCase();
    const key = `${row.model}:${part}`;
    let series = byKey.get(key);
    if (!series) {
      const summary = models.find((m) => m.id === row.model);
      series = {
        key,
        model: row.model,
        name: summary?.name ?? FALLBACK_NAMES[row.model] ?? row.model,
        part,
        group: summary ? groupOf(summary) : row.model === "qlassifier" ? "ours" : "quantum",
        subset: part !== "A",
        points: [],
      };
      byKey.set(key, series);
    }
    series.points.push({
      mode: row.device,
      x,
      drop: row.drop_mean_pp,
      worst: row.drop_worst_pp ?? null,
      kernelR: row.kernel_r_mean ?? null,
    });
  }
  const series = [...byKey.values()].map((s) => {
    const points = [...s.points];
    if (!points.some((p) => p.mode === "exact"))
      points.push({ mode: "exact", x: 0, drop: 0, worst: 0, kernelR: 1 });
    points.sort((a, b) => a.x - b.x);
    return { ...s, points };
  });
  // Qlassifier first (drawn last = on top is handled by the view), then Part A before B.
  series.sort((a, b) => {
    if (a.part !== b.part) return a.part < b.part ? -1 : 1;
    if (a.model === "qlassifier") return -1;
    if (b.model === "qlassifier") return 1;
    return a.name.localeCompare(b.name);
  });
  const drops = series.flatMap((s) => s.points.map((p) => p.drop));
  return {
    series,
    modes,
    maxDrop: drops.length ? Math.max(0, ...drops) : 0,
    minDrop: drops.length ? Math.min(0, ...drops) : 0,
  };
}

/** Spreads end labels so that none overlap (min gap in px), keeping their order by y. */
export function spreadLabels(ys: number[], gap: number, min: number, max: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
  const placed = order.map((o) => o.y);
  for (let k = 1; k < placed.length; k++) placed[k] = Math.max(placed[k], placed[k - 1] + gap);
  const overflow = placed.length ? placed[placed.length - 1] - max : 0;
  if (overflow > 0) {
    placed[placed.length - 1] -= overflow;
    for (let k = placed.length - 2; k >= 0; k--)
      placed[k] = Math.min(placed[k], placed[k + 1] - gap);
  }
  if (placed.length && placed[0] < min) {
    placed[0] = min;
    for (let k = 1; k < placed.length; k++) placed[k] = Math.max(placed[k], placed[k - 1] + gap);
  }
  const out = new Array<number>(ys.length);
  order.forEach((o, k) => {
    out[o.i] = placed[k];
  });
  return out;
}
