// Proof chips (TDD §8.3): Δ vs the pinned ZZ baseline, worst device-noise loss, circuit ratio.
// Values come from the result (highlights, comparisons, noise summary, cost); nothing is
// re-estimated here.
import { fmtCiPp, fmtDiffPp, fmtInt, fmtPpPlain, fmtRatio, fmtWins } from "./format";
import type { BenchmarkResult, MetricKey, NoiseStudy } from "./types";

export const ZZ_ID = "zz_qsvc";
export const FQK_PIPELINE_IDS = ["fqk_bandwidth_linear", "fqk_bandwidth_full"] as const;
export const DEVICES = ["fake_torino", "fake_fez", "fake_brisbane"] as const;

export const DEVICE_LABELS: Record<string, string> = {
  exact: "exact",
  shots: "shots",
  fake_torino: "Torino",
  fake_fez: "Fez",
  fake_brisbane: "Brisbane",
};

export interface ChipDetail {
  label: string;
  value: string;
}

export interface Chip {
  id: "zz" | "noise" | "cost";
  /** Formatted value with its unit, e.g. "+37.6 pp". */
  value: string;
  label: string;
  detail: ChipDetail[];
  /** Numeric value for tests and aria. */
  raw: number | null;
}

export function zzChip(result: BenchmarkResult, metric: MetricKey): Chip {
  const comparison = result.comparisons.find((c) => c.model === ZZ_ID)?.metrics?.[metric] ?? null;
  const stat =
    comparison ??
    (metric === "balanced_accuracy" ? (result.highlights?.vs_standard_qml ?? null) : null);
  return {
    id: "zz",
    value: fmtDiffPp(stat?.mean_diff),
    label: "vs standard QML",
    raw: stat?.mean_diff != null ? stat.mean_diff * 100 : null,
    detail: [
      { label: "95% CI", value: fmtCiPp(stat?.ci_low, stat?.ci_high) },
      { label: "wins", value: fmtWins(stat?.win_rate, stat?.n) },
    ],
  };
}

/** Qlassifier's Part A drop per device (all rows), in pp. */
export function deviceDrops(noise: NoiseStudy | null): { device: string; drop: number }[] {
  if (!noise) return [];
  const rows = noise.summary.filter(
    (r) =>
      r.model === "qlassifier" &&
      r.part.toUpperCase() === "A" &&
      (DEVICES as readonly string[]).includes(r.device),
  );
  return DEVICES.flatMap((device) => {
    const row = rows.find((r) => r.device === device);
    return row ? [{ device, drop: row.drop_mean_pp }] : [];
  });
}

export function noiseChip(
  result: BenchmarkResult,
  noise: NoiseStudy | null,
  reference: BenchmarkResult | null = null,
): Chip {
  const drops = deviceDrops(noise);
  const fromHighlights =
    result.highlights?.noise_drop_max_pp ?? reference?.highlights?.noise_drop_max_pp ?? null;
  const worst = fromHighlights ?? (drops.length ? Math.max(...drops.map((d) => d.drop)) : null);
  return {
    id: "noise",
    value: fmtPpPlain(worst),
    label: "max loss · IBM noise",
    raw: worst,
    detail: drops.map((d) => ({
      label: DEVICE_LABELS[d.device] ?? d.device,
      value: fmtPpPlain(d.drop, 2),
    })),
  };
}

export function costChip(result: BenchmarkResult, reference: BenchmarkResult | null = null): Chip {
  const ratio =
    result.highlights?.circuit_ratio_vs_fqk_pipeline ??
    reference?.highlights?.circuit_ratio_vs_fqk_pipeline ??
    null;
  const models = (result.cost ?? reference?.cost)?.models ?? [];
  const ours = models.find((m) => m.model === "qlassifier");
  const fqk = FQK_PIPELINE_IDS.map((id) => models.find((m) => m.model === id)).find(Boolean);
  const detail: ChipDetail[] = [];
  if (ours)
    detail.push({
      label: "Qlassifier",
      value: fmtInt(ours.circuits_fit_predict + ours.tuning_circuits_on_device),
    });
  if (fqk)
    detail.push({
      label: "FQK-bw",
      value: fmtInt(fqk.circuits_fit_predict + fqk.tuning_circuits_on_device),
    });
  return { id: "cost", value: fmtRatio(ratio), label: "fewer circuits", raw: ratio, detail };
}

/**
 * @param reference the cached result of the same dataset: a live run has no noise study and gets
 * its cost task late, so those two chips keep the cached numbers (same protocol, same seeds).
 */
export function buildChips(
  result: BenchmarkResult,
  metric: MetricKey,
  noise: NoiseStudy | null,
  reference: BenchmarkResult | null = null,
): Chip[] {
  return [zzChip(result, metric), noiseChip(result, noise, reference), costChip(result, reference)];
}
