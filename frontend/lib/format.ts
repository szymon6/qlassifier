// Number formats for the dashboard (TDD §8.5: numbers and short labels only).
import { format as d3format } from "d3-format";

const MINUS = "−";
const THIN = " ";

const fixed = (digits: number) => d3format(`.${digits}f`);
const grouped = d3format(",.0f");

function minus(text: string): string {
  return text.replace(/^-/, MINUS);
}

/** A metric value in [0, 1], e.g. 0.948. */
export function fmtMetric(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return minus(fixed(digits)(value));
}

/** Signed percentage points, e.g. +37.6 pp. `value` is already in pp. */
export function fmtPp(value: number | null | undefined, digits = 1, unit = true): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const rounded = Number(value.toFixed(digits));
  const body = fixed(digits)(Math.abs(rounded));
  const sign = rounded > 0 ? "+" : rounded < 0 ? MINUS : "";
  return `${sign}${body}${unit ? `${THIN}pp` : ""}`;
}

/**
 * Percentage points without a plus sign, e.g. 0.5 pp; a negative value keeps its minus
 * (an accuracy "loss" below zero is a gain and must read as one).
 */
export function fmtPpPlain(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const rounded = Number(value.toFixed(digits));
  return `${rounded < 0 ? MINUS : ""}${fixed(digits)(Math.abs(rounded))}${THIN}pp`;
}

/** Difference of two metric values in [0, 1], shown in pp. */
export function fmtDiffPp(diff: number | null | undefined, digits = 1): string {
  if (diff == null || !Number.isFinite(diff)) return "—";
  return fmtPp(diff * 100, digits);
}

/** A CI end clipped to the metric's range [0, 1] for display (TDD §7.7); the JSON keeps the raw value. */
export function clipUnit(value: number | null | undefined): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  return Math.min(1, Math.max(0, value));
}

/** Confidence interval of a metric, clipped to [0, 1], e.g. [0.930, 0.966]. */
export function fmtCi(
  low: number | null | undefined,
  high: number | null | undefined,
  digits = 3,
): string {
  if (low == null || high == null || !Number.isFinite(low) || !Number.isFinite(high)) return "—";
  return `[${fmtMetric(clipUnit(low), digits)}, ${fmtMetric(clipUnit(high), digits)}]`;
}

/** Confidence interval of a difference in pp, e.g. [−3.1, +0.5]. */
export function fmtCiPp(
  low: number | null | undefined,
  high: number | null | undefined,
  digits = 1,
): string {
  if (low == null || high == null || !Number.isFinite(low) || !Number.isFinite(high)) return "—";
  return `[${fmtPp(low * 100, digits, false)}, ${fmtPp(high * 100, digits, false)}]`;
}

/** Integer with thousands separators, e.g. 2,423,299. */
export function fmtInt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return minus(grouped(value));
}

/** Compact count for chart labels, e.g. 1.7k, 2.4M. */
export function fmtCompact(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs < 1000) return minus(grouped(value));
  const units: [number, string][] = [
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [size, suffix] of units) {
    if (abs >= size) {
      const scaled = value / size;
      const digits = Math.abs(scaled) >= 100 ? 0 : 1;
      return `${minus(fixed(digits)(scaled)).replace(/\.0$/, "")}${suffix}`;
    }
  }
  return minus(grouped(value));
}

/** Ratio, e.g. 1,420×. */
export function fmtRatio(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const digits = value >= 10 ? 0 : 1;
  return `${minus(d3format(`,.${digits}f`)(value))}×`;
}

/** Seconds, e.g. 0.42 s, 61 s, 4 min. */
export function fmtSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value < 0) return "—";
  if (value < 0.01) return `<0.01${THIN}s`;
  if (value < 10) return `${fixed(2)(value)}${THIN}s`;
  if (value < 120) return `${fixed(0)(value)}${THIN}s`;
  return `${fixed(0)(value / 60)}${THIN}min`;
}

/** Share in [0, 1] as a count of seeds, e.g. 9/10. */
export function fmtWins(rate: number | null | undefined, n: number | null | undefined): string {
  if (rate == null || n == null || !Number.isFinite(rate) || n <= 0) return "—";
  return `${Math.round(rate * n)}/${n}`;
}

/** Axis tick: as few decimals as the step needs. */
export function tickFormatter(step: number): (value: number) => string {
  const digits = Math.max(0, Math.min(3, Math.ceil(-Math.log10(step) - 1e-9)));
  return (value: number) => minus(fixed(digits)(value));
}
