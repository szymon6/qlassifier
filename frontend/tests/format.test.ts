import { describe, expect, it } from "vitest";
import {
  clipUnit,
  fmtCi,
  fmtCiPp,
  fmtCompact,
  fmtDiffPp,
  fmtInt,
  fmtMetric,
  fmtPp,
  fmtPpPlain,
  fmtRatio,
  fmtSeconds,
  fmtWins,
  tickFormatter,
} from "@/lib/format";

const T = " ";
const M = "−";

describe("format", () => {
  it("formats metric values with 3 decimals and a dash for missing", () => {
    expect(fmtMetric(0.94812)).toBe("0.948");
    expect(fmtMetric(0.5, 2)).toBe("0.50");
    expect(fmtMetric(null)).toBe("—");
    expect(fmtMetric(Number.NaN)).toBe("—");
  });

  it("formats signed percentage points with a true minus sign", () => {
    expect(fmtPp(37.64)).toBe(`+37.6${T}pp`);
    expect(fmtPp(-1.26)).toBe(`${M}1.3${T}pp`);
    expect(fmtPp(0)).toBe(`0.0${T}pp`);
    expect(fmtPp(-0.01)).toBe(`0.0${T}pp`);
    expect(fmtPp(2, 1, false)).toBe("+2.0");
    expect(fmtPpPlain(0.47)).toBe(`0.5${T}pp`);
    expect(fmtPpPlain(-0.377)).toBe(`${M}0.4${T}pp`);
    expect(fmtPpPlain(-0.01)).toBe(`0.0${T}pp`);
    expect(fmtPpPlain(null)).toBe("—");
    expect(fmtPp(undefined)).toBe("—");
  });

  it("turns metric differences into pp", () => {
    expect(fmtDiffPp(0.3764)).toBe(`+37.6${T}pp`);
    expect(fmtDiffPp(-0.013)).toBe(`${M}1.3${T}pp`);
    expect(fmtDiffPp(null)).toBe("—");
  });

  it("formats confidence intervals", () => {
    expect(fmtCi(0.93, 0.966)).toBe("[0.930, 0.966]");
    expect(fmtCi(null, 0.9)).toBe("—");
    // few seeds: the corrected CI leaves [0, 1]; the display clips it, the JSON keeps the raw value
    expect(fmtCi(0.826, 1.043)).toBe("[0.826, 1.000]");
    expect(fmtCi(-0.2, 0.5)).toBe("[0.000, 0.500]");
    expect(clipUnit(1.043)).toBe(1);
    expect(clipUnit(null)).toBeNull();
    expect(fmtCiPp(-0.031, 0.005)).toBe(`[${M}3.1, +0.5]`);
    expect(fmtCiPp(0.1, null)).toBe("—");
  });

  it("formats counts, ratios, seconds and wins", () => {
    expect(fmtInt(2423299)).toBe("2,423,299");
    expect(fmtInt(-5)).toBe(`${M}5`);
    expect(fmtCompact(999)).toBe("999");
    expect(fmtCompact(1707)).toBe("1.7k");
    expect(fmtCompact(147061)).toBe("147k");
    expect(fmtCompact(2423299)).toBe("2.4M");
    expect(fmtCompact(2_000_000_000)).toBe("2B");
    expect(fmtRatio(1419.6)).toBe("1,420×");
    expect(fmtRatio(2.25)).toBe("2.3×");
    expect(fmtSeconds(0.004)).toBe(`<0.01${T}s`);
    expect(fmtSeconds(0.42)).toBe(`0.42${T}s`);
    expect(fmtSeconds(61.5)).toBe(`62${T}s`);
    expect(fmtSeconds(600)).toBe(`10${T}min`);
    expect(fmtSeconds(-1)).toBe("—");
    expect(fmtWins(0.9, 10)).toBe("9/10");
    expect(fmtWins(null, 10)).toBe("—");
    expect(fmtWins(0.5, 0)).toBe("—");
  });

  it("picks tick decimals from the step", () => {
    expect(tickFormatter(0.1)(0.5)).toBe("0.5");
    expect(tickFormatter(0.05)(0.95)).toBe("0.95");
    expect(tickFormatter(5)(10)).toBe("10");
    expect(tickFormatter(0.1)(-0.2)).toBe(`${M}0.2`);
  });
});
