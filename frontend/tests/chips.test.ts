import { describe, expect, it } from "vitest";
import { makeResult } from "./helpers";
import { buildChips, costChip, deviceDrops, noiseChip, zzChip } from "@/lib/chips";
import type { CostStudy, NoiseStudy, PairStat } from "@/lib/types";

const T = " ";

const pair = (d: number): PairStat => ({
  mean_diff: d,
  ci_low: d - 0.04,
  ci_high: d + 0.04,
  p_value: 0.001,
  win_rate: 1,
  n: 10,
});

const noise: NoiseStudy = {
  shots: 1024,
  devices: ["fake_torino", "fake_fez", "fake_brisbane"],
  parts: [],
  records: [],
  summary: [
    ["exact", 0],
    ["shots", 0.3],
    ["fake_torino", 0.12],
    ["fake_fez", 0.23],
    ["fake_brisbane", 0.47],
  ].map(([device, drop]) => ({
    part: "A" as const,
    model: "qlassifier",
    device: device as string,
    accuracy_mean: 0.95,
    balanced_accuracy_mean: 0.94,
    drop_mean_pp: drop as number,
    drop_worst_pp: (drop as number) * 2,
    kernel_r_mean: 0.998,
    n_seeds: 5,
  })),
};

const cost: CostStudy = {
  n_train: 398,
  n_test: 171,
  devices: ["fake_torino"],
  models: [
    {
      model: "qlassifier",
      qubits: 4,
      two_qubit_gates: { fake_torino: 0 },
      depth: { fake_torino: 6 },
      circuits_fit_predict: 1707,
      tuning_circuits_on_device: 0,
    },
    {
      model: "fqk_bandwidth_linear",
      qubits: 8,
      two_qubit_gates: { fake_torino: 54 },
      depth: { fake_torino: 135 },
      circuits_fit_predict: 147061,
      tuning_circuits_on_device: 2276238,
    },
  ],
};

const result = makeResult([{ id: "qlassifier", values: [0.9] }], {
  comparisons: [
    {
      model: "zz_qsvc",
      metrics: {
        balanced_accuracy: pair(0.376),
        accuracy: pair(0.279),
        roc_auc: pair(0.3),
        f1: pair(0.5),
      },
    },
  ],
  highlights: {
    best_classical: "mlp",
    vs_standard_qml: pair(0.376),
    vs_best_classical: pair(-0.013),
    published: null,
    vs_published_pp: null,
    noise_drop_max_pp: 0.47,
    circuit_ratio_vs_fqk_pipeline: 1419.6,
  },
  noise,
  cost,
});

describe("proof chips", () => {
  it("shows Δ vs the pinned ZZ baseline for the chosen metric", () => {
    expect(zzChip(result, "balanced_accuracy")).toMatchObject({
      value: `+37.6${T}pp`,
      label: "vs standard QML",
    });
    expect(zzChip(result, "accuracy").value).toBe(`+27.9${T}pp`);
    expect(zzChip(result, "balanced_accuracy").detail).toEqual([
      { label: "95% CI", value: "[+33.6, +41.6]" },
      { label: "wins", value: "10/10" },
    ]);
  });

  it("falls back to highlights for BA and to a dash otherwise", () => {
    const noComparisons = { ...result, comparisons: [] };
    expect(zzChip(noComparisons, "balanced_accuracy").value).toBe(`+37.6${T}pp`);
    expect(zzChip(noComparisons, "f1").raw).toBeNull();
  });

  it("shows the worst device-noise loss with per-device detail", () => {
    const chip = noiseChip(result, noise);
    expect(chip.value).toBe(`0.5${T}pp`);
    expect(chip.detail.map((d) => d.label)).toEqual(["Torino", "Fez", "Brisbane"]);
    expect(deviceDrops(null)).toEqual([]);
    const noHighlight = {
      ...result,
      highlights: { ...result.highlights, noise_drop_max_pp: null },
    };
    expect(noiseChip(noHighlight, noise).raw).toBeCloseTo(0.47);
    expect(noiseChip(noHighlight, null, result).raw).toBeCloseTo(0.47);
    expect(noiseChip(noHighlight, null).raw).toBeNull();
  });

  it("shows the circuit ratio with both totals", () => {
    const chip = costChip(result);
    expect(chip.value).toBe("1,420×");
    expect(chip.detail).toEqual([
      { label: "Qlassifier", value: "1,707" },
      { label: "FQK-bw", value: "2,423,299" },
    ]);
    const live = {
      ...result,
      cost: null,
      highlights: { ...result.highlights, circuit_ratio_vs_fqk_pipeline: null },
    };
    expect(costChip(live, result).value).toBe("1,420×");
    expect(costChip(live).raw).toBeNull();
  });

  it("builds the three chips in order", () => {
    expect(buildChips(result, "balanced_accuracy", noise).map((c) => c.id)).toEqual([
      "zz",
      "noise",
      "cost",
    ]);
  });
});
