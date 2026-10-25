import { describe, expect, it } from "vitest";
import { makeResult } from "./helpers";
import { buildCost } from "@/lib/cost";
import { buildNoise, spreadLabels } from "@/lib/noise";
import type { CostStudy, NoiseStudy } from "@/lib/types";

const row = (part: "A" | "B", model: string, device: string, drop: number) => ({
  part,
  model,
  device,
  accuracy_mean: 0.9,
  balanced_accuracy_mean: 0.9,
  drop_mean_pp: drop,
  drop_worst_pp: drop + 1,
  kernel_r_mean: 0.99,
  n_seeds: 3,
});

const noise: NoiseStudy = {
  shots: 1024,
  devices: ["fake_torino", "fake_fez", "fake_brisbane"],
  parts: [],
  records: [],
  summary: [
    row("B", "fqk_bandwidth_linear", "fake_torino", 22.5),
    row("B", "fqk_bandwidth_linear", "exact", 0),
    row("B", "fqk_bandwidth_linear", "fake_fez", 20),
    row("A", "qlassifier", "fake_torino", 0.12),
    row("A", "qlassifier", "shots", -0.3),
    row("A", "qlassifier", "exact", 0),
    row("B", "qlassifier", "fake_fez", 0.4),
    row("A", "qlassifier", "unknown_device", 9),
  ],
};

const models = makeResult([
  { id: "qlassifier", name: "Qlassifier", values: [0.9] },
  {
    id: "fqk_bandwidth_linear",
    name: "FQK-bw linear",
    family: "quantum",
    role: "published_quantum",
    values: [0.8],
  },
]).models;

describe("noise view model", () => {
  it("builds one series per (model, part), Part A first, Qlassifier first", () => {
    const view = buildNoise(noise, models);
    expect(view.series.map((s) => s.key)).toEqual([
      "qlassifier:A",
      "qlassifier:B",
      "fqk_bandwidth_linear:B",
    ]);
    expect(view.series[0].subset).toBe(false);
    expect(view.series[2].subset).toBe(true);
    expect(view.series[2].group).toBe("quantum");
    expect(view.series[0].group).toBe("ours");
  });

  it("orders points by mode, adds exact = 0 when missing and skips unknown modes", () => {
    const view = buildNoise(noise, models);
    expect(view.series[0].points.map((p) => p.mode)).toEqual(["exact", "shots", "fake_torino"]);
    expect(view.series[1].points.map((p) => [p.mode, p.drop])).toEqual([
      ["exact", 0],
      ["fake_fez", 0.4],
    ]);
    expect(view.maxDrop).toBe(22.5);
    expect(view.minDrop).toBe(-0.3);
    expect(view.modes.map((m) => m.label)).toEqual(["exact", "shots", "Torino", "Fez", "Brisbane"]);
  });

  it("is empty without a study and names models it does not know", () => {
    expect(buildNoise(null, models).series).toEqual([]);
    const other = buildNoise({ ...noise, summary: [row("B", "pqk", "fake_fez", 3)] }, []);
    expect(other.series[0].name).toBe("PQK");
    expect(other.series[0].group).toBe("quantum");
  });

  it("spreads end labels apart within bounds, keeping their order", () => {
    expect(spreadLabels([10, 11, 12], 14, 0, 200)).toEqual([10, 24, 38]);
    expect(spreadLabels([195, 196], 14, 0, 200)).toEqual([186, 200]);
    expect(spreadLabels([-5], 14, 0, 100)).toEqual([0]);
    const ys = spreadLabels([50, 10, 12], 14, 0, 200);
    expect(ys[1]).toBeLessThan(ys[2]);
    expect(ys[2]).toBeLessThan(ys[0]);
  });
});

describe("cost view model", () => {
  const cost: CostStudy = {
    n_train: 398,
    n_test: 171,
    devices: ["fake_torino", "fake_fez"],
    models: [
      {
        model: "fqk_bandwidth_linear",
        qubits: 8,
        two_qubit_gates: { fake_torino: 54, fake_fez: 50 },
        depth: { fake_torino: 135 },
        circuits_fit_predict: 147061,
        tuning_circuits_on_device: 2276238,
      },
      {
        model: "qlassifier",
        qubits: 4,
        two_qubit_gates: { fake_torino: 0 },
        depth: { fake_torino: 6 },
        circuits_fit_predict: 1707,
        tuning_circuits_on_device: 0,
      },
    ],
  };

  it("sorts quantum models by total circuits and reads FakeTorino gates", () => {
    const view = buildCost(cost, models);
    expect(view.rows.map((r) => r.id)).toEqual(["qlassifier", "fqk_bandwidth_linear"]);
    expect(view.rows[1]).toMatchObject({
      total: 147061 + 2276238,
      tuning: 2276238,
      gates: 54,
      depth: 135,
      name: "FQK-bw linear",
    });
    expect(view.maxCircuits).toBe(147061 + 2276238);
    expect(view.minCircuits).toBe(1707);
    expect(view.maxQubits).toBe(8);
    expect(view.maxGates).toBe(54);
  });

  it("is empty without a study", () => {
    expect(buildCost(null, models).rows).toEqual([]);
  });
});

describe("noise summary", () => {
  it("says so when device noise costs nothing", async () => {
    const { noiseSummary } = await import("@/components/noise-chart");
    const gain: NoiseStudy = {
      ...noise,
      summary: [row("A", "qlassifier", "exact", 0), row("A", "qlassifier", "fake_fez", -0.38)],
    };
    expect(noiseSummary(buildNoise(gain, models))).toBe(
      "Qlassifier loses no accuracy under shot and IBM device noise (worst case −0.4 pp).",
    );
    expect(noiseSummary(buildNoise(noise, models))).toMatch(
      /^Qlassifier loses at most 0\.1 pp accuracy .*; on the 60\/40 subset FQK-bw linear loses up to 22\.5 pp\.$/,
    );
  });
});
