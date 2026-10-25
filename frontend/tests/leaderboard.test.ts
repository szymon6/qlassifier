import { describe, expect, it } from "vitest";
import { makeResult } from "./helpers";
import { buildLeaderboard, buildRows, domainOf, orderRows, tieBands } from "@/lib/leaderboard";

const base = makeResult(
  [
    { id: "trivial", values: [0.5, 0.5, 0.5] },
    { id: "mlp", name: "MLP", values: [0.96, 0.95, 0.97], tied: true },
    { id: "qlassifier", name: "Qlassifier", values: [0.95, 0.94, 0.95], tied: true },
    { id: "rbf_svm", name: "RBF-SVM", values: [0.93, 0.9, 0.91], tied: false },
    { id: "pqk", name: "PQK", values: [0.94, 0.93, 0.92], tied: true },
    { id: "zz_qsvc", name: "ZZ QSVC", values: [0.57, 0.6, 0.55] },
  ],
  {
    comparisons: [
      {
        model: "mlp",
        metrics: {
          balanced_accuracy: {
            mean_diff: -0.013,
            ci_low: -0.031,
            ci_high: 0.005,
            p_value: 0.1,
            win_rate: 0.3,
            n: 3,
          },
          accuracy: { mean_diff: 0, ci_low: 0, ci_high: 0, p_value: 1, win_rate: 0, n: 3 },
          roc_auc: { mean_diff: 0, ci_low: 0, ci_high: 0, p_value: 1, win_rate: 0, n: 3 },
          f1: { mean_diff: 0, ci_low: 0, ci_high: 0, p_value: 1, win_rate: 0, n: 3 },
        },
      },
    ],
  },
);

describe("leaderboard", () => {
  it("sorts rows by mean, keeps Qlassifier in its sorted place and drops the majority row", () => {
    const lb = buildLeaderboard(base, "balanced_accuracy", null);
    expect(lb.rows.map((r) => r.id)).toEqual(["mlp", "qlassifier", "pqk", "rbf_svm", "zz_qsvc"]);
    expect(lb.majority).toBeCloseTo(0.5);
    expect(lb.best?.id).toBe("mlp");
    expect(lb.ours?.id).toBe("qlassifier");
  });

  it("groups tied rows into contiguous bands", () => {
    const lb = buildLeaderboard(base, "balanced_accuracy", null);
    // mlp, qlassifier, pqk tied; rbf_svm not; zz not.
    expect(lb.bands).toEqual([{ start: 0, end: 2 }]);
    const rows = lb.rows.map((r, i) => ({ ...r, tied: i !== 1 }));
    expect(tieBands(rows)).toEqual([
      { start: 0, end: 0 },
      { start: 2, end: 4 },
    ]);
  });

  it("puts the classical twin right under Qlassifier, whatever its mean", () => {
    const withTwin = makeResult([
      { id: "classical_twin", name: "Classical twin", values: [0.99, 0.99] },
      { id: "mlp", values: [0.9, 0.9] },
      { id: "qlassifier", values: [0.8, 0.8] },
      { id: "rbf_svm", values: [0.7, 0.7] },
    ]);
    expect(orderRows(buildRows(withTwin, "balanced_accuracy")).map((r) => r.id)).toEqual([
      "mlp",
      "qlassifier",
      "classical_twin",
      "rbf_svm",
    ]);
  });

  it("collects per-seed dots, CI, comparisons and seconds", () => {
    const lb = buildLeaderboard(base, "balanced_accuracy", null);
    const mlp = lb.rows[0];
    expect(mlp.dots.map((d) => d.seed)).toEqual([0, 1, 2]);
    expect(mlp.ciLow).not.toBeNull();
    expect(mlp.vsOurs?.mean_diff).toBeCloseTo(-0.013);
    expect(mlp.seconds).toBeCloseTo(1.75);
    expect(lb.rows[1].vsOurs).toBeNull();
    expect(mlp.shape).toBe("square");
    expect(lb.rows[1].shape).toBe("circle");
    expect(lb.rows[2].shape).toBe("diamond");
  });

  it("shows the published mark on balanced accuracy only", () => {
    const pub = {
      value: 0.968,
      metric: "balanced_accuracy" as const,
      source: "arXiv:2604.18837",
      method: "QKT",
    };
    const card = { published: pub } as Parameters<typeof buildLeaderboard>[2];
    expect(buildLeaderboard(base, "balanced_accuracy", card).published).toEqual(pub);
    expect(buildLeaderboard(base, "accuracy", card).published).toBeNull();
  });

  it("keeps empty rows for models without a finished seed (live run)", () => {
    const partial = makeResult([{ id: "mlp", name: "MLP", values: [0.9] }], { status: "partial" });
    const protocol = { ...partial.protocol, models: ["trivial", "mlp", "qlassifier", "zz_qsvc"] };
    const lb = buildLeaderboard({ ...partial, protocol }, "balanced_accuracy", null, base);
    expect(lb.rows.map((r) => r.id)).toEqual(["mlp", "qlassifier", "zz_qsvc"]);
    expect(lb.rows[1]).toMatchObject({ name: "Qlassifier", mean: null, dots: [], n: 0 });
    expect(lb.rows[2].name).toBe("ZZ QSVC");
    expect(lb.majority).toBeNull();
    // Axis holds still: the cached result's values still count.
    expect(lb.domain[0]).toBeLessThanOrEqual(0.5);
  });

  it("names unknown placeholders by id and Qlassifier by name", () => {
    const empty = makeResult([], { status: "partial" });
    const lb = buildLeaderboard(
      { ...empty, protocol: { ...empty.protocol, models: ["qlassifier", "x_model"] } },
      "f1",
      null,
    );
    expect(lb.rows.map((r) => r.name)).toEqual(["Qlassifier", "x_model"]);
    expect(lb.best).toBeNull();
  });

  it("computes a 0.05-grid domain inside [0, 1]", () => {
    expect(domainOf([0.57, 0.97, 0.5])).toEqual([0.45, 1]);
    expect(domainOf([0.93, 0.95])).toEqual([0.9, 1]);
    expect(domainOf([0.41, 0.44])).toEqual([0.4, 0.5]);
    expect(domainOf([])).toEqual([0.5, 1]);
    expect(domainOf([null, Number.NaN])).toEqual([0.5, 1]);
    expect(domainOf([0, 0.02])).toEqual([0, 0.1]);
  });
});
