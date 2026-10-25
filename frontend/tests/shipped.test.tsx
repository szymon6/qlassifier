// Real data (TDD §6.7, §9.1): the shipped results in src/qlassifier/_results/ are the fixtures.
// They must validate against the contract and drive every view model the way the TDD describes.
import { render, screen, within } from "@testing-library/react";
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";
import { Leaderboard } from "@/components/leaderboard";
import { buildChips } from "@/lib/chips";
import { buildCost } from "@/lib/cost";
import { fmtDiffPp, fmtMetric, fmtPpPlain, fmtRatio } from "@/lib/format";
import { buildLeaderboard, OURS_ID, TWIN_ID } from "@/lib/leaderboard";
import { buildNoise } from "@/lib/noise";
import type { BenchmarkResult, MetricKey } from "@/lib/types";

const DIR = resolve(__dirname, "../../src/qlassifier/_results");
const DATASETS = ["breast_cancer", "ionosphere", "sonar", "heart", "protein_angles"];
const METRICS: MetricKey[] = ["balanced_accuracy", "accuracy", "roc_auc", "f1"];

const files = readdirSync(DIR).filter((f) => f.endsWith(".json"));
const results: [string, BenchmarkResult][] = files.map((f) => [
  f.replace(/\.json$/, ""),
  JSON.parse(readFileSync(resolve(DIR, f), "utf8")) as BenchmarkResult,
]);
const schema = JSON.parse(
  readFileSync(resolve(__dirname, "../../src/qlassifier/schemas/benchmark-result.json"), "utf8"),
);
const validate = new Ajv2020({ strict: false, allErrors: true }).compile(schema);

describe("shipped results", () => {
  it("one complete shipped result per bundled dataset", () => {
    expect(results.map(([id]) => id).sort()).toEqual([...DATASETS].sort());
    for (const [id, r] of results) {
      expect(r.dataset.id).toBe(id);
      expect(r.status).toBe("complete");
      expect(r.source).toBe("shipped");
    }
  });

  it.each(results)("%s validates against the BenchmarkResult schema", (_id, result) => {
    const ok = validate(result);
    expect(validate.errors ?? []).toEqual([]);
    expect(ok).toBe(true);
  });

  it.each(results)(
    "%s: leaderboard rows follow the means, the tie band and the twin rule",
    (id, result) => {
      for (const metric of METRICS) {
        const lb = buildLeaderboard(result, metric, null);
        const visible = result.models.filter((m) => m.role !== "trivial");
        expect(lb.rows).toHaveLength(visible.length);
        expect(lb.rows.some((r) => r.id === "trivial")).toBe(false);
        const ranked = lb.rows.filter((r) => r.id !== TWIN_ID).map((r) => r.mean!);
        expect(ranked).toEqual([...ranked].sort((a, b) => b - a));
        const tied = new Set(
          visible.filter((m) => m.metrics[metric].tied_with_best).map((m) => m.id),
        );
        const inBands = new Set(
          lb.bands.flatMap((b) => lb.rows.slice(b.start, b.end + 1).map((r) => r.id)),
        );
        expect(inBands).toEqual(tied);
        const trivial = result.models.find((m) => m.role === "trivial");
        expect(lb.majority).toBe(trivial?.metrics[metric].mean ?? null);
        expect(lb.ours?.id).toBe(OURS_ID);
        if (result.models.some((m) => m.id === TWIN_ID)) {
          const at = lb.rows.findIndex((r) => r.id === OURS_ID);
          expect(lb.rows[at + 1].id).toBe(TWIN_ID);
        }
        for (const row of lb.rows) {
          expect(row.dots).toHaveLength(row.n);
          expect(row.n).toBe(result.protocol.seeds.length);
        }
      }
      if (id === "protein_angles") expect(result.models.some((m) => m.id === TWIN_ID)).toBe(true);
    },
  );

  it.each(results)("%s: chips read the result's own numbers", (_id, result) => {
    const [zz, noise, cost] = buildChips(result, "balanced_accuracy", result.noise);
    const vsZz = result.comparisons.find((c) => c.model === "zz_qsvc")!.metrics.balanced_accuracy;
    expect(zz.value).toBe(fmtDiffPp(vsZz.mean_diff));
    expect(vsZz.mean_diff).toBeCloseTo(result.highlights.vs_standard_qml!.mean_diff, 12);
    expect(noise.value).toBe(fmtPpPlain(result.highlights.noise_drop_max_pp));
    expect(cost.value).toBe(fmtRatio(result.highlights.circuit_ratio_vs_fqk_pipeline));
  });

  it.each(results)("%s: noise and cost studies are present and drawable", (_id, result) => {
    const noise = buildNoise(result.noise, result.models);
    const ours = noise.series.find((s) => s.model === OURS_ID && !s.subset);
    expect(ours?.points.map((p) => p.mode)).toEqual([
      "exact",
      "shots",
      "fake_torino",
      "fake_fez",
      "fake_brisbane",
    ]);
    const cost = buildCost(result.cost, result.models);
    const q = cost.rows.find((r) => r.id === OURS_ID)!;
    expect(q.gates).toBe(0);
    expect(cost.rows.every((r) => r.group === "ours" || r.group === "quantum")).toBe(true);
  });

  it("Breast Cancer, BA: Qlassifier inside the top band, the pinned ZZ far outside (TDD §8.3)", () => {
    const bc = results.find(([id]) => id === "breast_cancer")![1];
    const lb = buildLeaderboard(bc, "balanced_accuracy", null);
    const ours = lb.rows.find((r) => r.id === OURS_ID)!;
    const zz = lb.rows.find((r) => r.id === "zz_qsvc")!;
    expect(ours.tied).toBe(true);
    expect(zz.tied).toBe(false);
    expect(lb.rows.at(-1)!.id).toBe("zz_qsvc");
    expect(lb.published?.value).toBe(0.968);
  });

  it("the hidden leaderboard table lists every number on screen", () => {
    const bc = results.find(([id]) => id === "breast_cancer")![1];
    const model = buildLeaderboard(bc, "balanced_accuracy", null);
    render(
      <Leaderboard
        model={model}
        metric="ba"
        dataset="breast_cancer"
        status="complete"
        focus={null}
        pinned={null}
        reduced
        onHover={() => undefined}
        onTogglePin={() => undefined}
        onClear={() => undefined}
      />,
    );
    const rows = within(screen.getByTestId("leaderboard-table")).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(model.rows.length);
    rows.forEach((tr, i) => {
      const r = model.rows[i];
      expect(tr).toHaveTextContent(r.name);
      expect(tr).toHaveTextContent(fmtMetric(r.mean));
      expect(tr).toHaveTextContent(fmtMetric(r.ciLow));
      expect(tr).toHaveTextContent(fmtMetric(r.ciHigh));
    });
  });
});
