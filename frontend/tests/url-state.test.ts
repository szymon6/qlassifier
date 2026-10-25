import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_METRIC,
  isMetricCode,
  metricKey,
  metricName,
  parseUrlView,
  serializeUrlView,
  writeUrlView,
} from "@/lib/url-state";

describe("url state", () => {
  afterEach(() => window.history.replaceState(null, "", "/"));

  it("parses d, m and h", () => {
    expect(parseUrlView("?d=sonar&m=auc&h=mlp")).toEqual({
      dataset: "sonar",
      metric: "auc",
      pinned: "mlp",
    });
  });

  it("falls back to defaults and ignores unknown m / malformed h", () => {
    expect(parseUrlView("")).toEqual({ dataset: null, metric: DEFAULT_METRIC, pinned: null });
    expect(parseUrlView("?m=rmse&h=<script>")).toEqual({
      dataset: null,
      metric: "ba",
      pinned: null,
    });
    expect(parseUrlView("?d=%20&h=")).toEqual({ dataset: null, metric: "ba", pinned: null });
  });

  it("keeps an unknown dataset for the API to reject (toast)", () => {
    expect(parseUrlView("?d=iris").dataset).toBe("iris");
  });

  it("omits defaults when serialising", () => {
    expect(
      serializeUrlView({ dataset: "breast_cancer", metric: "ba", pinned: null }, "breast_cancer"),
    ).toBe("");
    expect(
      serializeUrlView({ dataset: "sonar", metric: "f1", pinned: "qlassifier" }, "breast_cancer"),
    ).toBe("?d=sonar&m=f1&h=qlassifier");
  });

  it("round-trips", () => {
    const view = { dataset: "heart", metric: "acc" as const, pinned: "zz_qsvc" };
    const parsed = parseUrlView(serializeUrlView(view, "breast_cancer"));
    expect(parsed).toEqual(view);
  });

  it("replaces the URL without adding history entries", () => {
    const before = window.history.length;
    writeUrlView({ dataset: "sonar", metric: "auc", pinned: null }, "breast_cancer");
    expect(window.location.search).toBe("?d=sonar&m=auc");
    writeUrlView({ dataset: "breast_cancer", metric: "ba", pinned: null }, "breast_cancer");
    expect(window.location.search).toBe("");
    expect(window.history.length).toBe(before);
  });

  it("maps metric codes", () => {
    expect(metricKey("ba")).toBe("balanced_accuracy");
    expect(metricKey("auc")).toBe("roc_auc");
    expect(metricName("acc")).toBe("accuracy");
    expect(isMetricCode("f1")).toBe(true);
    expect(isMetricCode("x")).toBe(false);
  });
});
