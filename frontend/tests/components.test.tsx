import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { makeResult } from "./helpers";
import { CostChart } from "@/components/cost-chart";
import { DatasetTiles, shortName } from "@/components/dataset-tiles";
import { Leaderboard, leaderboardTable, summaryText } from "@/components/leaderboard";
import { MetricSwitch } from "@/components/metric-switch";
import { NoiseChart } from "@/components/noise-chart";
import { ProofChips } from "@/components/proof-chips";
import type { Chip } from "@/lib/chips";
import { buildLeaderboard } from "@/lib/leaderboard";
import type { DatasetCard } from "@/lib/types";
import type { MetricCode } from "@/lib/url-state";

const result = makeResult([
  { id: "trivial", values: [0.5, 0.5, 0.5] },
  { id: "mlp", name: "MLP", values: [0.96, 0.95, 0.97], tied: true, ci: [0.93, 0.99] },
  {
    id: "qlassifier",
    name: "Qlassifier",
    values: [0.95, 0.94, 0.95],
    tied: true,
    ci: [0.92, 0.97],
  },
  { id: "zz_qsvc", name: "ZZ QSVC", values: [0.57, 0.6, 0.55], ci: [0.5, 0.65] },
]);
const model = buildLeaderboard(result, "balanced_accuracy", null);

function renderBoard(props: Partial<Parameters<typeof Leaderboard>[0]> = {}) {
  const handlers = { onHover: vi.fn(), onTogglePin: vi.fn(), onClear: vi.fn() };
  render(
    <Leaderboard
      model={model}
      metric="ba"
      dataset="toy"
      status="complete"
      focus={null}
      pinned={null}
      reduced
      {...handlers}
      {...props}
    />,
  );
  return handlers;
}

describe("Leaderboard", () => {
  it("renders one option per model row (no majority row) with numbers in the name", () => {
    renderBoard();
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.getAttribute("data-model"))).toEqual([
      "mlp",
      "qlassifier",
      "zz_qsvc",
    ]);
    expect(options[1]).toHaveAttribute("data-emphasis", "ours");
    expect(options[1]).toHaveAccessibleName(
      "Qlassifier: balanced accuracy 0.947, 95% CI 0.920 to 0.970, tied with the best",
    );
    expect(screen.getByRole("listbox")).toHaveAccessibleDescription(
      /Qlassifier balanced accuracy 0\.947/,
    );
    expect(screen.getByTestId("leaderboard")).toHaveAttribute("data-ready", "true");
    expect(screen.getAllByTestId("tie-band")).toHaveLength(1);
    expect(screen.getByTestId("majority-rule")).toBeInTheDocument();
  });

  it("explains its short labels on hover: model names, the CI title, ≈ best and the published mark", () => {
    const published = {
      value: 0.968,
      metric: "balanced_accuracy" as const,
      source: "arXiv:2604.18837",
      method: "QKT",
    };
    const withPublished = buildLeaderboard(
      { ...result, highlights: { ...result.highlights, published } },
      "balanced_accuracy",
      null,
    );
    renderBoard({ model: withPublished });
    expect(screen.getByTitle(/ZZ QSVC: QSVC on the Qiskit ZZ-feature-map/)).toBeInTheDocument();
    expect(screen.getByTitle(/corrected 95% confidence interval/)).toBeInTheDocument();
    expect(screen.getByTitle(/statistically tied with the best/)).toHaveTextContent("≈ best");
    const mark = screen.getByTestId("published-mark");
    expect(mark).toHaveAttribute(
      "aria-label",
      expect.stringContaining("BA 0.968, QKT, arXiv:2604.18837"),
    );
    expect(within(mark).getByText(/best published quantum result/)).toBeInTheDocument();
    expect(
      screen.getByLabelText(/best published quantum result/, { selector: "span" }),
    ).toHaveTextContent("published");
  });

  it("clips CI ends to [0, 1] in the summary, row names and hidden table (few seeds)", () => {
    const wide = makeResult([
      { id: "trivial", values: [0.5, 0.5] },
      { id: "qlassifier", name: "Qlassifier", values: [0.83, 0.82], ci: [0.595, 1.06] },
    ]);
    const m = buildLeaderboard(wide, "balanced_accuracy", null);
    expect(summaryText(m, "ba")).toContain("95% CI 0.59 to 1.00");
    expect(leaderboardTable(m, "ba").rows[0]).toContain("1.000");
    renderBoard({ model: m });
    expect(screen.getByRole("option", { name: /Qlassifier/ })).toHaveAccessibleName(/to 1\.000/);
  });

  it("has a hidden table with every number", () => {
    renderBoard();
    const table = screen.getByTestId("leaderboard-table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(1 + model.rows.length);
    expect(within(rows[1]).getByRole("rowheader")).toHaveTextContent("MLP");
    expect(rows[1]).toHaveTextContent("0.960");
    expect(leaderboardTable(model, "ba").columns).toContain("Tied with best");
  });

  it("moves focus with arrows (roving tabindex), pins with Enter, clears with Escape", () => {
    const h = renderBoard();
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.tabIndex)).toEqual([-1, 0, -1]);
    act(() => options[1].focus());
    expect(h.onHover).toHaveBeenLastCalledWith("qlassifier");
    fireEvent.keyDown(options[1], { key: "ArrowDown" });
    expect(document.activeElement).toBe(options[2]);
    fireEvent.keyDown(options[2], { key: "ArrowDown" });
    expect(document.activeElement).toBe(options[0]);
    fireEvent.keyDown(options[0], { key: "End" });
    expect(document.activeElement).toBe(options[2]);
    fireEvent.keyDown(options[2], { key: "Home" });
    fireEvent.keyDown(options[0], { key: "ArrowUp" });
    expect(document.activeElement).toBe(options[2]);
    fireEvent.keyDown(options[2], { key: "Enter" });
    expect(h.onTogglePin).toHaveBeenLastCalledWith("zz_qsvc");
    fireEvent.keyDown(options[2], { key: " " });
    expect(h.onTogglePin).toHaveBeenCalledTimes(2);
    fireEvent.keyDown(options[2], { key: "Escape" });
    expect(h.onClear).toHaveBeenCalled();
    fireEvent.click(options[0]);
    expect(h.onTogglePin).toHaveBeenLastCalledWith("mlp");
    fireEvent.mouseEnter(options[0]);
    expect(h.onHover).toHaveBeenLastCalledWith("mlp");
  });

  it("dims other rows and shows the numbers card for the focused model", () => {
    renderBoard({ focus: "mlp", pinned: "mlp" });
    expect(screen.getByTestId("row-mlp")).toHaveAttribute("data-focus", "focus");
    expect(screen.getByTestId("row-zz_qsvc")).toHaveAttribute("data-focus", "dim");
    expect(screen.getByTestId("row-mlp")).toHaveAttribute("aria-selected", "true");
    const card = screen.getByTestId("hover-card");
    expect(card).toHaveTextContent("0.960 ± 0.010");
    expect(card).toHaveTextContent("[0.930, 0.990]");
  });

  it("summarises the relation to the best model", () => {
    expect(summaryText(model, "ba")).toMatch(/statistically tied with the best \(MLP 0\.960\)/);
    const alone = buildLeaderboard(
      makeResult([{ id: "qlassifier", values: [0.9, 0.91] }]),
      "balanced_accuracy",
      null,
    );
    expect(summaryText(alone, "ba")).toMatch(/the best mean/);
    const none = buildLeaderboard(
      makeResult([{ id: "mlp", values: [0.9] }]),
      "balanced_accuracy",
      null,
    );
    expect(summaryText(none, "acc")).toBe("Leaderboard by accuracy.");
  });
});

describe("controls", () => {
  const cards = ["breast_cancer", "sonar", "protein_angles"].map(
    (id) =>
      ({
        id,
        name:
          id === "breast_cancer"
            ? "Breast Cancer"
            : id === "sonar"
              ? "Sonar"
              : "Protein backbone angles",
        glyph: {
          kind: id === "protein_angles" ? "torus" : "pca",
          points: [
            [0.1, 0.2, 1],
            [0.5, 0.5, 0],
          ],
        },
      }) as unknown as DatasetCard,
  );

  function Tiles() {
    const [value, setValue] = useState("breast_cancer");
    return (
      <DatasetTiles
        datasets={cards}
        value={value}
        onChange={setValue}
        running={{ dataset: "sonar", progress: 0.5 }}
      />
    );
  }

  it("dataset tiles show a focus ring distinct from the selected state", () => {
    const onChange = vi.fn();
    render(
      <DatasetTiles datasets={cards} value="breast_cancer" onChange={onChange} running={null} />,
    );
    const tile = screen.getByRole("radio", { name: "Breast Cancer" });
    expect(tile.className).toMatch(/focus-visible:ring-2/);
    expect(tile.className).not.toMatch(/focus-visible:outline/);
    expect(screen.getByTestId("tile-breast_cancer").className).toMatch(
      /group-data-\[state=on\]:shadow/,
    );
  });

  it("dataset tiles are one radio group: arrows move and select, names are accessible", () => {
    render(<Tiles />);
    const group = screen.getByRole("radiogroup", { name: "Dataset" });
    const radios = within(group).getAllByRole("radio");
    expect(radios.map((r) => r.getAttribute("aria-label"))).toEqual([
      "Breast Cancer",
      "Sonar",
      "Protein backbone angles",
    ]);
    expect(radios[0]).toHaveAttribute("aria-checked", "true");
    expect(radios.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
    fireEvent.keyDown(radios[0], { key: "ArrowRight" });
    expect(radios[1]).toHaveAttribute("aria-checked", "true");
    expect(document.activeElement).toBe(radios[1]);
    fireEvent.keyDown(radios[1], { key: "ArrowLeft" });
    fireEvent.keyDown(radios[0], { key: "ArrowLeft" });
    expect(radios[2]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[2], { key: "Home" });
    expect(radios[0]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[0], { key: "End" });
    expect(radios[2]).toHaveAttribute("aria-checked", "true");
    fireEvent.click(radios[1]);
    expect(radios[1]).toHaveAttribute("aria-checked", "true");
    expect(screen.getByTestId("tile-running")).toBeInTheDocument();
    expect(shortName(cards[2])).toBe("Protein");
    expect(shortName({ id: "x", name: "New data" })).toBe("New");
  });

  it("metric switch selects with Enter / Space too", () => {
    function Switch() {
      const [m, setM] = useState<MetricCode>("ba");
      return <MetricSwitch value={m} onChange={setM} />;
    }
    render(<Switch />);
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["BA", "ACC", "AUC", "F1"]);
    fireEvent.keyDown(radios[0], { key: "ArrowDown" });
    expect(radios[1]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[3], { key: " " });
    expect(radios[3]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[2], { key: "Enter" });
    expect(radios[2]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[2], { key: "ArrowUp" });
    expect(radios[1]).toHaveAttribute("aria-checked", "true");
    fireEvent.keyDown(radios[1], { key: "x" });
    expect(radios[1]).toHaveAttribute("aria-checked", "true");
  });
});

describe("views", () => {
  it("proof chips show numbers, units and a dash when missing", () => {
    const chips: Chip[] = [
      {
        id: "zz",
        value: "+37.6 pp",
        label: "vs standard QML",
        raw: 37.6,
        detail: [{ label: "95% CI", value: "[+33.6, +41.6]" }],
      },
      { id: "noise", value: "—", label: "max loss · IBM noise", raw: null, detail: [] },
      { id: "cost", value: "1,420×", label: "fewer circuits", raw: 1420, detail: [] },
    ];
    render(<ProofChips chips={chips} />);
    expect(screen.getByTestId("chip-zz")).toHaveTextContent("+37.6pp");
    expect(screen.getByTestId("chip-zz")).toHaveTextContent("[+33.6, +41.6]");
    expect(screen.getByTestId("chip-noise")).toHaveTextContent("–");
    expect(screen.getByTestId("chip-cost")).toHaveTextContent("1,420×");
  });

  it("noise and cost charts render an image with a summary and a hidden table", () => {
    const noise = {
      shots: 1024,
      devices: ["fake_torino"],
      parts: [],
      records: [],
      summary: [
        {
          part: "A" as const,
          model: "qlassifier",
          device: "exact",
          accuracy_mean: 0.95,
          balanced_accuracy_mean: 0.94,
          drop_mean_pp: 0,
          drop_worst_pp: 0,
          kernel_r_mean: 1,
          n_seeds: 5,
        },
        {
          part: "A" as const,
          model: "qlassifier",
          device: "fake_torino",
          accuracy_mean: 0.95,
          balanced_accuracy_mean: 0.94,
          drop_mean_pp: 0.12,
          drop_worst_pp: 0.6,
          kernel_r_mean: 0.998,
          n_seeds: 5,
        },
      ],
    };
    const cost = {
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
      ],
    };
    const onHover = vi.fn();
    render(
      <>
        <NoiseChart
          noise={noise}
          models={result.models}
          focus="qlassifier"
          reduced
          onHover={onHover}
          onTogglePin={() => undefined}
        />
        <CostChart
          cost={cost}
          models={result.models}
          focus="qlassifier"
          reduced
          onHover={onHover}
          onTogglePin={() => undefined}
        />
      </>,
    );
    const images = screen.getAllByRole("img");
    expect(images[0]).toHaveAccessibleName(/Qlassifier loses at most 0\.1/);
    expect(images[1]).toHaveAccessibleName(
      /Qlassifier needs 1,707 circuits on 4 qubits with 0 two-qubit gates/,
    );
    expect(within(screen.getByTestId("noise-table")).getAllByRole("row")).toHaveLength(2);
    expect(within(screen.getByTestId("cost-table")).getAllByRole("row")).toHaveLength(2);
    expect(screen.getByTestId("cost-card")).toHaveTextContent("Torino");
  });

  it("noise and cost show a dash without a study", () => {
    render(
      <>
        <NoiseChart
          noise={null}
          models={[]}
          focus={null}
          reduced
          onHover={() => undefined}
          onTogglePin={() => undefined}
        />
        <CostChart
          cost={null}
          models={[]}
          focus={null}
          reduced
          onHover={() => undefined}
          onTogglePin={() => undefined}
        />
      </>,
    );
    const images = screen.getAllByRole("img");
    expect(images[0]).toHaveAccessibleName("Noise study: not available for this result.");
    expect(images[1]).toHaveAccessibleName("Cost study: not available for this result.");
  });
});

describe("provenance", () => {
  it("marks a cached study by a hover detail and in the summary", async () => {
    const { provenanceText, withProvenance } = await import("@/components/chart-frame");
    expect(provenanceText({ createdAt: "2026-10-24T18:02:11Z" })).toBe("cached study, 2026-10-24");
    expect(provenanceText({ createdAt: null })).toBe("cached study");
    expect(provenanceText(null)).toBe("");
    expect(withProvenance("Summary.", { createdAt: null })).toBe("Summary. Cached study.");
    expect(withProvenance("Summary.", null)).toBe("Summary.");
    const noise = {
      shots: 1024,
      devices: ["fake_torino"],
      parts: [],
      records: [],
      source: "shipped" as const,
      created_at: "2026-10-24T18:02:11Z",
      summary: [
        {
          part: "A" as const,
          model: "qlassifier",
          device: "fake_torino",
          accuracy_mean: 0.9,
          balanced_accuracy_mean: 0.9,
          drop_mean_pp: 0.2,
          drop_worst_pp: 0.6,
          kernel_r_mean: 0.99,
          n_seeds: 5,
        },
      ],
    };
    render(
      <NoiseChart
        noise={noise}
        models={result.models}
        focus={null}
        reduced
        onHover={() => undefined}
        onTogglePin={() => undefined}
        provenance={{ createdAt: noise.created_at }}
      />,
    );
    expect(screen.getByTestId("provenance")).toHaveTextContent("cached study, 2026-10-24");
    expect(screen.getByRole("img")).toHaveAccessibleName(/Cached study, 2026-10-24\.$/);
  });
});
