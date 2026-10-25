import { describe, expect, it } from "vitest";
import {
  contrast,
  groupOf,
  PALETTE,
  paletteCss,
  parseColor,
  SHAPES,
  type Theme,
} from "@/lib/palette";

const THEMES: Theme[] = ["light", "dark"];

describe("palette (WCAG AA, both themes)", () => {
  it.each(THEMES)("text tokens reach 4.5:1 on card and page (%s)", (theme) => {
    const t = PALETTE[theme];
    for (const fg of [t.foreground, t.muted, t.faint, t.oursText]) {
      expect(contrast(fg, t.card)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(fg, t.background)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it.each(THEMES)("marks reach 3:1 on the card, also on a band (%s)", (theme) => {
    const t = PALETTE[theme];
    for (const mark of [t.ours, t.quantum, t.classical, t.twin, t.trivial]) {
      expect(contrast(mark, t.card)).toBeGreaterThanOrEqual(3);
    }
  });

  it.each(THEMES)("text on the soft Qlassifier band stays AA (%s)", (theme) => {
    const t = PALETTE[theme];
    const r = parseColor(t.oursSoft);
    const card = parseColor(t.card);
    const a = r[3];
    const mixed = `rgb(${r[0] * a + card[0] * (1 - a)}, ${r[1] * a + card[1] * (1 - a)}, ${r[2] * a + card[2] * (1 - a)})`;
    expect(contrast(t.oursText, mixed)).toBeGreaterThanOrEqual(4.5);
  });

  it("gives every group its own shape", () => {
    expect(new Set(Object.values(SHAPES)).size).toBe(Object.keys(SHAPES).length);
  });

  it("maps models to groups", () => {
    expect(groupOf({ id: "qlassifier", family: "quantum", role: "ours" })).toBe("ours");
    expect(groupOf({ id: "zz_qsvc", family: "quantum", role: "standard_qml" })).toBe("quantum");
    expect(groupOf({ id: "pqk", family: "quantum", role: "published_quantum" })).toBe("quantum");
    expect(groupOf({ id: "mlp", family: "classical", role: "classical" })).toBe("classical");
    expect(groupOf({ id: "classical_twin", family: "control", role: "twin" })).toBe("twin");
    expect(groupOf({ id: "trivial", family: "trivial", role: "trivial" })).toBe("trivial");
  });

  it("emits CSS variables for both themes", () => {
    const css = paletteCss();
    expect(css).toContain(":root{");
    expect(css).toContain(".dark{");
    expect(css).toContain(`--c-ours:${PALETTE.light.ours};`);
    expect(css).toContain(`--c-ours:${PALETTE.dark.ours};`);
  });

  it("parses colours and rejects unknown formats", () => {
    expect(parseColor("#fff")).toEqual([255, 255, 255, 1]);
    expect(parseColor("rgba(1, 2, 3, 0.5)")).toEqual([1, 2, 3, 0.5]);
    expect(() => parseColor("hsl(0 0% 0%)")).toThrow();
    expect(contrast("#000000", "#ffffff")).toBeCloseTo(21, 5);
  });
});
