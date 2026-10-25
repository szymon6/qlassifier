// Okabe–Ito hues + shapes per model group (TDD §8.5). Colour is never the only cue.
// Single source of truth for theme colours: injected as CSS variables by app/layout.tsx
// and contrast-checked in tests/palette.test.ts (WCAG AA: text ≥ 4.5:1, marks ≥ 3:1).

export type Group = "ours" | "quantum" | "classical" | "twin" | "trivial";
export type Shape = "circle" | "diamond" | "square" | "ring" | "rule";
export type Theme = "light" | "dark";

export interface ModelLike {
  id: string;
  family: string;
  role: string;
}

export function groupOf(model: ModelLike): Group {
  if (model.role === "ours" || model.id === "qlassifier") return "ours";
  if (model.role === "twin" || model.id === "classical_twin") return "twin";
  if (model.role === "trivial" || model.family === "trivial") return "trivial";
  if (model.family === "quantum") return "quantum";
  return "classical";
}

export const SHAPES: Record<Group, Shape> = {
  ours: "circle",
  quantum: "diamond",
  classical: "square",
  twin: "ring",
  trivial: "rule",
};

export interface ThemeTokens {
  background: string;
  card: string;
  foreground: string;
  muted: string;
  faint: string;
  hairline: string;
  band: string;
  ours: string;
  oursText: string;
  oursSoft: string;
  quantum: string;
  classical: string;
  twin: string;
  trivial: string;
}

export const PALETTE: Record<Theme, ThemeTokens> = {
  light: {
    background: "#f4f5f7",
    card: "#ffffff",
    foreground: "#0e1116",
    muted: "#5e6470",
    faint: "#656b78",
    hairline: "rgba(14, 17, 22, 0.08)",
    band: "rgba(14, 17, 22, 0.035)",
    ours: "#d55e00",
    oursText: "#a84300",
    oursSoft: "rgba(213, 94, 0, 0.07)",
    quantum: "#cc79a7",
    classical: "#0072b2",
    twin: "#8a8a8a",
    trivial: "#8a8a8a",
  },
  dark: {
    background: "#0b0c0f",
    card: "#14161b",
    foreground: "#ecedef",
    muted: "#9aa0aa",
    faint: "#8c929c",
    hairline: "rgba(255, 255, 255, 0.08)",
    band: "rgba(255, 255, 255, 0.045)",
    ours: "#ff8c42",
    oursText: "#ff9a57",
    oursSoft: "rgba(255, 140, 66, 0.10)",
    quantum: "#e59cc6",
    classical: "#56b4e9",
    twin: "#9a9a9a",
    trivial: "#9a9a9a",
  },
};

/** CSS variable holding the hue of a group, e.g. `var(--c-ours)`. */
export function hueVar(group: Group): string {
  return `var(--c-${group})`;
}

const VAR_NAMES: Record<keyof ThemeTokens, string> = {
  background: "--background",
  card: "--card",
  foreground: "--foreground",
  muted: "--muted-foreground",
  faint: "--faint",
  hairline: "--hairline",
  band: "--band",
  ours: "--c-ours",
  oursText: "--c-ours-text",
  oursSoft: "--c-ours-soft",
  quantum: "--c-quantum",
  classical: "--c-classical",
  twin: "--c-twin",
  trivial: "--c-trivial",
};

function block(tokens: ThemeTokens): string {
  return (Object.keys(VAR_NAMES) as (keyof ThemeTokens)[])
    .map((key) => `${VAR_NAMES[key]}:${tokens[key]};`)
    .join("");
}

/** CSS text with the theme variables for `:root` (light) and `.dark`. */
export function paletteCss(): string {
  return `:root{${block(PALETTE.light)}}.dark{${block(PALETTE.dark)}}`;
}

// --- WCAG contrast -------------------------------------------------------------------------

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** Parses #rgb / #rrggbb / rgba(r, g, b, a). */
export function parseColor(color: string): [number, number, number, number] {
  const hex = color.trim().toLowerCase();
  if (hex.startsWith("#")) {
    const body = hex.slice(1);
    const full = body.length === 3 ? [...body].map((c) => c + c).join("") : body;
    const n = Number.parseInt(full, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1];
  }
  const match = hex.match(/rgba?\(([^)]+)\)/);
  if (!match) throw new Error(`Unsupported colour: ${color}`);
  const parts = match[1].split(",").map((p) => Number.parseFloat(p));
  return [parts[0], parts[1], parts[2], parts[3] ?? 1];
}

/** Composites a (possibly translucent) colour over an opaque background. */
export function over(color: string, background: string): [number, number, number] {
  const [r, g, b, a] = parseColor(color);
  const [br, bg, bb] = parseColor(background);
  return [r * a + br * (1 - a), g * a + bg * (1 - a), b * a + bb * (1 - a)];
}

export function luminance([r, g, b]: [number, number, number]): number {
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export function contrast(fg: string, bg: string): number {
  const background = over(bg, "#ffffff");
  const toHex = (c: [number, number, number]) =>
    `#${c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
  const front = over(fg, toHex(background));
  const a = luminance(front);
  const b = luminance(background);
  const [hi, lo] = a > b ? [a, b] : [b, a];
  return (hi + 0.05) / (lo + 0.05);
}
