import type { Glyph } from "@/lib/types";

const PAD = 0.1;

/** Data-shape picture of a dataset: PCA scatter or Ramachandran square (TDD §8.2). */
export function DatasetGlyph({ glyph, size = 36 }: { glyph: Glyph; size?: number }) {
  const span = 1 - 2 * PAD;
  const r = glyph.points.length > 80 ? 0.026 : 0.032;
  // Draw the positive class last so it sits on top.
  const points = [...glyph.points].sort((a, b) => a[2] - b[2]);
  return (
    <svg
      viewBox="0 0 1 1"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className="block overflow-visible"
    >
      {glyph.kind === "torus" ? (
        <g stroke="currentColor" strokeWidth={0.012} opacity={0.22} fill="none">
          <rect x={PAD} y={PAD} width={span} height={span} rx={0.04} strokeDasharray="0.03 0.03" />
          <line x1={0.5} y1={PAD} x2={0.5} y2={1 - PAD} />
          <line x1={PAD} y1={0.5} x2={1 - PAD} y2={0.5} />
        </g>
      ) : null}
      {points.map(([x, y, c], i) => (
        <circle
          key={i}
          cx={PAD + x * span}
          cy={PAD + (1 - y) * span}
          r={c ? r : r * 0.9}
          fill="currentColor"
          opacity={c ? 0.9 : 0.3}
        />
      ))}
    </svg>
  );
}
