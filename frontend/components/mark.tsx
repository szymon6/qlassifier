import type { Shape } from "@/lib/palette";

interface MarkProps {
  shape: Shape;
  x: number;
  y: number;
  /** Outer size in px. */
  size: number;
  color: string;
  /** Halo in the card colour so marks read on top of bars and dots. */
  halo?: boolean;
  className?: string;
}

/** Mean mark by family: ● ours, ◆ quantum, ■ classical, ○ twin (TDD §8.5). */
export function Mark({ shape, x, y, size, color, halo = true, className }: MarkProps) {
  const s = size / 2;
  const stroke = halo
    ? { stroke: "var(--card)", strokeWidth: 2, paintOrder: "stroke" as const }
    : {};
  switch (shape) {
    case "diamond": {
      const d = s * 1.18;
      return (
        <path
          className={className}
          d={`M${x},${y - d}L${x + d},${y}L${x},${y + d}L${x - d},${y}Z`}
          fill={color}
          {...stroke}
        />
      );
    }
    case "square":
      return (
        <rect
          className={className}
          x={x - s * 0.9}
          y={y - s * 0.9}
          width={s * 1.8}
          height={s * 1.8}
          rx={1.5}
          fill={color}
          {...stroke}
        />
      );
    case "ring":
      return (
        <circle
          className={className}
          cx={x}
          cy={y}
          r={s - 1}
          fill="var(--card)"
          stroke={color}
          strokeWidth={2.2}
        />
      );
    default:
      return <circle className={className} cx={x} cy={y} r={s} fill={color} {...stroke} />;
  }
}

/** Small legend glyph (HTML inline SVG). */
export function MarkIcon({
  shape,
  color,
  size = 10,
}: {
  shape: Shape;
  color: string;
  size?: number;
}) {
  const box = size + 4;
  if (shape === "rule") {
    return (
      <svg width={box} height={box} aria-hidden="true" focusable="false">
        <line
          x1={box / 2}
          x2={box / 2}
          y1={1}
          y2={box - 1}
          stroke={color}
          strokeWidth={1.5}
          strokeDasharray="2 2"
        />
      </svg>
    );
  }
  return (
    <svg width={box} height={box} aria-hidden="true" focusable="false">
      <Mark shape={shape} x={box / 2} y={box / 2} size={size} color={color} halo={false} />
    </svg>
  );
}
