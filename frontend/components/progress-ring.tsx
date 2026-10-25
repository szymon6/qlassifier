interface ProgressRingProps {
  /** 0–1, or null for an empty track. */
  value: number | null;
  size: number;
  stroke: number;
  className?: string;
  trackClassName?: string;
  /** Dashed outline instead of a filled arc (stopped run). */
  hollow?: boolean;
}

/** Circular progress arc; continuous fill via a stroke-dashoffset transition. */
export function ProgressRing({
  value,
  size,
  stroke,
  className,
  trackClassName,
  hollow,
}: ProgressRingProps) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-hidden="true"
      focusable="false"
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        strokeWidth={stroke}
        className={trackClassName}
        stroke="currentColor"
        strokeDasharray={hollow ? `${c / 48} ${c / 48}` : undefined}
      />
      {hollow ? null : (
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          strokeWidth={stroke}
          strokeLinecap="round"
          stroke="currentColor"
          className={className}
          strokeDasharray={c}
          strokeDashoffset={c * (1 - v)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 400ms linear" }}
        />
      )}
    </svg>
  );
}
