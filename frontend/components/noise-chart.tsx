"use client";

import { scaleLinear, scalePoint } from "d3-scale";
import { line } from "d3-shape";
import { useMemo } from "react";
import {
  ChartFrame,
  ChartTitle,
  withProvenance,
  type Provenance,
  type SrTable,
} from "./chart-frame";
import { Mark } from "./mark";
import { fmtPpPlain, tickFormatter } from "@/lib/format";
import { modelTitle, TERMS } from "@/lib/glossary";
import { useTween } from "@/lib/motion";
import { buildNoise, spreadLabels, type NoiseModel, type NoiseSeries } from "@/lib/noise";
import { hueVar, SHAPES } from "@/lib/palette";
import type { ModelSummary, NoiseStudy } from "@/lib/types";
import { useSize } from "@/lib/use-size";

const M = { top: 14, right: 132, bottom: 30, left: 44 };
const LABEL_GAP = 15;
const MIN_SPAN_PP = 5;

interface NoiseChartProps {
  noise: NoiseStudy | null;
  models: ModelSummary[];
  focus: string | null;
  reduced: boolean;
  onHover: (id: string | null) => void;
  onTogglePin: (id: string) => void;
  /** Set when the study is the cached one next to a live or session result. */
  provenance?: Provenance | null;
}

function seriesLabel(s: NoiseSeries): string {
  return s.subset ? `${s.name} 60/40` : s.name;
}

export function noiseTable(view: NoiseModel): SrTable {
  return {
    caption: "Accuracy lost vs exact simulation, percentage points (mean over seeds; worst seed)",
    columns: ["Series", ...view.modes.map((m) => m.label)],
    rows: view.series.map((s) => [
      seriesLabel(s),
      ...view.modes.map((m) => {
        const p = s.points.find((q) => q.mode === m.mode);
        return p ? `${fmtPpPlain(p.drop, 2)} (worst ${fmtPpPlain(p.worst, 2)})` : "—";
      }),
    ]),
  };
}

export function noiseSummary(view: NoiseModel): string {
  const ours = view.series.find((s) => s.model === "qlassifier" && !s.subset);
  if (!ours) return "Noise study: not available for this result.";
  // Worst over the noisy modes (exact is 0 by definition).
  const worst = Math.max(...ours.points.filter((p) => p.mode !== "exact").map((p) => p.drop));
  const others = view.series.filter((s) => s.subset && s.model !== "qlassifier");
  const tail = others.length
    ? `; on the 60/40 subset ${others.map((s) => `${s.name} loses up to ${fmtPpPlain(Math.max(...s.points.map((p) => p.drop)))}`).join(", ")}`
    : "";
  const head =
    worst > 0
      ? `Qlassifier loses at most ${fmtPpPlain(worst)} accuracy under shot and IBM device noise`
      : `Qlassifier loses no accuracy under shot and IBM device noise (worst case ${fmtPpPlain(worst)})`;
  return `${head}${tail}.`;
}

export function NoiseChart({
  noise,
  models,
  focus,
  reduced,
  onHover,
  onTogglePin,
  provenance,
}: NoiseChartProps) {
  const view = useMemo(() => buildNoise(noise, models), [noise, models]);
  // Dim the others only when the focused model is drawn here (linked focus, TDD §8.3).
  const focused = focus != null && view.series.some((x) => x.model === focus) ? focus : null;
  const [ref, size] = useSize<HTMLDivElement>();

  const target = useMemo(() => {
    const s: Record<string, number[]> = {};
    for (const series of view.series)
      s[series.key] = view.modes.map(
        (m) => series.points.find((p) => p.mode === m.mode)?.drop ?? NaN,
      );
    // At least 5 pp of loss on the axis: sub-point wiggles (both signs) stay visibly small.
    const max = Math.max(MIN_SPAN_PP, view.maxDrop);
    return { y: [Math.min(0, view.minDrop), max] as [number, number], s };
  }, [view]);
  const key = useMemo(() => JSON.stringify(target), [target]);
  const shown = useTween(target, key, !reduced);

  const w = size.width;
  const h = size.height;
  const x = scalePoint<string>()
    .domain(view.modes.map((m) => m.mode))
    .range([M.left + 6, Math.max(M.left + 60, w - M.right)]);
  // 0 on top; only the far end is rounded so a small gain under noise (drop < 0) stays a sliver.
  const top = scaleLinear().domain([0, shown.y[1]]).nice(4).domain()[1];
  const y = scaleLinear()
    .domain([shown.y[0], top])
    .range([M.top, Math.max(M.top + 20, h - M.bottom)]);
  // Ticks on both sides of 0: a gain under noise (drop < 0) gets a labelled scale too.
  const yTicks = scaleLinear().domain([shown.y[0], top]).ticks(5);
  const yTick = tickFormatter(yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1);
  // Part B has no shots point: connect the measured points only.
  const draw = line<[number, number]>()
    .x((d) => d[0])
    .y((d) => y(d[1]));

  // Direct labels at line ends, de-overlapped.
  const ends = view.series.map((s) => {
    const values = shown.s[s.key] ?? [];
    let last = values.length - 1;
    while (last > 0 && !Number.isFinite(values[last])) last--;
    return y(values[last] ?? 0);
  });
  const labelY = spreadLabels(ends, LABEL_GAP, M.top, h - M.bottom);

  // Draw order: subsets first, Qlassifier all-rows on top.
  const order = view.series
    .map((s, i) => ({ s, i }))
    .sort(
      (a, b) =>
        Number(b.s.subset) - Number(a.s.subset) ||
        Number(a.s.model === "qlassifier") - Number(b.s.model === "qlassifier"),
    );

  return (
    <ChartFrame
      testId="noise"
      className="min-h-0 flex-1"
      title={<ChartTitle text="accuracy lost · pp" provenance={provenance} />}
      table={noiseTable(view)}
      dataAttrs={{ "data-series": view.series.length }}
    >
      <div ref={ref} className="absolute inset-0 mx-2 mb-1">
        {w > 0 && h > 0 && view.series.length === 0 ? (
          <div
            role="img"
            aria-label={withProvenance(noiseSummary(view), provenance)}
            className="grid h-full place-items-center text-[32px] font-semibold text-muted-foreground opacity-40"
          >
            {"\u2013"}
          </div>
        ) : w > 0 && h > 0 ? (
          <svg
            role="img"
            aria-label={withProvenance(noiseSummary(view), provenance)}
            tabIndex={0}
            width={w}
            height={h}
            className="overflow-visible rounded-[14px] outline-none focus-visible:ring-2 focus-visible:ring-foreground"
            onMouseLeave={() => onHover(null)}
          >
            {yTicks.map((t) => (
              <g key={t}>
                <line
                  x1={M.left}
                  x2={w - M.right + 16}
                  y1={y(t)}
                  y2={y(t)}
                  stroke={t === 0 ? "var(--faint)" : "var(--hairline)"}
                  strokeOpacity={t === 0 ? 0.45 : 1}
                />
                <text className="tick" x={M.left - 10} y={y(t) + 4} textAnchor="end">
                  {yTick(t)}
                </text>
              </g>
            ))}
            {view.modes.map((m) => (
              <text key={m.mode} className="tick" x={x(m.mode)} y={h - 8} textAnchor="middle">
                {m.label}
              </text>
            ))}
            {order.map(({ s, i }) => {
              const values = shown.s[s.key] ?? [];
              const pts = view.modes.map((m, k) => [x(m.mode) ?? 0, values[k]] as [number, number]);
              const ours = s.model === "qlassifier";
              const color = hueVar(s.group);
              const dim = focused != null && focused !== s.model;
              const path = draw(pts.filter((p) => Number.isFinite(p[1]))) ?? "";
              const end = pts.filter((p) => Number.isFinite(p[1])).at(-1);
              return (
                <g
                  key={s.key}
                  data-series={s.key}
                  data-model={s.model}
                  data-focus={focused == null ? "none" : focused === s.model ? "focus" : "dim"}
                  opacity={dim ? 0.3 : 1}
                  style={{
                    transition: reduced ? undefined : "opacity 160ms ease-out",
                    cursor: "pointer",
                  }}
                  onMouseEnter={() => onHover(s.model)}
                  onClick={() => onTogglePin(s.model)}
                >
                  <path d={path} fill="none" stroke="transparent" strokeWidth={14} />
                  <path
                    d={path}
                    fill="none"
                    stroke={color}
                    strokeWidth={s.subset ? 1.5 : 2.6}
                    strokeDasharray={s.subset ? "4 3" : undefined}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  {pts.map(([px, py], k) =>
                    Number.isFinite(py) ? (
                      <Mark
                        key={k}
                        shape={SHAPES[s.group]}
                        x={px}
                        y={y(py)}
                        size={s.subset ? 6 : ours ? 9 : 7}
                        color={color}
                      />
                    ) : null,
                  )}
                  {end ? (
                    <text
                      x={end[0] + 12}
                      y={labelY[i] + 4}
                      className={ours && !s.subset ? "font-semibold" : "font-medium"}
                      fontSize={12}
                      fill={ours ? "var(--c-ours-text)" : "var(--foreground)"}
                      style={{ pointerEvents: "all" }}
                      aria-label={`${modelTitle(s.model, s.name)}${s.subset ? `; ${TERMS.subset}` : ""}`}
                    >
                      <title>
                        {modelTitle(s.model, s.name)}
                        {s.subset ? `; ${TERMS.subset}` : ""}
                      </title>
                      {s.name}
                      {s.subset ? (
                        <tspan fill="var(--muted-foreground)" fontWeight={500}>
                          {" "}
                          60/40
                        </tspan>
                      ) : null}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </svg>
        ) : null}
      </div>
    </ChartFrame>
  );
}
