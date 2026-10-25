"use client";

import { scaleLinear } from "d3-scale";
import { CornerDownRight } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { ChartFrame, LegendItem, type SrTable } from "./chart-frame";
import { Mark, MarkIcon } from "./mark";
import {
  clipUnit,
  fmtCi,
  fmtCiPp,
  fmtDiffPp,
  fmtMetric,
  fmtSeconds,
  fmtWins,
  tickFormatter,
} from "@/lib/format";
import { modelTitle, publishedTitle, TERMS } from "@/lib/glossary";
import { OURS_ID, TWIN_ID, type LeaderRow, type LeaderboardModel } from "@/lib/leaderboard";
import { useTween } from "@/lib/motion";
import { hueVar } from "@/lib/palette";
import { metricName, type MetricCode } from "@/lib/url-state";
import { useSize } from "@/lib/use-size";

const LABEL_W = 156;
const RIGHT_PAD = 28;
const AXIS_H = 30;
const MAX_ROW_H = 56;
const DOT_R = 2.6;

interface LeaderboardProps {
  model: LeaderboardModel;
  metric: MetricCode;
  dataset: string;
  status: string;
  focus: string | null;
  pinned: string | null;
  reduced: boolean;
  onHover: (id: string | null) => void;
  onTogglePin: (id: string) => void;
  onClear: () => void;
}

interface RowTarget {
  i: number;
  m: number | null;
  lo: number | null;
  hi: number | null;
  dots: Record<string, number>;
}

interface Target {
  d: [number, number];
  r: Record<string, RowTarget>;
}

function jitter(seed: number): number {
  // Deterministic vertical spread so seed dots do not stack (strip plot).
  return (((seed * 7919) % 11) - 5) * 0.9;
}

export function summaryText(model: LeaderboardModel, metric: MetricCode): string {
  const ours = model.ours;
  const name = metricName(metric);
  if (!ours || ours.mean == null) return `Leaderboard by ${name}.`;
  const ci =
    ours.ciLow != null && ours.ciHigh != null
      ? `, 95% CI ${fmtMetric(clipUnit(ours.ciLow), 2)} to ${fmtMetric(clipUnit(ours.ciHigh), 2)}`
      : "";
  const best = model.best;
  let relation = "";
  if (best && best.id === ours.id) relation = ", the best mean";
  else if (best)
    relation = `${ours.tied ? ", statistically tied with the best" : ", below the best"} (${best.name} ${fmtMetric(best.mean)})`;
  return `Qlassifier ${name} ${fmtMetric(ours.mean)}${ci}${relation}.`;
}

export function leaderboardTable(model: LeaderboardModel, metric: MetricCode): SrTable {
  return {
    caption: `Leaderboard, ${metricName(metric)}, sorted by mean`,
    columns: [
      "Model",
      "Mean",
      "SD",
      "95% CI low",
      "95% CI high",
      "Tied with best",
      "Seeds",
      "Qlassifier minus model (pp)",
      "95% CI of difference (pp)",
      "Qlassifier wins",
    ],
    rows: model.rows.map((r) => [
      r.name,
      fmtMetric(r.mean),
      fmtMetric(r.std),
      fmtMetric(clipUnit(r.ciLow)),
      fmtMetric(clipUnit(r.ciHigh)),
      r.tied ? "yes" : "no",
      r.n,
      r.vsOurs ? fmtDiffPp(r.vsOurs.mean_diff) : "—",
      r.vsOurs ? fmtCiPp(r.vsOurs.ci_low, r.vsOurs.ci_high) : "—",
      r.vsOurs ? fmtWins(r.vsOurs.win_rate, r.vsOurs.n) : "—",
    ]),
  };
}

function rowLabel(r: LeaderRow, metric: MetricCode): string {
  if (r.mean == null) return `${r.name}: no result yet`;
  const ci =
    r.ciLow != null && r.ciHigh != null
      ? `, 95% CI ${fmtMetric(clipUnit(r.ciLow))} to ${fmtMetric(clipUnit(r.ciHigh))}`
      : "";
  return `${r.name}: ${metricName(metric)} ${fmtMetric(r.mean)}${ci}${r.tied ? ", tied with the best" : ""}`;
}

export function Leaderboard(props: LeaderboardProps) {
  const { model, metric, dataset, status, focus, pinned, reduced, onHover, onTogglePin, onClear } =
    props;
  const [ref, size] = useSize<HTMLDivElement>();
  const summaryId = useId();
  // Gradient ids go into url(#…): keep them to safe characters.
  const gradientId = `ci${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const rowRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [tabStop, setTabStop] = useState<string | null>(null);
  const marked = useRef(false);

  const target = useMemo<Target>(() => {
    const r: Record<string, RowTarget> = {};
    model.rows.forEach((row, i) => {
      const dots: Record<string, number> = {};
      for (const dot of row.dots) dots[String(dot.seed)] = dot.value;
      r[row.id] = { i, m: row.mean, lo: row.ciLow, hi: row.ciHigh, dots };
    });
    return { d: model.domain, r };
  }, [model]);
  const key = useMemo(() => JSON.stringify(target), [target]);
  const shown = useTween(target, key, !reduced);

  const n = model.rows.length;
  const areaH = Math.max(0, size.height - AXIS_H);
  const rowH = n ? Math.min(MAX_ROW_H, areaH / n) : MAX_ROW_H;
  // The axis sits right under the last row (tall screens leave the slack below it).
  const rowsH = n ? rowH * n : areaH;
  const x = useMemo(
    () =>
      scaleLinear()
        .domain(shown.d)
        .range([LABEL_W + 12, Math.max(LABEL_W + 40, size.width - RIGHT_PAD)]),
    [shown.d, size.width],
  );
  const ticks = x.ticks(shown.d[1] - shown.d[0] > 0.3 ? 5 : 6);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 0.1;
  const tickText = tickFormatter(step);
  const [x0, x1] = x.range();
  const clamp = (v: number) => Math.max(x0, Math.min(x1, x(v)));
  const y = (i: number) => i * rowH + rowH / 2;
  const ready = size.width > 0 && n > 0 && model.rows.some((r) => r.mean != null);

  useEffect(() => {
    if (!ready || marked.current) return;
    marked.current = true;
    performance.mark("leaderboard-ready");
  }, [ready]);

  const order = model.rows.map((r) => r.id);
  const current =
    tabStop && order.includes(tabStop)
      ? tabStop
      : pinned && order.includes(pinned)
        ? pinned
        : order.includes(OURS_ID)
          ? OURS_ID
          : order[0];

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>, id: string) => {
    const at = order.indexOf(id);
    const go = (to: number) => {
      const next = order[(to + order.length) % order.length];
      setTabStop(next);
      rowRefs.current[next]?.focus();
    };
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      go(at + 1);
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      go(at - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      go(0);
    } else if (event.key === "End") {
      event.preventDefault();
      go(order.length - 1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onTogglePin(id);
    } else if (event.key === "Escape") {
      event.preventDefault();
      onClear();
    }
  };

  const focusRow = focus ? (model.rows.find((r) => r.id === focus) ?? null) : null;
  const focusIndex = focusRow ? (shown.r[focusRow.id]?.i ?? 0) : 0;

  return (
    <ChartFrame
      testId="leaderboard"
      className="h-full"
      dataAttrs={{
        "data-dataset": dataset,
        "data-metric": metric,
        "data-status": status,
        "data-ready": ready,
      }}
      title={
        <span title={TERMS.ci} aria-label={`${metricName(metric)}; ${TERMS.ci}`}>
          {`${metricName(metric)} · 95% CI`}
        </span>
      }
      legend={
        <>
          {model.bands.length ? (
            <LegendItem
              icon={
                <span className="h-3 w-4 rounded-[4px] bg-band shadow-[inset_0_0_0_1px_var(--hairline)]" />
              }
              label="≈ best"
              title="statistically tied with the best: the corrected 95% CI of the paired difference contains 0 (needs ≥ 5 seeds)"
            />
          ) : null}
          <LegendItem
            icon={<MarkIcon shape="diamond" color={hueVar("quantum")} size={9} />}
            label="quantum"
          />
          <LegendItem
            icon={<MarkIcon shape="square" color={hueVar("classical")} size={8} />}
            label="classical"
          />
          {model.majority != null ? (
            <LegendItem
              icon={<MarkIcon shape="rule" color={hueVar("trivial")} />}
              label="majority"
            />
          ) : null}
          {model.published ? (
            <LegendItem
              icon={
                <svg width={12} height={12} aria-hidden="true">
                  <path
                    d="M6,1L11,6L6,11L1,6Z"
                    fill="none"
                    stroke="var(--foreground)"
                    strokeWidth={1.3}
                  />
                </svg>
              }
              label="published"
              title={publishedTitle(model.published)}
            />
          ) : null}
        </>
      }
      table={leaderboardTable(model, metric)}
    >
      <p id={summaryId} className="sr-only">
        {summaryText(model, metric)}
      </p>
      <div ref={ref} className="absolute inset-0 mx-2 mb-2">
        {size.width > 0 ? (
          <>
            {/* Bands: tied with the best (neutral), Qlassifier (accent). */}
            {model.bands.map((band, k) => {
              const top = (shown.r[model.rows[band.start].id]?.i ?? band.start) * rowH;
              const bottom = ((shown.r[model.rows[band.end].id]?.i ?? band.end) + 1) * rowH;
              return (
                <div
                  key={`band-${k}`}
                  data-testid="tie-band"
                  data-start={model.rows[band.start].id}
                  data-end={model.rows[band.end].id}
                  className="absolute right-0 left-0 rounded-[14px] bg-band"
                  style={{ top, height: bottom - top }}
                />
              );
            })}
            {model.rows
              .filter((r) => r.id === OURS_ID)
              .map((r) => {
                const i = shown.r[r.id]?.i ?? 0;
                return (
                  <div
                    key="ours-band"
                    aria-hidden="true"
                    className="absolute right-0 left-0 rounded-[12px] bg-ours-soft"
                    style={{ top: i * rowH + 3, height: rowH - 6 }}
                  />
                );
              })}

            <svg
              className="pointer-events-none absolute inset-0 z-10 overflow-visible"
              width={size.width}
              height={size.height}
              aria-hidden="true"
              focusable="false"
            >
              <g>
                {ticks.map((t) => (
                  <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={rowsH} stroke="var(--hairline)" />
                ))}
                {model.majority != null && model.majority >= shown.d[0] ? (
                  <line
                    data-testid="majority-rule"
                    x1={x(model.majority)}
                    x2={x(model.majority)}
                    y1={0}
                    y2={rowsH}
                    stroke={hueVar("trivial")}
                    strokeWidth={1.5}
                    strokeDasharray="3 4"
                  />
                ) : null}
              </g>
              {model.rows.map((row) => {
                const t = shown.r[row.id];
                if (!t) return null;
                const cy = y(t.i);
                const color = hueVar(row.group);
                const ours = row.id === OURS_ID;
                const dim = focus != null && focus !== row.id;
                return (
                  <g
                    key={row.id}
                    data-model={row.id}
                    opacity={dim ? 0.3 : 1}
                    style={{ transition: reduced ? undefined : "opacity 160ms ease-out" }}
                  >
                    {Object.entries(t.dots).map(([seed, v]) => (
                      <circle
                        key={seed}
                        data-dot={seed}
                        cx={clamp(v)}
                        cy={cy + jitter(Number(seed))}
                        r={DOT_R}
                        fill={color}
                        opacity={0.32}
                      />
                    ))}
                    {t.lo != null && t.hi != null ? (
                      <CiBar
                        id={`${gradientId}-${row.id}`}
                        model={row.id}
                        a={clamp(t.lo)}
                        b={clamp(t.hi)}
                        clipLow={x(t.lo) < x0}
                        clipHigh={x(t.hi) > x1}
                        y={cy}
                        color={color}
                        width={ours ? 3 : 2.4}
                      />
                    ) : null}
                    {t.m != null ? (
                      <Mark
                        shape={row.shape}
                        x={clamp(t.m)}
                        y={cy}
                        size={ours ? 13 : 10}
                        color={color}
                      />
                    ) : null}
                  </g>
                );
              })}
              <g transform={`translate(0, ${rowsH})`}>
                <line x1={x0} x2={x1} y1={0.5} y2={0.5} stroke="var(--hairline)" />
                {ticks.map((t) => (
                  <text key={t} className="tick" x={x(t)} y={20} textAnchor="middle">
                    {tickText(t)}
                  </text>
                ))}
                {model.published &&
                model.published.value >= shown.d[0] &&
                model.published.value <= shown.d[1] ? (
                  <path
                    data-testid="published-mark"
                    d={`M${x(model.published.value)},${-6}l6,6l-6,6l-6,-6Z`}
                    fill="var(--card)"
                    stroke="var(--foreground)"
                    strokeWidth={1.4}
                    style={{ pointerEvents: "all" }}
                    aria-label={publishedTitle(model.published)}
                  >
                    <title>{publishedTitle(model.published)}</title>
                  </path>
                ) : null}
              </g>
            </svg>

            <div
              role="listbox"
              aria-label={`Leaderboard, ${metricName(metric)}`}
              aria-describedby={summaryId}
              className="absolute inset-x-0 top-0 z-20"
              style={{ height: rowsH }}
              onMouseLeave={() => onHover(null)}
            >
              {model.rows.map((row) => {
                const i = shown.r[row.id]?.i ?? 0;
                const ours = row.id === OURS_ID;
                const dim = focus != null && focus !== row.id;
                return (
                  <div
                    key={row.id}
                    ref={(el) => {
                      rowRefs.current[row.id] = el;
                    }}
                    role="option"
                    aria-selected={pinned === row.id}
                    aria-label={rowLabel(row, metric)}
                    tabIndex={row.id === current ? 0 : -1}
                    data-testid={`row-${row.id}`}
                    data-model={row.id}
                    data-mean={row.mean ?? ""}
                    data-ci-width={
                      row.ciLow != null && row.ciHigh != null ? row.ciHigh - row.ciLow : ""
                    }
                    data-n={row.n}
                    data-dots={row.dots.length}
                    data-tied={row.tied}
                    data-emphasis={ours ? "ours" : undefined}
                    data-focus={focus == null ? "none" : focus === row.id ? "focus" : "dim"}
                    data-pinned={pinned === row.id}
                    className="absolute inset-x-0 flex items-center rounded-[12px] outline-none focus-visible:ring-2 focus-visible:ring-foreground"
                    style={{ top: i * rowH, height: rowH }}
                    onMouseEnter={() => onHover(row.id)}
                    onFocus={() => {
                      setTabStop(row.id);
                      onHover(row.id);
                    }}
                    onBlur={() => onHover(null)}
                    onClick={() => onTogglePin(row.id)}
                    onKeyDown={(event) => onKeyDown(event, row.id)}
                  >
                    <span
                      className={
                        "flex items-center gap-1.5 truncate pl-4 text-[13px] leading-4 transition-opacity duration-150 " +
                        (ours ? "font-semibold text-ours-text" : "font-medium text-foreground") +
                        (dim ? " opacity-30" : "") +
                        (row.mean == null ? " text-muted-foreground" : "")
                      }
                      style={{ width: LABEL_W }}
                    >
                      {row.id === TWIN_ID ? (
                        <CornerDownRight
                          className="size-3.5 shrink-0 text-muted-foreground"
                          aria-hidden="true"
                        />
                      ) : null}
                      <span className="truncate" title={modelTitle(row.id, row.name)}>
                        {row.name}
                      </span>
                      {pinned === row.id ? (
                        <span
                          className="size-1.5 shrink-0 rounded-full bg-foreground"
                          aria-hidden="true"
                        />
                      ) : null}
                    </span>
                  </div>
                );
              })}
            </div>

            {focusRow && focusRow.mean != null ? (
              <HoverCard
                row={focusRow}
                x={clamp(shown.r[focusRow.id]?.m ?? focusRow.mean)}
                y={y(focusIndex)}
                width={size.width}
                above={y(focusIndex) > rowsH * 0.55}
              />
            ) : null}
          </>
        ) : null}
      </div>
    </ChartFrame>
  );
}

interface CiBarProps {
  id: string;
  model: string;
  a: number;
  b: number;
  /** The interval runs past the axis: fade that end out (few seeds give very wide CIs). */
  clipLow: boolean;
  clipHigh: boolean;
  y: number;
  color: string;
  width: number;
}

function CiBar({ id, model, a, b, clipLow, clipHigh, y, color, width }: CiBarProps) {
  const faded = clipLow || clipHigh;
  return (
    <>
      {faded ? (
        <defs>
          <linearGradient id={id} gradientUnits="userSpaceOnUse" x1={a} x2={b} y1={y} y2={y}>
            <stop offset="0" stopColor={color} stopOpacity={clipLow ? 0 : 1} />
            <stop offset={clipLow ? 0.12 : 0} stopColor={color} stopOpacity={1} />
            <stop offset={clipHigh ? 0.88 : 1} stopColor={color} stopOpacity={1} />
            <stop offset="1" stopColor={color} stopOpacity={clipHigh ? 0 : 1} />
          </linearGradient>
        </defs>
      ) : null}
      <line
        data-ci={model}
        data-clipped={faded || undefined}
        x1={a}
        x2={b}
        y1={y}
        y2={y}
        stroke={faded ? `url(#${id})` : color}
        strokeWidth={width}
        strokeLinecap={faded ? "butt" : "round"}
        opacity={0.75}
      />
    </>
  );
}

interface HoverCardProps {
  row: LeaderRow;
  x: number;
  y: number;
  width: number;
  /** Lower rows open the card upwards so it stays inside the leaderboard. */
  above: boolean;
}

function HoverCard({ row, x, y, width, above }: HoverCardProps) {
  const W = 212;
  const left = x + 20 + W > width ? x - 20 - W : x + 20;
  const items: [string, string][] = [
    ["mean ± sd", `${fmtMetric(row.mean)} ± ${fmtMetric(row.std)}`],
    ["95% CI", fmtCi(row.ciLow, row.ciHigh)],
  ];
  if (row.vsOurs) {
    items.push(["Qlassifier Δ", `${fmtDiffPp(row.vsOurs.mean_diff)}`]);
    items.push(["Δ 95% CI", fmtCiPp(row.vsOurs.ci_low, row.vsOurs.ci_high)]);
    items.push(["Qlassifier wins", fmtWins(row.vsOurs.win_rate, row.vsOurs.n)]);
  }
  items.push(["seeds", String(row.n)]);
  items.push(["time / seed", fmtSeconds(row.seconds)]);
  return (
    <div
      aria-hidden="true"
      data-testid="hover-card"
      data-model={row.id}
      className="card pointer-events-none absolute z-30 px-4 py-3"
      style={
        above
          ? { left, top: y + 24, width: W, transform: "translateY(-100%)" }
          : { left, top: Math.max(0, y - 24), width: W }
      }
    >
      <p className="mb-1.5 text-[12px] leading-4 font-semibold text-foreground">{row.name}</p>
      <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 text-[12px] leading-[18px]">
        {items.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right font-medium whitespace-nowrap text-foreground">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
