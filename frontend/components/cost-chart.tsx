"use client";

import { scaleLinear, scaleLog } from "d3-scale";
import { useId, useMemo } from "react";
import {
  ChartFrame,
  ChartTitle,
  withProvenance,
  type Provenance,
  type SrTable,
} from "./chart-frame";
import { buildCost, COST_DEVICE, type CostModelView, type CostRow } from "@/lib/cost";
import { DEVICE_LABELS } from "@/lib/chips";
import { fmtCompact, fmtInt } from "@/lib/format";
import { modelTitle, TERMS } from "@/lib/glossary";
import { useTween } from "@/lib/motion";
import { hueVar } from "@/lib/palette";
import type { CostStudy, ModelSummary } from "@/lib/types";
import { useSize } from "@/lib/use-size";

const LABEL_W = 116;
const HEAD_H = 18;
const BAR_H = 8;
const GAP = 22;
const MAX_ROW_H = 38;

interface CostChartProps {
  cost: CostStudy | null;
  models: ModelSummary[];
  focus: string | null;
  reduced: boolean;
  onHover: (id: string | null) => void;
  onTogglePin: (id: string) => void;
  /** Set when the study is the cached one next to a live or session result. */
  provenance?: Provenance | null;
}

export function costTable(view: CostModelView): SrTable {
  const devices = view.devices.length ? view.devices : [COST_DEVICE];
  return {
    caption:
      "Cost per quantum model: circuits to train and predict, on-device tuning circuits, qubits, two-qubit gates and depth per circuit",
    columns: [
      "Model",
      "Circuits (train + predict)",
      "On-device tuning circuits",
      "Qubits",
      ...devices.map((d) => `Two-qubit gates, ${DEVICE_LABELS[d] ?? d}`),
      ...devices.map((d) => `Depth, ${DEVICE_LABELS[d] ?? d}`),
    ],
    rows: view.rows.map((r) => [
      r.name,
      fmtInt(r.circuits),
      fmtInt(r.tuning),
      r.qubits,
      ...devices.map((d) => (r.gatesByDevice[d] != null ? fmtInt(r.gatesByDevice[d]) : "—")),
      ...devices.map((d) => (r.depthByDevice[d] != null ? fmtInt(r.depthByDevice[d]) : "—")),
    ]),
  };
}

export function costSummary(view: CostModelView): string {
  const ours = view.rows.find((r) => r.id === "qlassifier");
  if (!ours) return "Cost study: not available for this result.";
  const most = view.rows.reduce((a, b) => (b.total > a.total ? b : a), ours);
  return `Qlassifier needs ${fmtInt(ours.total)} circuits on ${ours.qubits} qubits with ${ours.gates ?? 0} two-qubit gates per circuit; ${most.name} needs ${fmtInt(most.total)}.`;
}

export function CostChart({
  cost,
  models,
  focus,
  reduced,
  onHover,
  onTogglePin,
  provenance,
}: CostChartProps) {
  const view = useMemo(() => buildCost(cost, models), [cost, models]);
  // Dim the others only when the focused model is drawn here (linked focus, TDD §8.3).
  const focused = focus != null && view.rows.some((r) => r.id === focus) ? focus : null;
  const [ref, size] = useSize<HTMLDivElement>();
  const hatch = `hatch${useId().replace(/[^a-zA-Z0-9]/g, "")}`;

  const target = useMemo(() => {
    const r: Record<string, { i: number; c: number; t: number; q: number; g: number }> = {};
    view.rows.forEach((row, i) => {
      r[row.id] = { i, c: row.circuits, t: row.total, q: row.qubits, g: row.gates ?? 0 };
    });
    return { r, mc: view.maxCircuits, mq: view.maxQubits, mg: view.maxGates };
  }, [view]);
  const key = useMemo(() => JSON.stringify(target), [target]);
  const shown = useTween(target, key, !reduced);

  const w = size.width;
  const h = size.height;
  const n = view.rows.length;
  const rowH = n ? Math.min(MAX_ROW_H, (h - HEAD_H - 8) / n) : MAX_ROW_H;
  const free = Math.max(120, w - LABEL_W - 3 * GAP - 16);
  const cols = { c: free * 0.5, q: free * 0.25, g: free * 0.25 };
  const cx = LABEL_W + GAP;
  const qx = cx + cols.c + GAP;
  const gx = qx + cols.q + GAP;
  const VALUE_W = 34;
  const lowPow = Math.pow(10, Math.floor(Math.log10(Math.max(1, view.minCircuits))) - 1);
  const xc = scaleLog()
    .domain([lowPow, Math.max(lowPow * 10, shown.mc)])
    .range([0, cols.c - VALUE_W])
    .clamp(true);
  const xq = scaleLinear()
    .domain([0, Math.max(1, shown.mq)])
    .range([0, cols.q - VALUE_W]);
  const xg = scaleLinear()
    .domain([0, Math.max(1, shown.mg)])
    .range([0, cols.g - VALUE_W]);

  const focusRow: CostRow | null = focused
    ? (view.rows.find((r) => r.id === focused) ?? null)
    : null;

  return (
    <ChartFrame
      testId="cost"
      className="min-h-0 flex-1"
      title={<ChartTitle text="cost · quantum models" provenance={provenance} />}
      table={costTable(view)}
      dataAttrs={{ "data-rows": n }}
    >
      <div ref={ref} className="absolute inset-0 mx-4 mb-3">
        {w > 0 && h > 0 && n === 0 ? (
          <div
            role="img"
            aria-label={withProvenance(costSummary(view), provenance)}
            className="grid h-full place-items-center text-[32px] font-semibold text-muted-foreground opacity-40"
          >
            {"\u2013"}
          </div>
        ) : w > 0 && h > 0 ? (
          <svg
            role="img"
            aria-label={withProvenance(costSummary(view), provenance)}
            tabIndex={0}
            width={w}
            height={h}
            className="overflow-visible rounded-[14px] outline-none focus-visible:ring-2 focus-visible:ring-foreground"
            onMouseLeave={() => onHover(null)}
          >
            <defs>
              <pattern
                id={hatch}
                width={5}
                height={5}
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <rect width={5} height={5} fill="var(--card)" />
                <line x1={0} y1={0} x2={0} y2={5} stroke="currentColor" strokeWidth={2.2} />
              </pattern>
            </defs>
            <g className="tick" fontSize={11} style={{ pointerEvents: "all" }}>
              <text x={cx} y={11} aria-label={TERMS.circuits}>
                <title>{TERMS.circuits}</title>
                circuits · log
              </text>
              <text x={qx} y={11} aria-label="qubits per circuit">
                <title>qubits per circuit</title>
                qubits
              </text>
              <text x={gx} y={11} aria-label={TERMS.twoQ}>
                <title>{TERMS.twoQ}</title>
                2Q gates · {DEVICE_LABELS[COST_DEVICE]}
              </text>
            </g>
            {view.rows.map((row) => {
              const t = shown.r[row.id];
              if (!t) return null;
              const top = HEAD_H + 8 + t.i * rowH;
              const mid = top + rowH / 2;
              const color = hueVar(row.group);
              const ours = row.id === "qlassifier";
              const dim = focused != null && focused !== row.id;
              const solid = xc(Math.max(lowPow, t.c));
              const total = xc(Math.max(lowPow, t.t));
              return (
                <g
                  key={row.id}
                  data-model={row.id}
                  data-focus={focused == null ? "none" : focused === row.id ? "focus" : "dim"}
                  opacity={dim ? 0.3 : 1}
                  style={{
                    transition: reduced ? undefined : "opacity 160ms ease-out",
                    cursor: "pointer",
                  }}
                  onMouseEnter={() => onHover(row.id)}
                  onClick={() => onTogglePin(row.id)}
                >
                  <rect x={0} y={top} width={w} height={rowH} fill="transparent" />
                  <text
                    x={0}
                    y={mid + 4}
                    fontSize={12}
                    className={ours ? "font-semibold" : "font-medium"}
                    fill={ours ? "var(--c-ours-text)" : "var(--foreground)"}
                    aria-label={modelTitle(row.id, row.name)}
                  >
                    <title>{modelTitle(row.id, row.name)}</title>
                    {row.name}
                  </text>
                  <rect
                    x={cx}
                    y={mid - BAR_H / 2}
                    width={Math.max(2, solid)}
                    height={BAR_H}
                    rx={BAR_H / 2}
                    fill={color}
                  />
                  {total > solid + 1 ? (
                    <g color={color}>
                      <rect
                        data-testid={`tuning-${row.id}`}
                        x={cx + solid - BAR_H / 2}
                        y={mid - BAR_H / 2}
                        width={total - solid + BAR_H / 2}
                        height={BAR_H}
                        rx={BAR_H / 2}
                        fill={`url(#${hatch})`}
                        stroke={color}
                        strokeWidth={1}
                      />
                    </g>
                  ) : null}
                  <text className="tick" x={cx + Math.max(2, total) + 6} y={mid + 4}>
                    {fmtCompact(row.total)}
                  </text>
                  <rect
                    x={qx}
                    y={mid - BAR_H / 2}
                    width={Math.max(2, xq(t.q))}
                    height={BAR_H}
                    rx={BAR_H / 2}
                    fill={color}
                  />
                  <text className="tick" x={qx + Math.max(2, xq(t.q)) + 6} y={mid + 4}>
                    {row.qubits}
                  </text>
                  {t.g > 0 ? (
                    <rect
                      x={gx}
                      y={mid - BAR_H / 2}
                      width={Math.max(2, xg(t.g))}
                      height={BAR_H}
                      rx={BAR_H / 2}
                      fill={color}
                    />
                  ) : (
                    <circle cx={gx + 3} cy={mid} r={3} fill={color} />
                  )}
                  <text className="tick" x={gx + Math.max(6, xg(t.g)) + 6} y={mid + 4}>
                    {row.gates != null ? fmtInt(row.gates) : "—"}
                  </text>
                </g>
              );
            })}
          </svg>
        ) : null}
        {focusRow && w > 0 ? <DeviceCard row={focusRow} devices={view.devices} /> : null}
      </div>
    </ChartFrame>
  );
}

function DeviceCard({ row, devices }: { row: CostRow; devices: string[] }) {
  if (!devices.length) return null;
  return (
    <p
      aria-hidden="true"
      data-testid="cost-card"
      data-model={row.id}
      className="pointer-events-none absolute -top-[38px] right-2 z-30 flex items-center gap-3 rounded-full bg-band px-3 py-1 text-[12px] leading-[18px] whitespace-nowrap"
    >
      <span className="font-semibold text-foreground">2Q</span>
      {devices.map((d) => (
        <span key={d} className="text-muted-foreground">
          {DEVICE_LABELS[d] ?? d}{" "}
          <span className="font-medium text-foreground">{row.gatesByDevice[d] ?? "—"}</span>
        </span>
      ))}
    </p>
  );
}
