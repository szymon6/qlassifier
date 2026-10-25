import { cn } from "cn";
import type { ReactNode } from "react";

export interface SrTable {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
}

interface ChartFrameProps {
  title: ReactNode;
  legend?: ReactNode;
  table: SrTable;
  children: ReactNode;
  className?: string;
  testId: string;
  dataAttrs?: Record<`data-${string}`, string | number | boolean | undefined>;
}

/** One card surface per view + a visually hidden table with every number (TDD §8.6). */
export function ChartFrame({
  title,
  legend,
  table,
  children,
  className,
  testId,
  dataAttrs,
}: ChartFrameProps) {
  return (
    <section
      className={cn("card relative flex min-h-0 flex-col", className)}
      data-testid={testId}
      {...dataAttrs}
    >
      <header className="flex h-12 shrink-0 items-center justify-between gap-4 px-6 pt-1">
        <h2 className="label">{title}</h2>
        {legend ? <div className="flex items-center gap-3">{legend}</div> : null}
      </header>
      <div className="relative min-h-0 flex-1">{children}</div>
      <SrOnlyTable table={table} testId={`${testId}-table`} />
    </section>
  );
}

export function SrOnlyTable({ table, testId }: { table: SrTable; testId?: string }) {
  return (
    <div className="sr-only">
      <table data-testid={testId}>
        <caption>{table.caption}</caption>
        <thead>
          <tr>
            {table.columns.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) =>
                j === 0 ? (
                  <th key={j} scope="row">
                    {cell}
                  </th>
                ) : (
                  <td key={j}>{cell}</td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Legend entry: glyph + short label; `title` is the hover definition of the label (no text on screen). */
export function LegendItem({
  icon,
  label,
  title,
}: {
  icon: ReactNode;
  label: string;
  title?: string;
}) {
  return (
    <span
      className="flex items-center gap-1.5 text-[12px] leading-4 text-muted-foreground"
      title={title}
      aria-label={title}
    >
      {icon}
      {label}
    </span>
  );
}

/** When a view shows a cached study next to a live / session result. */
export interface Provenance {
  /** ISO 8601 time the cached study was measured, if known. */
  createdAt: string | null;
}

export function provenanceText(provenance: Provenance | null | undefined): string {
  if (!provenance) return "";
  const day = provenance.createdAt ? provenance.createdAt.slice(0, 10) : null;
  return day ? `cached study, ${day}` : "cached study";
}

/** A chart's aria summary, with the provenance sentence when the study is cached. */
export function withProvenance(summary: string, provenance: Provenance | null | undefined): string {
  return provenance
    ? `${summary} ${provenanceText(provenance)[0].toUpperCase()}${provenanceText(provenance).slice(1)}.`
    : summary;
}

/**
 * Card title; with `provenance`, hovering it shows where the numbers come from
 * (no extra text on screen, TDD §8.5). Screen readers get it in the chart summary.
 */
export function ChartTitle({ text, provenance }: { text: string; provenance?: Provenance | null }) {
  if (!provenance) return <>{text}</>;
  return (
    <span
      className="group/title relative cursor-default"
      data-testid="provenance"
      data-source="shipped"
    >
      {text}
      <span
        aria-hidden="true"
        className="pointer-events-none absolute top-full left-0 z-30 mt-2 translate-y-1 rounded-full bg-band px-3 py-1 text-[12px] leading-[18px] whitespace-nowrap text-foreground opacity-0 transition-[opacity,transform] duration-150 group-hover/title:translate-y-0 group-hover/title:opacity-100"
      >
        {provenanceText(provenance)}
      </span>
    </span>
  );
}
