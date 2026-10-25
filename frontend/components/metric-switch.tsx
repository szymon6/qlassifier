"use client";

import { RadioGroup } from "./radio-group";
import { METRIC_TITLES } from "@/lib/glossary";
import { METRICS, type MetricCode } from "@/lib/url-state";

/** Control 2: segmented BA · ACC · AUC · F1 (TDD §8.2). */
export function MetricSwitch({
  value,
  onChange,
}: {
  value: MetricCode;
  onChange: (m: MetricCode) => void;
}) {
  return (
    <RadioGroup
      label="Metric"
      value={value}
      onChange={onChange}
      className="flex items-center gap-0.5 rounded-full bg-band p-1"
      itemClassName="h-8 min-w-12 rounded-full px-3 text-[12px] font-semibold tracking-[0.02em] text-muted-foreground transition-[background-color,color,box-shadow] duration-200 hover:text-foreground data-[state=on]:bg-card data-[state=on]:text-foreground data-[state=on]:shadow-[0_1px_2px_rgba(0,0,0,0.08),0_0_0_1px_var(--hairline)]"
      options={METRICS.map((m) => ({
        value: m.code,
        label: m.name,
        title: METRIC_TITLES[m.code],
        content: m.label,
      }))}
    />
  );
}
