"use client";

import { DatasetGlyph } from "./dataset-glyph";
import { ProgressRing } from "./progress-ring";
import { RadioGroup } from "./radio-group";
import type { DatasetCard } from "@/lib/types";

const SHORT: Record<string, string> = {
  breast_cancer: "Breast",
  ionosphere: "Iono",
  sonar: "Sonar",
  heart: "Heart",
  protein_angles: "Protein",
};

export function shortName(card: Pick<DatasetCard, "id" | "name">): string {
  return SHORT[card.id] ?? card.name.split(/\s+/)[0];
}

interface DatasetTilesProps {
  datasets: DatasetCard[];
  value: string;
  onChange: (id: string) => void;
  running: { dataset: string; progress: number } | null;
}

/** Control 1: one radio group of glyph tiles (TDD §8.2). */
export function DatasetTiles({ datasets, value, onChange, running }: DatasetTilesProps) {
  return (
    <RadioGroup
      label="Dataset"
      value={value}
      onChange={onChange}
      className="flex items-start gap-1"
      itemClassName="group relative flex w-[60px] flex-col items-center gap-1 rounded-[14px] pt-1 pb-0.5 outline-none focus-visible:ring-2 focus-visible:ring-foreground focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      options={datasets.map((card) => ({
        value: card.id,
        label: card.name,
        content: (
          <>
            <span
              data-testid={`tile-${card.id}`}
              className="relative grid size-11 place-items-center rounded-[12px] text-foreground transition-[background-color,box-shadow] duration-200 group-hover:bg-band group-data-[state=on]:bg-card group-data-[state=on]:shadow-[0_0_0_1px_var(--foreground),0_10px_24px_-10px_rgba(0,0,0,0.35)]"
            >
              <DatasetGlyph glyph={card.glyph} size={36} />
              {running?.dataset === card.id ? (
                <span
                  className="absolute -top-1 -right-1 rounded-full bg-card text-ours"
                  data-testid="tile-running"
                >
                  <ProgressRing
                    value={running.progress}
                    size={14}
                    stroke={2.5}
                    trackClassName="opacity-20"
                  />
                </span>
              ) : null}
            </span>
            <span className="text-[11px] leading-[14px] font-medium text-muted-foreground group-data-[state=on]:text-foreground">
              {shortName(card)}
            </span>
          </>
        ),
      }))}
    />
  );
}
