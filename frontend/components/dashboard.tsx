"use client";

import { useEffect, useMemo, useState } from "react";
import { CostChart } from "./cost-chart";
import { DatasetTiles } from "./dataset-tiles";
import { Leaderboard } from "./leaderboard";
import { MetricSwitch } from "./metric-switch";
import { NoiseChart } from "./noise-chart";
import { ProofChips } from "./proof-chips";
import { ThemeToggle } from "./theme-toggle";
import { Toaster } from "./toaster";
import { TopBar } from "./top-bar";
import { TooltipProvider } from "./ui/tooltip";
import { configsTitle, RECIPE_TITLES, seedsTitle } from "@/lib/glossary";
import { buildChips } from "@/lib/chips";
import { buildLeaderboard } from "@/lib/leaderboard";
import { useReducedMotion } from "@/lib/motion";
import { activeNoise, activeResult, datasetCard, focusModel, isRunActive } from "@/lib/state";
import type { BenchmarkResult, DatasetCard, Meta } from "@/lib/types";
import { metricKey } from "@/lib/url-state";
import { useDashboard } from "@/lib/use-dashboard";

function useDelayed(flag: boolean, ms: number): boolean {
  const [late, setLate] = useState(false);
  useEffect(() => {
    if (!flag) return;
    const t = window.setTimeout(() => setLate(true), ms);
    return () => window.clearTimeout(t);
  }, [flag, ms]);
  return flag && late;
}

const RECIPES: Record<string, string> = { pca8: "PCA-8", native: "raw angles" };

function Footer({
  result,
  card,
  meta,
}: {
  result: BenchmarkResult | null;
  card: DatasetCard | null;
  meta: Meta | null;
}) {
  const items: { text: string; title?: string }[] = [];
  if (result) {
    const seeds = result.protocol?.seeds?.length ?? 0;
    if (seeds) items.push({ text: `${seeds} seeds`, title: seedsTitle(seeds) });
    const recipe = result.protocol?.recipe ?? card?.recipe;
    if (recipe) items.push({ text: RECIPES[recipe] ?? recipe, title: RECIPE_TITLES[recipe] });
    const configs = Object.values(result.protocol?.configs_per_model ?? {});
    if (configs.length) {
      const n = Math.max(...configs);
      items.push({ text: `${n} configs per model`, title: configsTitle(n) });
    }
    const licence = result.dataset?.licence ?? card?.licence;
    if (licence) items.push({ text: licence, title: "data licence" });
    items.push(
      result.source === "run"
        ? { text: "this session", title: "numbers from a run in this session" }
        : { text: "cached", title: "numbers from the results shipped with the library" },
    );
  }
  return (
    <footer
      className="flex h-8 shrink-0 items-end justify-between text-[12px] leading-4 text-muted-foreground"
      data-testid="footer"
    >
      <p>
        {items.map((item, i) => (
          <span key={item.text}>
            {i ? " · " : ""}
            <span title={item.title} aria-label={item.title}>
              {item.text}
            </span>
          </span>
        ))}
      </p>
      <p>{meta ? `v${meta.qlassifier}` : ""}</p>
    </footer>
  );
}

export function Dashboard() {
  const c = useDashboard();
  const { state } = c;
  const reduced = useReducedMotion();
  const result = activeResult(state);
  const noise = activeNoise(state);
  const card = datasetCard(state, state.view.dataset);
  const key = metricKey(state.view.metric);
  const focus = focusModel(state.view);
  const run = state.run;
  const running = isRunActive(run);
  // Skeleton only when the first paint takes longer than 300 ms (TDD §8.4).
  const slow = useDelayed(!result && !c.settled, 300);

  // Cached result of this dataset (names for empty live rows, steady axis); any cached result
  // still names the models when this dataset has none.
  const reference = state.shipped[state.view.dataset] ?? Object.values(state.shipped)[0] ?? null;
  const leaderboard = useMemo(
    () => (result ? buildLeaderboard(result, key, card, reference) : null),
    [result, key, card, reference],
  );
  const chips = useMemo(
    () => (result ? buildChips(result, key, noise, reference) : []),
    [result, key, noise, reference],
  );
  // A live run gets its cost task late: show the cached cost meanwhile (deterministic, same protocol).
  const cost = result?.cost ?? reference?.cost ?? null;
  // Names for the charts: a live partial lists only the models with a finished seed, the cached
  // result names the rest (cost and noise studies cover every quantum model from the start).
  const models = useMemo(() => {
    const live = result?.models ?? [];
    const seen = new Set(live.map((m) => m.id));
    return [...live, ...(reference?.models ?? []).filter((m) => !seen.has(m.id))];
  }, [result, reference]);
  // Provenance (hover detail only): a session or live result shown with the cached studies.
  const noiseProvenance =
    result &&
    noise &&
    result.source === "run" &&
    (!result.noise || result.noise.source === "shipped")
      ? { createdAt: noise.created_at ?? null }
      : null;
  const costProvenance =
    result && !result.cost && cost ? { createdAt: reference?.created_at ?? null } : null;

  const { clearFocus } = c;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") clearFocus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearFocus]);

  return (
    <TooltipProvider delayDuration={150}>
      <main
        className="mx-auto flex h-dvh min-h-[760px] max-w-[1680px] flex-col px-10 py-10"
        data-testid="dashboard"
        data-reduced-motion={reduced}
        data-ready={c.ready}
      >
        <TopBar
          tiles={
            state.datasets.length ? (
              <DatasetTiles
                datasets={state.datasets}
                value={state.view.dataset}
                onChange={c.setDataset}
                running={
                  running && run
                    ? { dataset: run.dataset, progress: run.total ? run.done / run.total : 0 }
                    : null
                }
              />
            ) : null
          }
          metric={<MetricSwitch value={state.view.metric} onChange={c.setMetric} />}
          actions={<ThemeToggle />}
        />
        {result ? <ProofChips chips={chips} /> : <div className="h-[112px] shrink-0" />}
        <div className={"grid min-h-0 flex-1 grid-cols-12 gap-6" + (slow ? " animate-pulse" : "")}>
          <div className="col-span-7 min-h-0">
            {leaderboard ? (
              <Leaderboard
                model={leaderboard}
                metric={state.view.metric}
                dataset={state.view.dataset}
                status={result?.status ?? ""}
                focus={focus}
                pinned={state.view.pinned}
                reduced={reduced}
                onHover={c.hover}
                onTogglePin={c.togglePin}
                onClear={c.clearFocus}
              />
            ) : (
              <div className="card h-full" />
            )}
          </div>
          <div className="col-span-5 flex min-h-0 flex-col gap-6">
            {result ? (
              <>
                <NoiseChart
                  noise={noise}
                  models={models}
                  focus={focus}
                  reduced={reduced}
                  onHover={c.hover}
                  onTogglePin={c.togglePin}
                  provenance={noiseProvenance}
                />
                <CostChart
                  cost={cost}
                  models={models}
                  focus={focus}
                  reduced={reduced}
                  onHover={c.hover}
                  onTogglePin={c.togglePin}
                  provenance={costProvenance}
                />
              </>
            ) : (
              <>
                <div className="card min-h-0 flex-1" />
                <div className="card min-h-0 flex-1" />
              </>
            )}
          </div>
        </div>
        <div className="h-6 shrink-0" />
        <Footer result={result} card={card} meta={state.meta} />
        <div className="sr-only" role="status" aria-live="polite" data-testid="announcer">
          {c.announcement}
        </div>
      </main>
      <Toaster />
    </TooltipProvider>
  );
}
