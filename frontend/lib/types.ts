// API types used by the dashboard: generated from the contract (TDD §6.7), never hand-written.
// Regenerate with `npm run gen:types`; `npm run check:types` fails CI when they drift.
import type { ModelMetrics, ModelSummary, Run } from "./api-types.gen";

export type {
  BenchmarkResult,
  Comparison,
  CostModel,
  CostStudy,
  DatasetCard,
  DatasetInfo,
  ErrorBody as ApiErrorBody,
  ErrorResponse,
  FailedData,
  Glyph,
  Highlights,
  Meta,
  MetricSummary,
  ModelSummary,
  NoiseStudy,
  NoiseSummary as NoiseSummaryRow,
  PairedDiff as PairStat,
  PartialData,
  ProgressData,
  Protocol,
  Published,
  Run,
  RunRecord,
  TerminalData,
} from "./api-types.gen";

export type MetricKey = keyof ModelMetrics;
export type RunStatus = Run["status"];
export type Family = ModelSummary["family"];
export type Role = ModelSummary["role"];
