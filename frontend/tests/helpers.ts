// Small synthetic results for mechanism tests (TDD §9.1: synthetic data only for mechanisms).
// Real-data checks use the shipped results in tests/fixtures/.
import type { BenchmarkResult, ModelSummary, RunRecord } from "@/lib/types";

type Family = ModelSummary["family"];
type Role = ModelSummary["role"];

export interface SpecRow {
  id: string;
  name?: string;
  family?: Family;
  role?: Role;
  values: number[];
  tied?: boolean;
  ci?: [number, number] | null;
}

const ROLE: Record<string, [Family, Role]> = {
  trivial: ["trivial", "trivial"],
  qlassifier: ["quantum", "ours"],
  zz_qsvc: ["quantum", "standard_qml"],
  pqk: ["quantum", "published_quantum"],
  separable: ["quantum", "published_quantum"],
  classical_twin: ["control", "twin"],
};

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function metric(values: number[], tied = false, ci: [number, number] | null = null) {
  const m = mean(values);
  const sd =
    values.length > 1
      ? Math.sqrt(values.reduce((a, v) => a + (v - m) ** 2, 0) / (values.length - 1))
      : null;
  return {
    mean: m,
    std: sd,
    ci_low: ci ? ci[0] : sd == null ? null : m - 2 * sd,
    ci_high: ci ? ci[1] : sd == null ? null : m + 2 * sd,
    n: values.length,
    tied_with_best: tied,
  };
}

export function makeResult(rows: SpecRow[], extra: Partial<BenchmarkResult> = {}): BenchmarkResult {
  const models: ModelSummary[] = rows.map((r, i) => {
    const [family, role] = ROLE[r.id] ?? ["classical", "classical"];
    const s = metric(r.values, r.tied ?? false, r.ci ?? null);
    return {
      id: r.id,
      name: r.name ?? r.id,
      family: r.family ?? family,
      role: r.role ?? role,
      metrics: { balanced_accuracy: s, accuracy: s, roc_auc: s, f1: s },
      time_s: { tune: 1, fit: 0.5, predict: 0.25 },
      n_seeds_done: r.values.length,
      rank: i + 1,
      edge_share: 0,
    };
  });
  const runs: RunRecord[] = rows.flatMap((r) =>
    r.values.map((v, seed) => ({
      seed,
      model: r.id,
      config: {},
      config_index: 0,
      cv_score: v,
      edge: false,
      balanced_accuracy: v,
      accuracy: v,
      roc_auc: v,
      f1: v,
      t_tune: 1,
      t_fit: 0.5,
      t_predict: 0.25,
      n_train: 70,
      n_test: 30,
    })),
  );
  return {
    schema_version: "1.0",
    status: "complete",
    source: "shipped",
    created_at: "2026-10-24T00:00:00Z",
    versions: {
      qlassifier: "0.1.0",
      python: "3.12.0",
      numpy: "2.5.4",
      scipy: "1.18.1",
      scikit_learn: "1.9.1",
      pandas: "3.1.0",
      joblib: "1.6.0",
      qiskit: "2.6.1",
      qiskit_aer: "0.18.0",
      qiskit_ibm_runtime: "0.51.0",
    },
    environment: { platform: "test", machine: "test", cpu_count: 1, n_jobs: 1, wall_s: 1 },
    dataset: {
      id: "toy",
      name: "Toy",
      n_samples: 100,
      n_features: 8,
      class_counts: { "0": 60, "1": 40 },
      positive_class: "one",
      source: "test",
      licence: "CC0 1.0",
      citation: "test",
      recipe: "pca8",
      grouped: false,
      sha256: null,
    },
    protocol: {
      fingerprint: "test",
      seeds: [0, 1, 2],
      test_size: 0.3,
      split: "stratified",
      test_train_ratio: 0.43,
      cv_folds: 5,
      recipe: "pca8",
      n_features_used: 8,
      configs_per_model: Object.fromEntries(rows.map((r) => [r.id, 30])),
      selection_metric: "balanced_accuracy",
      ci_level: 0.95,
      statistics: "nadeau_bengio_corrected_t",
      models: rows.map((r) => r.id),
    },
    models,
    comparisons: [],
    highlights: {
      best_classical: null,
      vs_standard_qml: null,
      vs_best_classical: null,
      published: null,
      vs_published_pp: null,
      noise_drop_max_pp: null,
      circuit_ratio_vs_fqk_pipeline: null,
    },
    runs,
    noise: null,
    cost: null,
    ...extra,
  };
}

/** A ReadableStream body that yields the given text chunks. */
export function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
}
