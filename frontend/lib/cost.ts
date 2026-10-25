// Cost view model (TDD §8.3): quantum models only; circuits to train + predict (log scale,
// on-device tuning as a hatched extension), qubits, two-qubit gates per circuit (FakeTorino).
import { groupOf, type Group } from "./palette";
import type { CostStudy, ModelSummary } from "./types";

export const COST_DEVICE = "fake_torino";

export interface CostRow {
  id: string;
  name: string;
  group: Group;
  circuits: number;
  tuning: number;
  total: number;
  qubits: number;
  gates: number | null;
  depth: number | null;
  gatesByDevice: Record<string, number>;
  depthByDevice: Record<string, number>;
}

export interface CostModelView {
  rows: CostRow[];
  devices: string[];
  maxCircuits: number;
  minCircuits: number;
  maxQubits: number;
  maxGates: number;
}

export function buildCost(cost: CostStudy | null, models: ModelSummary[]): CostModelView {
  if (!cost)
    return { rows: [], devices: [], maxCircuits: 1, minCircuits: 1, maxQubits: 1, maxGates: 1 };
  const rows: CostRow[] = cost.models.map((m) => {
    const summary = models.find((s) => s.id === m.model);
    const tuning = m.tuning_circuits_on_device ?? 0;
    return {
      id: m.model,
      name: summary?.name ?? m.model,
      group: summary ? groupOf(summary) : m.model === "qlassifier" ? "ours" : "quantum",
      circuits: m.circuits_fit_predict,
      tuning,
      total: m.circuits_fit_predict + tuning,
      qubits: m.qubits,
      gates: m.two_qubit_gates?.[COST_DEVICE] ?? null,
      depth: m.depth?.[COST_DEVICE] ?? null,
      gatesByDevice: m.two_qubit_gates ?? {},
      depthByDevice: m.depth ?? {},
    };
  });
  rows.sort((a, b) => a.total - b.total || a.name.localeCompare(b.name));
  return {
    rows,
    devices: cost.devices,
    maxCircuits: Math.max(1, ...rows.map((r) => r.total)),
    minCircuits: Math.max(1, Math.min(...rows.map((r) => Math.max(1, r.circuits)))),
    maxQubits: Math.max(1, ...rows.map((r) => r.qubits)),
    maxGates: Math.max(1, ...rows.map((r) => r.gates ?? 0)),
  };
}
