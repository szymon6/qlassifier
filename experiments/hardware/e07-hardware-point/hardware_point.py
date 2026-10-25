"""e07 hardware point (BT-3): Qlassifier Bloch readout on a real IBM Heron device, Breast Cancer 60 / 40 subset.

Up: README.md · Protocol: ../README.md §2.5 · Cost model: ../e03-cost-model/README.md · Noise twin: ../e02-noise-sweep/README.md

Modes
  --dry-run (default)  no token needed: plan (circuits x shots, QPU estimate), transpilation against the fake Heron
                       twin, and the pre-registered prediction from the three fake devices (20 sampler seeds each).
                       Writes plan.json, prediction.json, transpiled circuits (fake twin).
  --go                 submits ONE SamplerV2 job per seed (3 PUBs x 100 parameter sets x 1,024 shots = 300 circuits)
                       on the least-busy operational Heron device (or --backend NAME). Needs a saved account:
                       QiskitRuntimeService() defaults only; the token is never read, printed or stored by this script.
                       Refuses when the high cost estimate exceeds --budget-s (default 180 s) unless --allow-over-budget.
  --job-id ID          recover: fetch an existing job's result and finish the analysis (no new submission).

Design (fixed before any run): seed 0 (then 1 if the budget allows), the stage-1 noise-study subset
(``qlassifier.bench.noise.part_b_subset``), ``pca8`` fit on the 60 training rows, exact-mode tuning on those rows
(0 circuits), Bloch readout of all 100 rows on hardware. Two analyses from the same counts:
  variant 1 "inference only": exact training features, hardware test features (what the product's hardware mode does);
  variant 2 "train + test readout": hardware features for both (the PRD's optional training-feature readout).
Provenance written to the experiment folder: backend, UTC timestamps, job id, usage seconds, shots, circuit count,
physical qubits, calibration snapshot of those qubits, transpiled circuits (QASM 3 + QPY), parameter values, raw counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qlassifier import backends
from qlassifier.bench import noise as shipped
from qlassifier.bench.protocol import Prep, arrays
from qlassifier.encoding import Encoding
from qlassifier.tuning import gamma0, gaussian_gram

HERE = Path(__file__).resolve().parent
FloatArray = NDArray[np.float64]
DATASET = "breast_cancer"
SHOTS = 1024
SEED_TRANSPILER = 0
PREDICTION_SAMPLER_SEEDS = tuple(range(11, 31))  # 20 shot-noise draws per fake device
FAKES = ("fake_fez", "fake_torino", "fake_brisbane")
PASS_PP = 2.0  # PRD BT-3: hardware accuracy within the noisy-simulation prediction +- 2 pp
# e03 cost model (IBM job model): low = shots x (rep_delay + duration) per circuit + 2 s per job of <= 300 circuits;
# high = shots x 350 us per circuit + overhead.
REP_DELAY_S, QUICK_S, OVERHEAD_S, JOB_CIRCUITS = 250e-6, 350e-6, 2.0, 300


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- data, tuning, circuits
def prepare(seed: int) -> dict[str, Any]:
    """Subset rows, per-split preprocessing, exact tuning (0 circuits), encoding and whole-register circuits."""
    a = arrays(DATASET)
    Xr, yr, Xs, ys = shipped.part_b_subset(seed)
    p = Prep(a.recipe).fit(Xr)
    A, B = p.transform(Xr), p.transform(Xs)
    cfg = shipped._subset_config("qlassifier", seed, Xr, yr)
    enc = Encoding.plan(A.shape[1], None, None).fit(A)
    tmpl = backends.templates(enc)
    return dict(seed=seed, A=A, B=B, ya=yr, yb=ys, g=float(cfg["bw"]), C=float(cfg["C"]), enc=enc, tmpl=tmpl,
                circuits=backends.bloch_circuits(tmpl))  # fmt: skip


def transpile_pinned(circuits: dict[str, Any], backend: Any) -> tuple[dict[str, Any], list[int]]:
    """Shipped rule (ADR 0006): Z circuit at optimisation level 3, its initial layout pinned for X and Y."""
    from qiskit.transpiler import generate_preset_pass_manager

    tz = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=SEED_TRANSPILER).run(circuits["Z"])
    n = circuits["Z"].num_qubits
    layout = [int(v) for v in tz.layout.initial_index_layout(filter_ancillas=True)[:n]]
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout, seed_transpiler=SEED_TRANSPILER)
    return {s: (tz if s == "Z" else pm.run(circuits[s])) for s in backends.SETTINGS}, layout


def evaluate(prep: dict[str, Any], Fa: FloatArray, Fb: FloatArray) -> dict[str, float]:
    g0 = gamma0(Fa)
    acc, ba = shipped._svc_eval(gaussian_gram(Fa, Fa, prep["g"] * g0), prep["ya"], gaussian_gram(Fb, Fa, prep["g"] * g0), prep["yb"], prep["C"])
    return {"accuracy": acc, "balanced_accuracy": ba}


def analyses(prep: dict[str, Any], R: FloatArray) -> dict[str, Any]:
    """Exact, variant 1 (exact train + measured test) and variant 2 (measured train + test) from measured vectors R."""
    enc = prep["enc"]
    Fa, Fb = enc.bloch(prep["A"]), enc.bloch(prep["B"])
    Ra, Rb = R[: len(prep["A"])], R[len(prep["A"]) :]
    g0 = gamma0(Fa)
    out = {"exact": evaluate(prep, Fa, Fb), "variant1_inference": evaluate(prep, Fa, Rb), "variant2_train_test": evaluate(prep, Ra, Rb)}
    out["kernel_r_train"] = shipped._kernel_r(gaussian_gram(Ra, Ra, prep["g"] * gamma0(Ra)), gaussian_gram(Fa, Fa, prep["g"] * g0))
    Ke_te, Kv1_te = gaussian_gram(Fb, Fa, prep["g"] * g0), gaussian_gram(Rb, Fa, prep["g"] * g0)
    out["kernel_r_test_variant1"] = float(np.corrcoef(Ke_te.ravel(), Kv1_te.ravel())[0, 1])
    out["mean_abs_bloch_error"] = float(np.abs(R - np.vstack([Fa, Fb])).mean())
    return out


def cost(circuits: int, shots: int, duration_s: float) -> dict[str, float]:
    jobs = math.ceil(circuits / JOB_CIRCUITS)
    return {"qpu_low_s": circuits * shots * (REP_DELAY_S + duration_s) + OVERHEAD_S * jobs,
            "qpu_high_s": circuits * shots * QUICK_S + OVERHEAD_S * jobs}  # fmt: skip


# ----------------------------------------------------------------------------- dry run
def dry_run(seeds: list[int]) -> None:
    plan: dict[str, Any] = {"created_at": now(), "dataset": DATASET, "subset": "stage-1 noise-study part B (60 train / 40 test)",
                            "shots": SHOTS, "seeds": {}, "cost_model": "e03 (IBM job model, rep_delay 250 us, 2 s per job)"}  # fmt: skip
    prediction: dict[str, Any] = {"created_at": now(), "sampler_seeds": list(PREDICTION_SAMPLER_SEEDS), "pass_rule_pp": PASS_PP, "seeds": {}}
    for seed in seeds:
        prep = prepare(seed)
        twin = backends.device("fake_fez")
        transpiled, layout = transpile_pinned(prep["circuits"], twin)
        stats = [backends.circuit_stats(tc) for tc in transpiled.values()]
        dur = max(float(tc.estimate_duration(twin.target, unit="s")) for tc in transpiled.values())
        n_rows = len(prep["A"]) + len(prep["B"])
        n_circ = len(backends.SETTINGS) * n_rows
        plan["seeds"][str(seed)] = {
            "n_train": len(prep["A"]), "n_test": len(prep["B"]), "qubits": prep["enc"].n_qubits, "tuned": {"gamma": prep["g"], "C": prep["C"]},
            "circuits": n_circ, "pubs": len(backends.SETTINGS), "parameter_sets_per_pub": n_rows, "shots": SHOTS, "total_shots": n_circ * SHOTS,
            "variant1_inference_only_circuits": len(backends.SETTINGS) * len(prep["B"]),
            "twin": {"backend": "fake_fez", "layout": layout, "two_qubit_gates": max(s[0] for s in stats), "depth": max(s[1] for s in stats), "duration_us": 1e6 * dur},
            "cost": cost(n_circ, SHOTS, dur), "cost_variant1_only": cost(len(backends.SETTINGS) * len(prep["B"]), SHOTS, dur),
        }  # fmt: skip
        write_circuits(transpiled, prep, HERE / f"dry-run-seed{seed}")
        # pre-registered prediction: three fake Heron / Eagle devices, 20 shot-noise draws each, per-qubit path
        pred: dict[str, Any] = {}
        X = np.vstack([prep["A"], prep["B"]])
        for dev in FAKES:
            tdev, lay = transpile_pinned(prep["circuits"], backends.device(dev))
            assert backends.per_qubit_exact(tdev, dev)
            pq = backends.per_qubit_circuits(prep["tmpl"], dev, lay)
            rows = []
            for s in PREDICTION_SAMPLER_SEEDS:
                R = backends.measure_bloch_per_qubit(pq, prep["enc"].angles(X), dev, SHOTS, s, threads=1)
                rows.append(analyses(prep, R))
            pred[dev] = {"layout": lay, "n_draws": len(rows), "exact": rows[0]["exact"]}
            for v in ("variant1_inference", "variant2_train_test"):
                for m in ("accuracy", "balanced_accuracy"):
                    vals = np.array([r[v][m] for r in rows])
                    pred[dev][f"{v}_{m}"] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()), "max": float(vals.max()),
                                             "p2.5": float(np.percentile(vals, 2.5)), "p97.5": float(np.percentile(vals, 97.5))}  # fmt: skip
            pred[dev]["kernel_r_train_mean"] = float(np.mean([r["kernel_r_train"] for r in rows]))
            pred[dev]["kernel_r_test_variant1_mean"] = float(np.mean([r["kernel_r_test_variant1"] for r in rows]))
            pred[dev]["mean_abs_bloch_error"] = float(np.mean([r["mean_abs_bloch_error"] for r in rows]))
        prediction["seeds"][str(seed)] = pred
        p = plan["seeds"][str(seed)]
        print(f"PLAN seed {seed}: {p['circuits']} circuits ({p['pubs']} PUBs x {p['parameter_sets_per_pub']} parameter sets) x {SHOTS} shots"
              f" = {p['total_shots']:,} shots; {p['qubits']} qubits, {p['twin']['two_qubit_gates']} two-qubit gates, depth {p['twin']['depth']},"
              f" {p['twin']['duration_us']:.2f} us; QPU estimate {p['cost']['qpu_low_s']:.0f}-{p['cost']['qpu_high_s']:.0f} s"
              f" (inference-only readout: {p['variant1_inference_only_circuits']} circuits, {p['cost_variant1_only']['qpu_low_s']:.0f}-{p['cost_variant1_only']['qpu_high_s']:.0f} s)")
        for dev in FAKES:
            d = pred[dev]
            print(f"  PREDICTION {dev}: exact acc {d['exact']['accuracy']:.3f} BA {d['exact']['balanced_accuracy']:.3f};"
                  f" v1 acc {d['variant1_inference_accuracy']['mean']:.3f} ± {d['variant1_inference_accuracy']['sd']:.3f} [{d['variant1_inference_accuracy']['min']:.3f}, {d['variant1_inference_accuracy']['max']:.3f}]"
                  f" BA {d['variant1_inference_balanced_accuracy']['mean']:.3f}; v2 acc {d['variant2_train_test_accuracy']['mean']:.3f} ± {d['variant2_train_test_accuracy']['sd']:.3f}"
                  f" BA {d['variant2_train_test_balanced_accuracy']['mean']:.3f}; r_train {d['kernel_r_train_mean']:.4f}")
    (HERE / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    (HERE / "prediction.json").write_text(json.dumps(prediction, indent=1) + "\n")


def write_circuits(transpiled: dict[str, Any], prep: dict[str, Any], folder: Path) -> None:
    """Transpiled parameterised circuits as QASM 3 and QPY, and the parameter values (one row per data row)."""
    from qiskit import qasm3, qpy

    folder.mkdir(parents=True, exist_ok=True)
    X = np.vstack([prep["A"], prep["B"]])
    flat = prep["enc"].angles(X).reshape(len(X), -1)
    for s, tc in transpiled.items():
        (folder / f"circuit_{s}.qasm3").write_text(qasm3.dumps(tc))
        with (folder / f"circuit_{s}.qpy").open("wb") as f:
            qpy.dump(tc, f)
        np.savetxt(folder / f"parameters_{s}.csv", backends._values(tc, flat), delimiter=",", header=",".join(str(p) for p in tc.parameters), comments="")


# ----------------------------------------------------------------------------- hardware
def calibration_snapshot(backend: Any, qubits: list[int]) -> dict[str, Any]:
    t = backend.target
    snap: dict[str, Any] = {"captured_at": now(), "backend": backend.name, "qubits": {}}
    props = None
    try:
        props = backend.properties()
    except Exception:  # noqa: BLE001  (some backends have no BackendProperties)
        props = None
    for q in qubits:
        qp = t.qubit_properties[q] if t.qubit_properties else None
        entry: dict[str, Any] = {"t1_us": None if qp is None or qp.t1 is None else 1e6 * qp.t1, "t2_us": None if qp is None or qp.t2 is None else 1e6 * qp.t2,
                                 "frequency_ghz": None if qp is None or qp.frequency is None else qp.frequency / 1e9}  # fmt: skip
        for g in ("sx", "x", "rz", "measure", "reset"):
            if g in t.operation_names and t[g] is not None and (q,) in t[g] and t[g][(q,)] is not None:
                ip = t[g][(q,)]
                entry[g] = {"error": ip.error, "duration_ns": None if ip.duration is None else 1e9 * ip.duration}
        if props is not None:
            try:
                entry["readout_error"] = props.readout_error(q)
                entry["prob_meas0_prep1"] = props.qubit_property(q, "prob_meas0_prep1")[0]
                entry["prob_meas1_prep0"] = props.qubit_property(q, "prob_meas1_prep0")[0]
            except Exception:  # noqa: BLE001
                pass
        snap["qubits"][str(q)] = entry
    if props is not None:
        try:
            snap["properties_last_update_date"] = str(props.last_update_date)
        except Exception:  # noqa: BLE001
            pass
    return snap


PREFERRED = ("ibm_fez", "ibm_kingston")  # Heron r2; FakeFez is the twin of ibm_fez. Fallback when fez is down or queue > 50.
MAX_QUEUE = 50


def pick_backend(service: Any, name: str | None) -> Any:
    """--backend NAME, else the first preferred Heron device that is operational with a queue <= 50, else least busy Heron."""
    if name:
        return service.backend(name)
    for cand in PREFERRED:
        try:
            b = service.backend(cand)
            st = b.status()
            print(f"candidate {cand}: operational {st.operational}, pending jobs {st.pending_jobs}, status '{st.status_msg}'")
            if st.operational and st.pending_jobs <= MAX_QUEUE:
                return b
        except Exception as e:  # noqa: BLE001
            print(f"candidate {cand}: unavailable ({type(e).__name__})")

    def heron(b: Any) -> bool:
        try:
            fam = b.configuration().processor_type.get("family", "")
        except Exception:  # noqa: BLE001
            return False
        return fam == "Heron" and b.num_qubits >= 4

    return service.least_busy(operational=True, simulator=False, filters=heron)


def p1_from_result(result: Any, n_rows: int, n_qubits: int) -> tuple[FloatArray, list[list[dict[str, int]]]]:
    """Bloch vectors (N, n, 3) and raw counts per (setting, row) from a 3-PUB SamplerV2 result."""
    R = np.empty((n_rows, n_qubits, 3))
    counts_all: list[list[dict[str, int]]] = []
    for k, _s in enumerate(backends.SETTINGS):
        ba = result[k].data.meas
        counts_rows = []
        for r in range(n_rows):
            c = ba.get_counts(r)
            counts_rows.append(c)
            shots = sum(c.values())
            for q in range(n_qubits):
                p1 = sum(v for key, v in c.items() if key.replace(" ", "")[n_qubits - 1 - q] == "1") / shots
                R[r, q, k] = 1.0 - 2.0 * p1
        counts_all.append(counts_rows)
    return backends.renormalise(R), counts_all


def go(seeds: list[int], backend_name: str | None, budget_s: float, allow_over: bool, job_id: str | None) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    try:
        service = QiskitRuntimeService()  # saved account only; nothing else is read
    except Exception as e:  # noqa: BLE001
        print(f"No saved IBM Quantum account for QiskitRuntimeService(): {type(e).__name__}. Save one with "
              "QiskitRuntimeService.save_account(channel='ibm_quantum_platform', ...) and rerun. Nothing was submitted.")
        sys.exit(2)
    pred_all = json.loads((HERE / "prediction.json").read_text()) if (HERE / "prediction.json").exists() else None
    for seed in seeds:
        out = HERE / f"run-seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        prep = prepare(seed)
        X = np.vstack([prep["A"], prep["B"]])
        n_rows, n_q = len(X), prep["enc"].n_qubits
        if job_id:
            job = service.job(job_id)
            backend = service.backend(job.backend().name)
            print(f"recovering job {job_id} on {backend.name}")
        else:
            backend = pick_backend(service, backend_name)
        transpiled, layout = transpile_pinned(prep["circuits"], backend)
        stats = [backends.circuit_stats(tc) for tc in transpiled.values()]
        try:
            dur = max(float(tc.estimate_duration(backend.target, unit="s")) for tc in transpiled.values())
        except Exception as e:  # noqa: BLE001  (a target without durations)
            dur = 1.7e-6
            print(f"estimate_duration unavailable ({type(e).__name__}); using 1.7 us")
        n_circ = len(backends.SETTINGS) * n_rows
        est = cost(n_circ, SHOTS, dur)
        record: dict[str, Any] = {
            "seed": seed, "dataset": DATASET, "backend": backend.name, "processor_type": getattr(backend.configuration(), "processor_type", None),
            "n_train": len(prep["A"]), "n_test": len(prep["B"]), "qubits_logical": n_q, "physical_qubits": layout, "shots": SHOTS,
            "circuits": n_circ, "pubs": len(backends.SETTINGS), "parameter_sets_per_pub": n_rows, "two_qubit_gates": max(s[0] for s in stats),
            "depth": max(s[1] for s in stats), "duration_us": 1e6 * dur, "cost_estimate": est, "tuned": {"gamma": prep["g"], "C": prep["C"]},
            "transpiler": {"optimization_level": 3, "seed_transpiler": SEED_TRANSPILER, "layout_rule": "Z circuit layout pinned for X and Y (ADR 0006)"},
            "sampler_options": {"default_shots": SHOTS, "dynamical_decoupling": False, "twirling_gates": False, "twirling_measure": False},
            "timestamps": {"prepared": now()},
        }  # fmt: skip
        write_circuits(transpiled, prep, out)
        record["calibration_before"] = calibration_snapshot(backend, layout)
        print(f"PLAN seed {seed} on {backend.name}: {n_circ} circuits x {SHOTS} shots, qubits {layout}, 2Q {record['two_qubit_gates']}, depth {record['depth']},"
              f" QPU estimate {est['qpu_low_s']:.0f}-{est['qpu_high_s']:.0f} s (budget {budget_s:.0f} s)")
        if not job_id:
            if est["qpu_high_s"] > budget_s and not allow_over:
                print("estimate above budget; not submitted (use --allow-over-budget to override)")
                (out / "record.json").write_text(json.dumps(record, indent=1) + "\n")
                sys.exit(3)
            sampler = SamplerV2(mode=backend)
            sampler.options.default_shots = SHOTS
            sampler.options.dynamical_decoupling.enable = False
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.enable_measure = False
            flat = prep["enc"].angles(X).reshape(n_rows, -1)
            pubs = [(transpiled[s], backends._values(transpiled[s], flat)) for s in backends.SETTINGS]
            record["timestamps"]["submitted"] = now()
            job = sampler.run(pubs, shots=SHOTS)
            record["job_id"] = job.job_id()
            (out / "record.json").write_text(json.dumps(record, indent=1) + "\n")  # job id saved before waiting
            print(f"submitted job {record['job_id']} at {record['timestamps']['submitted']}; waiting ...")
        else:
            record["job_id"] = job_id
        result = job.result()
        record["timestamps"]["result_received"] = now()
        try:
            record["usage_s"] = float(job.usage())
        except Exception as e:  # noqa: BLE001
            record["usage_s"] = f"n/a ({type(e).__name__})"
        try:
            record["metrics"] = job.metrics()
        except Exception as e:  # noqa: BLE001
            record["metrics"] = f"n/a ({type(e).__name__})"
        record["calibration_after"] = calibration_snapshot(backend, layout)
        R, counts = p1_from_result(result, n_rows, n_q)
        (out / "counts.json").write_text(json.dumps({"settings": list(backends.SETTINGS), "rows": n_rows, "counts": counts}) + "\n")
        np.save(out / "bloch_measured.npy", R)
        np.save(out / "bloch_exact.npy", prep["enc"].bloch(X))
        record["analysis"] = analyses(prep, R)
        if pred_all and str(seed) in pred_all["seeds"]:
            pred = pred_all["seeds"][str(seed)]
            fam = "fake_fez"
            verdict = {}
            for v in ("variant1_inference", "variant2_train_test"):
                hw = record["analysis"][v]["accuracy"]
                sim = pred[fam][f"{v}_accuracy"]["mean"]
                verdict[v] = {"hardware_accuracy": hw, "fake_fez_prediction": sim, "delta_pp": 100 * (hw - sim), "pass_2pp": abs(hw - sim) <= PASS_PP / 100,
                              "within_one_test_point": abs(hw - sim) <= 1.0 / len(prep["B"]) + 1e-12,
                              "inside_prediction_range": pred[fam][f"{v}_accuracy"]["min"] - 1e-12 <= hw <= pred[fam][f"{v}_accuracy"]["max"] + 1e-12}  # fmt: skip
            record["verdict_vs_prediction"] = verdict
        (out / "record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        print(json.dumps({k: record[k] for k in ("backend", "job_id", "usage_s", "analysis", "verdict_vs_prediction") if k in record}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="submit to real hardware (needs a saved IBM account; only after GO)")
    ap.add_argument("--dry-run", action="store_true", help="plan + prediction only (default)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--budget-s", type=float, default=180.0)
    ap.add_argument("--allow-over-budget", action="store_true")
    ap.add_argument("--job-id", default=None)
    a = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if a.go or a.job_id:
            go(a.seeds, a.backend, a.budget_s, a.allow_over_budget, a.job_id)
        else:
            dry_run(a.seeds)


if __name__ == "__main__":
    main()
