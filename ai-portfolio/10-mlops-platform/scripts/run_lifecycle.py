"""End-to-end MLOps lifecycle simulation.

Runs the full closed loop and logs a timeline that the screenshots visualise:

    train v1 -> promote to production -> stream shifting data -> drift detected
    -> auto-retrain v2 -> A/B canary -> gate passes -> promote v2 -> archive v1

Artifacts written to ``data/``:
    monitoring.csv          per-window PSI + accuracy series
    timeline.json           ordered lifecycle events with timestamps
    registry_snapshot.json  final registry table + lineage
    canary.json             the A/B canary comparison
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import CONFIG, seed_everything
from mlops.data import data_hash, drift_level_at, make_data, stream_batches
from mlops.drift import DriftMonitor, PerformanceMonitor
from mlops.gate import PromotionGate
from mlops.model import batch_accuracy, evaluate, load_model, save_model, train_model
from mlops.orchestrator import RetrainingOrchestrator
from mlops.registry import ModelRegistry, Stage
from mlops.tracking import ExperimentTracker

# --- stream / drift configuration ----------------------------------------
N_WINDOWS = 40
BATCH = 1500
DRIFT_START = 8      # stream is stationary before this window
DRIFT_PLATEAU = 12   # drift ramps to its maximum here, then holds (new steady state)
MAX_COVARIATE = 1.5
MAX_CONCEPT = 1.0
RECENT_BUFFER = 5    # windows of recent data used to retrain the challenger


def _fresh_db(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> dict:
    seed_everything(CONFIG.seed)
    CONFIG.ensure_dirs()
    _fresh_db(CONFIG.registry_db)
    tracker_db = CONFIG.data_dir / "experiments.duckdb"
    _fresh_db(tracker_db)

    registry = ModelRegistry(CONFIG.registry_db)
    tracker = ExperimentTracker(tracker_db)
    gate = PromotionGate(min_auc=CONFIG.min_auc, promotion_margin=CONFIG.promotion_margin)
    orch = RetrainingOrchestrator(registry, tracker, gate, CONFIG)

    timeline: list[dict] = []

    def event(kind: str, msg: str, **extra) -> None:
        timeline.append({"t": extra.pop("t", None), "kind": kind, "msg": msg, "ts": time.time(), **extra})

    # === 1. Train v1 on the reference (pre-drift) distribution =============
    Xtr, ytr = make_data(20_000, seed=CONFIG.seed, covariate_drift=0.0, concept_drift=0.0)
    Xho, yho = make_data(6_000, seed=CONFIG.seed + 7, covariate_drift=0.0, concept_drift=0.0)
    ref_X, _ = make_data(8_000, seed=CONFIG.seed + 3, covariate_drift=0.0, concept_drift=0.0)

    params_v1 = {"C": 1.0, "max_iter": 500}
    m1 = train_model(Xtr, ytr, params_v1, seed=CONFIG.seed)
    metrics_v1 = evaluate(m1, Xho, yho)
    art1 = str(CONFIG.artifacts_dir / "v1.pkl")
    save_model(m1, Path(art1))
    v1 = registry.register(
        name="churn_classifier",
        algo="LogisticRegression",
        params=params_v1,
        metrics=metrics_v1,
        data_hash=data_hash(Xtr, ytr),
        artifact_path=art1,
        parent_version=None,
        stage=Stage.STAGING,
    )
    tracker.log_run("baseline", params_v1, metrics_v1, art1, version=v1, tags={"stage": "initial"})
    registry.promote(v1, reason="initial production deployment")
    event("train", f"Trained v{v1} (AUC={metrics_v1['auc']:.3f}, acc={metrics_v1['accuracy']:.3f})", t=0, version=v1)
    event("promote", f"v{v1} -> PRODUCTION (initial deploy)", t=0, version=v1)

    # === 2. Set up monitors on the live-scoring stream ====================
    baseline_stream_acc = batch_accuracy(m1, Xho, yho)
    drift_mon = DriftMonitor(ref_X, CONFIG.features, bins=CONFIG.psi_bins, psi_threshold=CONFIG.psi_alert)
    perf_mon = PerformanceMonitor(baseline_stream_acc, rel_drop_threshold=CONFIG.perf_drop_alert, window=3)

    recent_X: list[np.ndarray] = []
    recent_y: list[np.ndarray] = []
    rows: list[dict] = []
    retrained = False
    v2: int | None = None
    prod_model = m1

    # === 3. Stream, monitor, and react ====================================
    for b in stream_batches(N_WINDOWS, BATCH, CONFIG.seed, DRIFT_START, DRIFT_PLATEAU, MAX_COVARIATE, MAX_CONCEPT):
        t = b.t
        recent_X.append(b.X)
        recent_y.append(b.y)
        if len(recent_X) > RECENT_BUFFER:
            recent_X.pop(0)
            recent_y.pop(0)

        # Score with whatever model is currently in production.
        prod = registry.production_version()
        prod_model = load_model(Path(prod.artifact_path))
        acc_prod = batch_accuracy(prod_model, b.X, b.y)

        # Shadow-evaluate v1 and (if it exists) v2 on this window for the
        # version-comparison chart — a truthful record of both models' behaviour.
        acc_v1 = batch_accuracy(m1, b.X, b.y)
        acc_v2 = batch_accuracy(load_model(Path(registry.get(v2).artifact_path)), b.X, b.y) if v2 else np.nan

        per_feat, data_alert = drift_mon.check(b.X, t)
        max_psi = max(per_feat.values())
        rolling_acc, perf_alert = perf_mon.update(acc_prod, t)

        decision = orch.observe(data_alert, perf_alert)

        row = {
            "t": t,
            "drift_level": round(b.drift, 4),
            "max_psi": round(max_psi, 4),
            "acc_prod": round(acc_prod, 4),
            "acc_v1": round(acc_v1, 4),
            "acc_v2": round(acc_v2, 4) if v2 else np.nan,
            "rolling_acc": round(rolling_acc, 4),
            "prod_version": prod.version,
            "data_alert": bool(data_alert),
            "perf_alert": bool(perf_alert),
            "trigger": bool(decision.triggered and not retrained),
            **{f"psi_{k}": round(v, 4) for k, v in per_feat.items()},
        }
        rows.append(row)

        if data_alert:
            event("alert", f"DATA DRIFT: {data_alert.detail}", t=t, value=data_alert.value)
        if perf_alert:
            event("alert", f"PERF DRIFT: {perf_alert.detail}", t=t, value=perf_alert.value)

        # === 4. Trigger auto-retrain (once) ================================
        if decision.triggered and not retrained:
            retrained = True
            event("trigger", f"RETRAIN TRIGGERED: {decision.reason}", t=t)

            # Retrain on the latest labelled production data from the *current*
            # regime (a fresh draw at the current drift level stands in for the
            # last N logged, labelled rows), and gate on an independent holdout
            # from the same regime — the distribution the model will serve.
            cur_cov = drift_level_at(t, DRIFT_START, DRIFT_PLATEAU, MAX_COVARIATE)
            cur_con = drift_level_at(t, DRIFT_START, DRIFT_PLATEAU, MAX_CONCEPT)
            Xrb, yrb = make_data(20_000, seed=CONFIG.seed + 500,
                                 covariate_drift=cur_cov, concept_drift=cur_con)
            evalX, evaly = make_data(6_000, seed=CONFIG.seed + 99,
                                     covariate_drift=cur_cov, concept_drift=cur_con)

            v2 = orch.retrain(
                name="churn_classifier",
                X=Xrb, y=yrb,
                params={"C": 1.0, "max_iter": 700},
                parent_version=prod.version,
                eval_X=evalX, eval_y=evaly,
            )
            m2 = load_model(Path(registry.get(v2).artifact_path))
            event("train", f"Trained challenger v{v2} on recent drifted data (parent=v{prod.version})", t=t, version=v2)

            # === 5. Canary / shadow A-B on the fresh holdout ===============
            canary = orch.canary(prod_model, prod.version, m2, v2, evalX, evaly, metric="auc")
            event(
                "canary",
                f"A/B canary: v{prod.version} AUC={canary.incumbent_metrics['auc']:.3f} vs "
                f"v{v2} AUC={canary.challenger_metrics['auc']:.3f} (Δ={canary.delta:+.3f}) -> winner {canary.winner}",
                t=t,
            )

            # === 6. Gate + promote =========================================
            incumbent_metrics_on_current = evaluate(prod_model, evalX, evaly)
            gate_result, promoted = orch.gate_and_promote(
                challenger_version=v2,
                challenger_metrics=canary.challenger_metrics,
                incumbent_metrics=incumbent_metrics_on_current,
            )
            event("gate", f"Promotion gate: {gate_result.summary()} -> {'PASS' if gate_result.passed else 'FAIL'}", t=t)
            if promoted:
                event("promote", f"v{v2} -> PRODUCTION; v{prod.version} -> ARCHIVED", t=t, version=v2)
            else:
                event("hold", f"Gate FAILED: keeping v{prod.version}; alert raised", t=t)

            with open(CONFIG.data_dir / "canary.json", "w") as f:
                json.dump(
                    {
                        "incumbent_version": canary.incumbent_version,
                        "challenger_version": canary.challenger_version,
                        "incumbent_metrics": canary.incumbent_metrics,
                        "challenger_metrics": canary.challenger_metrics,
                        "delta_auc": round(canary.delta, 4),
                        "winner": canary.winner,
                        "gate_passed": gate_result.passed,
                        "gate_checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in gate_result.checks],
                    },
                    f,
                    indent=2,
                )

    # === 7. Persist everything ===========================================
    df = pd.DataFrame(rows)
    df.to_csv(CONFIG.data_dir / "monitoring.csv", index=False)

    with open(CONFIG.data_dir / "timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)

    snapshot = {
        "versions": [
            {
                "version": mv.version,
                "stage": mv.stage.value,
                "algo": mv.algo,
                "auc": mv.metrics.get("auc"),
                "accuracy": mv.metrics.get("accuracy"),
                "f1": mv.metrics.get("f1"),
                "parent": mv.parent_version,
                "data_hash": mv.data_hash,
                "created": mv.created_iso,
            }
            for mv in registry.list_versions()
        ],
        "production": registry.production_version().version if registry.production_version() else None,
        "transitions": registry.transition_log(),
        "lineage_of_prod": [mv.version for mv in registry.lineage(registry.production_version().version)],
    }
    with open(CONFIG.data_dir / "registry_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    # === 8. Console summary ==============================================
    prod = registry.production_version()
    v1_end = df["acc_v1"].iloc[-1]
    v1_start = df["acc_v1"].iloc[0]
    print("\n" + "=" * 68)
    print("MLOPS LIFECYCLE SIMULATION — SUMMARY")
    print("=" * 68)
    for e in timeline:
        tt = f"t={e['t']:>2}" if e["t"] is not None else "t= -"
        print(f"  [{tt}] {e['kind'].upper():9s} {e['msg']}")
    print("-" * 68)
    print(f"  Windows streamed        : {N_WINDOWS} x {BATCH} rows = {N_WINDOWS*BATCH:,} scored")
    print(f"  Peak PSI (worst feature): {df['max_psi'].max():.3f}  (alert @ {CONFIG.psi_alert})")
    print(f"  v1 window acc start->end: {v1_start:.3f} -> {v1_end:.3f}  (degraded {v1_start-v1_end:+.3f})")
    if v2:
        v2_end = df["acc_v2"].iloc[-1]
        print(f"  v2 window acc at end     : {v2_end:.3f}  (recovery {v2_end-v1_end:+.3f} over v1)")
    print(f"  Final production version : v{prod.version} ({prod.metrics.get('auc')} AUC)")
    print(f"  Lineage of production    : {' -> '.join('v'+str(x) for x in snapshot['lineage_of_prod'])}")
    print("=" * 68 + "\n")

    registry.close()
    tracker.close()
    return {"timeline": timeline, "snapshot": snapshot, "monitoring_rows": len(df)}


if __name__ == "__main__":
    main()
