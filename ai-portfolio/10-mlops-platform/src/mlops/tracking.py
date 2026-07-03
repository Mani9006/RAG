"""Experiment tracking backed by DuckDB.

Logs training/eval runs with their params, metrics, tags and artifact path, and
supports querying and comparing runs — the minimal MLflow-style surface a
platform needs. Runs are linked to registry versions so an experiment can be
traced to the model it produced.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class ExperimentTracker:
    """DuckDB-backed run log with query/compare helpers."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id        VARCHAR PRIMARY KEY,
                experiment    VARCHAR,
                version       INTEGER,
                params        VARCHAR,
                metrics       VARCHAR,
                tags          VARCHAR,
                artifact_path VARCHAR,
                ts            DOUBLE
            );
            """
        )

    def log_run(
        self,
        experiment: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifact_path: str = "",
        version: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
            [
                run_id,
                experiment,
                version,
                json.dumps(params),
                json.dumps(metrics),
                json.dumps(tags or {}),
                artifact_path,
                time.time(),
            ],
        )
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.con.execute(
            "SELECT run_id,experiment,version,params,metrics,tags,artifact_path,ts FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row_to_dict(row)

    def query(self, experiment: str | None = None) -> pd.DataFrame:
        """Return all runs (optionally filtered) as a flat, comparable DataFrame."""
        if experiment:
            rows = self.con.execute(
                "SELECT run_id,experiment,version,params,metrics,tags,artifact_path,ts "
                "FROM runs WHERE experiment = ? ORDER BY ts",
                [experiment],
            ).fetchall()
        else:
            rows = self.con.execute(
                "SELECT run_id,experiment,version,params,metrics,tags,artifact_path,ts FROM runs ORDER BY ts"
            ).fetchall()
        records = []
        for r in rows:
            d = self._row_to_dict(r)
            flat = {"run_id": d["run_id"], "experiment": d["experiment"], "version": d["version"]}
            for k, v in d["metrics"].items():
                flat[f"metric.{k}"] = v
            for k, v in d["params"].items():
                flat[f"param.{k}"] = v
            records.append(flat)
        return pd.DataFrame.from_records(records)

    def compare(self, run_ids: list[str], metric: str) -> pd.DataFrame:
        """Compare a set of runs on a single metric, best first."""
        df = self.query()
        col = f"metric.{metric}"
        sub = df[df["run_id"].isin(run_ids)]
        if col in sub.columns:
            sub = sub.sort_values(col, ascending=False)
        return sub.reset_index(drop=True)

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        return {
            "run_id": row[0],
            "experiment": row[1],
            "version": int(row[2]) if row[2] is not None else None,
            "params": json.loads(row[3]),
            "metrics": json.loads(row[4]),
            "tags": json.loads(row[5]),
            "artifact_path": row[6],
            "ts": float(row[7]),
        }

    def close(self) -> None:
        self.con.close()
