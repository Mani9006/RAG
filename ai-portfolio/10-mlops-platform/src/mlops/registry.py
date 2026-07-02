"""Versioned model registry backed by DuckDB.

Stores every model version with its metadata, metrics and lineage (data hash,
params, parent version), tracks a lifecycle stage, and enforces a valid stage
state-machine. Promotion and rollback are first-class, audited operations.

Stage machine
-------------
    staging   -> production | archived
    production-> archived
    archived  -> staging | production      (restore / rollback)

Exactly one version may be in ``production`` at a time; ``promote`` enforces this
by archiving the incumbent atomically.
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


class Stage(str, enum.Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# Allowed stage transitions.
_ALLOWED: dict[Stage, set[Stage]] = {
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED},
    Stage.PRODUCTION: {Stage.ARCHIVED},
    Stage.ARCHIVED: {Stage.STAGING, Stage.PRODUCTION},
}


class InvalidTransition(Exception):
    """Raised when an illegal stage transition is attempted."""


@dataclass
class ModelVersion:
    version: int
    name: str
    algo: str
    stage: Stage
    params: dict[str, Any]
    metrics: dict[str, float]
    data_hash: str
    parent_version: int | None
    artifact_path: str
    created_at: float

    @property
    def created_iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))


class ModelRegistry:
    """DuckDB-backed model registry with a validated stage state-machine."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    # -- schema -------------------------------------------------------------
    def _init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                version       INTEGER PRIMARY KEY,
                name          VARCHAR,
                algo          VARCHAR,
                stage         VARCHAR,
                params        VARCHAR,
                metrics       VARCHAR,
                data_hash     VARCHAR,
                parent_version INTEGER,
                artifact_path VARCHAR,
                created_at    DOUBLE
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                id         BIGINT,
                version    INTEGER,
                from_stage VARCHAR,
                to_stage   VARCHAR,
                reason     VARCHAR,
                ts         DOUBLE
            );
            """
        )

    # -- registration -------------------------------------------------------
    def register(
        self,
        name: str,
        algo: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        data_hash: str,
        artifact_path: str,
        parent_version: int | None = None,
        stage: Stage = Stage.STAGING,
    ) -> int:
        """Register a new model version (defaults to ``staging``). Returns the version id."""
        version = self._next_version()
        now = time.time()
        self.con.execute(
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                version,
                name,
                algo,
                stage.value,
                json.dumps(params),
                json.dumps(metrics),
                data_hash,
                parent_version,
                artifact_path,
                now,
            ],
        )
        self._log_transition(version, None, stage, "register")
        return version

    def _next_version(self) -> int:
        row = self.con.execute("SELECT COALESCE(MAX(version), 0) FROM models").fetchone()
        return int(row[0]) + 1

    # -- reads --------------------------------------------------------------
    def get(self, version: int) -> ModelVersion:
        row = self.con.execute(
            "SELECT version,name,algo,stage,params,metrics,data_hash,parent_version,artifact_path,created_at "
            "FROM models WHERE version = ?",
            [version],
        ).fetchone()
        if row is None:
            raise KeyError(f"No such version: {version}")
        return self._row_to_version(row)

    def list_versions(self) -> list[ModelVersion]:
        rows = self.con.execute(
            "SELECT version,name,algo,stage,params,metrics,data_hash,parent_version,artifact_path,created_at "
            "FROM models ORDER BY version"
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def production_version(self) -> ModelVersion | None:
        row = self.con.execute(
            "SELECT version,name,algo,stage,params,metrics,data_hash,parent_version,artifact_path,created_at "
            "FROM models WHERE stage = 'production'"
        ).fetchone()
        return self._row_to_version(row) if row else None

    def lineage(self, version: int) -> list[ModelVersion]:
        """Return the ancestry chain [root, ..., version]."""
        chain: list[ModelVersion] = []
        cur: int | None = version
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            mv = self.get(cur)
            chain.append(mv)
            cur = mv.parent_version
        return list(reversed(chain))

    # -- stage transitions --------------------------------------------------
    def transition(self, version: int, to_stage: Stage, reason: str = "") -> None:
        """Move a version to ``to_stage`` if the transition is legal."""
        mv = self.get(version)
        if to_stage == mv.stage:
            return
        if to_stage not in _ALLOWED[mv.stage]:
            raise InvalidTransition(
                f"v{version}: {mv.stage.value} -> {to_stage.value} is not allowed"
            )
        self._set_stage(version, to_stage, mv.stage, reason)

    def promote(self, version: int, reason: str = "promotion") -> None:
        """Promote a version to production, archiving the current incumbent.

        This is the atomic 'ship it' operation. The candidate must currently be
        in staging or archived (a fresh model or a rolled-back one).
        """
        mv = self.get(version)
        if Stage.PRODUCTION not in _ALLOWED[mv.stage]:
            raise InvalidTransition(f"v{version}: cannot promote from {mv.stage.value}")
        incumbent = self.production_version()
        if incumbent is not None and incumbent.version != version:
            self._set_stage(incumbent.version, Stage.ARCHIVED, Stage.PRODUCTION,
                            f"superseded by v{version}")
        self._set_stage(version, Stage.PRODUCTION, mv.stage, reason)

    def rollback(self, reason: str = "rollback") -> int:
        """Roll production back to the previous production version.

        Uses the transition audit log to find the version that most recently
        held production before the current one, restores it, and archives the
        current production model. Returns the restored version id.
        """
        current = self.production_version()
        # Find prior production versions from the audit log (most recent first).
        rows = self.con.execute(
            "SELECT version, ts FROM transitions WHERE to_stage = 'production' ORDER BY ts DESC"
        ).fetchall()
        prior = [int(v) for v, _ in rows if current is None or int(v) != current.version]
        if not prior:
            raise InvalidTransition("No prior production version to roll back to")
        target = prior[0]
        if current is not None:
            self._set_stage(current.version, Stage.ARCHIVED, Stage.PRODUCTION,
                            f"rolled back in favour of v{target}")
        self._set_stage(target, Stage.PRODUCTION, self.get(target).stage, reason)
        return target

    # -- internals ----------------------------------------------------------
    def _set_stage(self, version: int, to_stage: Stage, from_stage: Stage, reason: str) -> None:
        self.con.execute("UPDATE models SET stage = ? WHERE version = ?", [to_stage.value, version])
        self._log_transition(version, from_stage, to_stage, reason)

    def _log_transition(self, version: int, from_stage: Stage | None, to_stage: Stage, reason: str) -> None:
        row = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM transitions").fetchone()
        next_id = int(row[0]) + 1
        self.con.execute(
            "INSERT INTO transitions VALUES (?,?,?,?,?,?)",
            [next_id, version, from_stage.value if from_stage else None, to_stage.value, reason, time.time()],
        )

    def transition_log(self) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT id,version,from_stage,to_stage,reason,ts FROM transitions ORDER BY id"
        ).fetchall()
        return [
            {"id": r[0], "version": r[1], "from_stage": r[2], "to_stage": r[3], "reason": r[4], "ts": r[5]}
            for r in rows
        ]

    @staticmethod
    def _row_to_version(row: tuple) -> ModelVersion:
        return ModelVersion(
            version=int(row[0]),
            name=row[1],
            algo=row[2],
            stage=Stage(row[3]),
            params=json.loads(row[4]),
            metrics=json.loads(row[5]),
            data_hash=row[6],
            parent_version=int(row[7]) if row[7] is not None else None,
            artifact_path=row[8],
            created_at=float(row[9]),
        )

    def close(self) -> None:
        self.con.close()
