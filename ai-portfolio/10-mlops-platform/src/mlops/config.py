"""Central, deterministic configuration for the MLOps platform.

Everything that influences results (seeds, thresholds, data shape) lives here so
runs are reproducible and the operating envelope is documented in one place.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Force a headless, deterministic matplotlib backend before pyplot is imported.
os.environ.setdefault("MPLBACKEND", "Agg")

SEED = 42

# Feature names for the synthetic scoring problem (a churn/fraud-like classifier).
FEATURES: tuple[str, ...] = (
    "amount",
    "tenure",
    "n_txn",
    "latency",
    "risk_score",
    "engagement",
)


@dataclass(frozen=True)
class Config:
    """Immutable platform configuration."""

    seed: int = SEED
    features: tuple[str, ...] = FEATURES

    # --- Storage locations -------------------------------------------------
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = field(default=Path(__file__).resolve().parents[2] / "data")
    assets_dir: Path = field(default=Path(__file__).resolve().parents[2] / "assets")

    # --- Drift monitoring --------------------------------------------------
    psi_bins: int = 10
    # PSI rule-of-thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift.
    psi_alert: float = 0.25
    # Sustained relative accuracy drop (vs. registered baseline) that signals
    # concept/performance drift.
    perf_drop_alert: float = 0.07
    # A retrain fires only when drift persists for this many consecutive windows,
    # avoiding thrashing on transient spikes.
    sustained_windows: int = 3

    # --- Promotion gate ----------------------------------------------------
    # A challenger must beat the incumbent on the holdout by at least this
    # absolute AUC margin (no-regression rule).
    promotion_margin: float = 0.02
    # Minimum absolute quality bar a candidate must clear regardless of incumbent.
    min_auc: float = 0.70

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def registry_db(self) -> Path:
        return self.data_dir / "registry.duckdb"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.assets_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()


def seed_everything(seed: int = SEED) -> None:
    """Seed all sources of randomness used in the platform."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
