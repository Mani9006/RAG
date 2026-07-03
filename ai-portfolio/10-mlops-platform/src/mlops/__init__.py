"""MLOps Platform: registry, experiment tracking, drift monitoring,
automated retraining, and CI/CD promotion gates.

A single, offline, deterministic library that implements the full model
lifecycle used by a production ML platform:

    train -> register -> promote -> monitor -> drift -> retrain
          -> A/B canary -> gate -> promote/rollback -> archive

All components are real and importable. See ``scripts/run_lifecycle.py`` for
the end-to-end simulation and ``tests/`` for behavioural assertions.
"""
from __future__ import annotations

from .config import CONFIG, Config
from .registry import ModelRegistry, Stage, InvalidTransition
from .tracking import ExperimentTracker
from .drift import DriftMonitor, PerformanceMonitor, psi, kl_divergence
from .gate import PromotionGate, GateResult, GateCheck
from .orchestrator import RetrainingOrchestrator, CanaryResult

__all__ = [
    "CONFIG",
    "Config",
    "ModelRegistry",
    "Stage",
    "InvalidTransition",
    "ExperimentTracker",
    "DriftMonitor",
    "PerformanceMonitor",
    "psi",
    "kl_divergence",
    "PromotionGate",
    "GateResult",
    "GateCheck",
    "RetrainingOrchestrator",
    "CanaryResult",
]

__version__ = "1.0.0"
