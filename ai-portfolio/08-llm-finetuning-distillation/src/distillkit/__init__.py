"""LLM Fine-Tuning & Distillation Pipeline — a runnable, GPU-free surrogate.

This package demonstrates the *techniques* used to fine-tune and compress large
language models — full fine-tuning, **LoRA** low-rank adaptation, **knowledge
distillation**, and an automated **eval/promotion gate** — on a small, fully
trainable NumPy model over a synthetic instruction-tuning dataset.

The model is a surrogate (a 1-hidden-layer MLP over hashed text features), but
every technique is implemented for real, with real gradient descent, and maps
directly onto the LLM stack (QLoRA + HF Trainer + vLLM). See the README for the
honest "surrogate model, real techniques" framing.

Modules:
    features : hashing-trick text featurizer (the embedding surrogate).
    data     : streaming synthetic instruction dataset generator.
    model    : from-scratch trainable MLP classifier + training loop.
    lora     : LoRA low-rank adapter (freeze base, train B·A + head, merge).
    distill  : teacher ensemble + temperature knowledge-distillation student.
    gate     : automated promotion gate (accuracy floor + regression checks).
    metrics  : stable softmax / cross-entropy / accuracy / KL.
    optim    : minimal Adam optimizer.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

from . import data, distill, features, gate, lora, metrics, model, optim  # noqa: E402
from .distill import DistillResult, TeacherEnsemble, distill as run_distill  # noqa: E402
from .features import HashingVectorizer  # noqa: E402
from .gate import GateResult, evaluate_gate  # noqa: E402
from .lora import LoRAModel  # noqa: E402
from .model import MLPClassifier  # noqa: E402

__all__ = [
    "SEED",
    "data",
    "features",
    "model",
    "lora",
    "distill",
    "gate",
    "metrics",
    "optim",
    "HashingVectorizer",
    "MLPClassifier",
    "LoRAModel",
    "TeacherEnsemble",
    "DistillResult",
    "run_distill",
    "GateResult",
    "evaluate_gate",
]
