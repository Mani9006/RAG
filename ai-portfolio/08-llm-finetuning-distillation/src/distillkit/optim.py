"""A tiny Adam optimizer over a dict of named parameter arrays.

Deterministic and dependency-free. Shared by the base trainer, the LoRA adapter,
and the distillation student so that every model in the repo optimizes the same
way — differences in results come from the *technique*, not the optimizer.
"""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Update ``params`` in place from ``grads`` (only keys present in grads)."""
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
