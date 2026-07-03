"""A small, from-scratch trainable classifier in pure NumPy.

This is the **surrogate for a language model**: a 1-hidden-layer MLP
(``input -> ReLU(hidden) -> softmax(classes)``) over hashed text features. It is
deliberately tiny so the whole repo runs on a laptop with no GPU, yet it is a
*real* trainable network with a real gradient-descent loop — which is all we
need to faithfully demonstrate full fine-tuning, LoRA, and distillation.

The forward/backward pass is written explicitly so that the LoRA adapter
(``lora.py``) and the distillation student (``distill.py``) can reuse the exact
same math, differing only in *which* parameters receive gradients and *what*
loss is applied at the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import accuracy, cross_entropy, softmax
from .optim import Adam


def _init_params(in_dim: int, hidden: int, n_classes: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # He init for the ReLU layer, small init for the linear head.
    return {
        "W1": rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((hidden, n_classes)) * np.sqrt(1.0 / hidden),
        "b2": np.zeros(n_classes),
    }


def forward(params: dict[str, np.ndarray], X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return ``(logits, cache)``. ``cache`` holds activations for backprop."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    logits = a1 @ params["W2"] + params["b2"]
    return logits, {"X": X, "z1": z1, "a1": a1}


def backward(
    params: dict[str, np.ndarray], cache: dict, dlogits: np.ndarray, l2: float
) -> dict[str, np.ndarray]:
    """Backprop a given output gradient ``dlogits`` (n, C) to all params."""
    a1, z1, X = cache["a1"], cache["z1"], cache["X"]
    grads = {}
    grads["W2"] = a1.T @ dlogits + l2 * params["W2"]
    grads["b2"] = dlogits.sum(axis=0)
    da1 = dlogits @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    grads["W1"] = X.T @ dz1 + l2 * params["W1"]
    grads["b1"] = dz1.sum(axis=0)
    return grads


def ce_output_grad(logits: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Softmax cross-entropy: returns ``(loss, dlogits)`` averaged over the batch.

    ``idx`` (the batch's row indices into the full training set) is accepted for
    a uniform trainer signature but unused here; KD losses use it to fetch
    per-example soft targets.
    """
    n = logits.shape[0]
    p = softmax(logits)
    loss = cross_entropy(p, y)
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


@dataclass
class MLPClassifier:
    """A 1-hidden-layer softmax classifier trained with Adam."""

    in_dim: int
    hidden: int
    n_classes: int
    seed: int = 42
    l2: float = 1e-4
    params: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = _init_params(self.in_dim, self.hidden, self.n_classes, rng)

    # -- introspection --------------------------------------------------------
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    # -- inference ------------------------------------------------------------
    def logits(self, X: np.ndarray) -> np.ndarray:
        return forward(self.params, X)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(self.logits(X), y)

    # -- training -------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 120,
        lr: float = 1e-2,
        batch_size: int = 256,
        output_grad=ce_output_grad,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> "MLPClassifier":
        """Mini-batch Adam. ``output_grad(logits, y_batch, idx) -> (loss, dlogits)``.

        Passing a custom ``output_grad`` (see :mod:`distillkit.distill`) is how
        knowledge distillation plugs a soft-label loss into the same loop. If
        ``eval_set`` is given, per-epoch accuracy is recorded in ``history_``
        (used to draw learning curves).
        """
        rng = np.random.default_rng(self.seed + 1)
        n = X.shape[0]
        opt = Adam(self.params, lr=lr)
        self.history_: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                bi = order[s : s + batch_size]
                logits, cache = forward(self.params, X[bi])
                _, dlogits = output_grad(logits, y[bi], bi)
                grads = backward(self.params, cache, dlogits, self.l2)
                opt.step(self.params, grads)
            if eval_set is not None:
                self.history_.append(self.score(eval_set[0], eval_set[1]))
        return self
