"""LoRA-style low-rank adaptation of the base model's first weight matrix.

Low-Rank Adaptation (Hu et al., 2021) freezes a pretrained weight matrix ``W``
and learns a low-rank update ``ΔW = (alpha / r) · B · A`` where ``B ∈ R^{d×r}``
and ``A ∈ R^{r×k}`` with ``r ≪ min(d, k)``. Only ``A`` and ``B`` (and, as is
common in practice, the small classifier head) are trained; the base is frozen.

Here we adapt ``W1`` (the largest matrix in the surrogate MLP), exactly as real
LoRA targets the big ``q/k/v/o`` projections of a transformer. The payoff is the
same: a large reduction in trainable parameters at near-parity accuracy, plus a
zero-cost ``merge`` back into the base weights for inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import accuracy, softmax
from .model import MLPClassifier, ce_output_grad
from .optim import Adam


@dataclass
class LoRAModel:
    """Wrap a *frozen* :class:`MLPClassifier` with a trainable low-rank adapter.

    Effective first layer: ``W1_eff = W1_base + (alpha / r) · B @ A``.
    Trainable tensors: ``A``, ``B`` and the classifier head (``W2``, ``b2``).
    Everything else (``W1_base``, ``b1``) stays frozen.
    """

    base: MLPClassifier
    r: int = 4
    alpha: float = 8.0
    seed: int = 0
    l2: float = 1e-4
    train_head: bool = True
    adapter: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        in_dim, hidden = self.base.params["W1"].shape
        rng = np.random.default_rng(self.seed)
        # Standard LoRA init: A ~ small Gaussian, B = 0  =>  ΔW = 0 at start,
        # so the adapted model begins identical to the frozen base.
        self.adapter = {
            "A": rng.standard_normal((self.r, hidden)) * 0.01,
            "B": np.zeros((in_dim, self.r)),
        }
        # Head starts from the base head (fine-tune it further).
        self.head = {"W2": self.base.params["W2"].copy(), "b2": self.base.params["b2"].copy()}
        self.scale = self.alpha / self.r

    # -- effective weights ----------------------------------------------------
    def w1_eff(self) -> np.ndarray:
        return self.base.params["W1"] + self.scale * (self.adapter["B"] @ self.adapter["A"])

    def _params_view(self) -> dict[str, np.ndarray]:
        return {"W1": self.w1_eff(), "b1": self.base.params["b1"], **self.head}

    # -- introspection --------------------------------------------------------
    def n_trainable(self) -> int:
        n = self.adapter["A"].size + self.adapter["B"].size
        if self.train_head:
            n += self.head["W2"].size + self.head["b2"].size
        return int(n)

    def n_base(self) -> int:
        return self.base.n_params()

    # -- inference ------------------------------------------------------------
    def logits(self, X: np.ndarray) -> np.ndarray:
        p = self._params_view()
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0.0)
        return a1 @ p["W2"] + p["b2"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

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
    ) -> "LoRAModel":
        rng = np.random.default_rng(self.seed + 1)
        n = X.shape[0]
        W1_base = self.base.params["W1"]
        b1 = self.base.params["b1"]

        train = {"A": self.adapter["A"], "B": self.adapter["B"]}
        if self.train_head:
            train["W2"] = self.head["W2"]
            train["b2"] = self.head["b2"]
        opt = Adam(train, lr=lr)

        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                bi = order[s : s + batch_size]
                Xb = X[bi]
                W1 = W1_base + self.scale * (self.adapter["B"] @ self.adapter["A"])
                z1 = Xb @ W1 + b1
                a1 = np.maximum(z1, 0.0)
                logits = a1 @ self.head["W2"] + self.head["b2"]
                _, dlogits = ce_output_grad(logits, y[bi])

                grads: dict[str, np.ndarray] = {}
                if self.train_head:
                    grads["W2"] = a1.T @ dlogits + self.l2 * self.head["W2"]
                    grads["b2"] = dlogits.sum(axis=0)
                da1 = dlogits @ self.head["W2"].T
                dz1 = da1 * (z1 > 0)
                dW1_eff = Xb.T @ dz1  # gradient wrt effective W1
                # Chain through ΔW = scale · B @ A (base is frozen).
                grads["B"] = self.scale * (dW1_eff @ self.adapter["A"].T) + self.l2 * self.adapter["B"]
                grads["A"] = self.scale * (self.adapter["B"].T @ dW1_eff) + self.l2 * self.adapter["A"]

                opt.step(train, grads)
        return self

    # -- merge ----------------------------------------------------------------
    def merge(self) -> MLPClassifier:
        """Fold the adapter into a standalone :class:`MLPClassifier`.

        The merged model has **no** adapter and no inference overhead, yet
        produces identical logits — this is what ships to production.
        """
        merged = MLPClassifier(
            in_dim=self.base.in_dim,
            hidden=self.base.hidden,
            n_classes=self.base.n_classes,
            seed=self.base.seed,
            l2=self.base.l2,
        )
        merged.params = {
            "W1": self.w1_eff(),
            "b1": self.base.params["b1"].copy(),
            "W2": self.head["W2"].copy(),
            "b2": self.head["b2"].copy(),
        }
        return merged
