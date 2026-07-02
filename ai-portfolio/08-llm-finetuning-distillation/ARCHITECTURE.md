# Architecture & Design Decisions

## The core idea: surrogate model, real techniques

Fine-tuning, LoRA, and distillation are **optimization/parameterization
techniques**. Their correctness and their trade-offs (trainable-parameter
reduction, dark-knowledge transfer, exact adapter merge, promotion gating) do
**not** depend on the model being a 70B transformer — they depend on there being
(a) a weight matrix to adapt, (b) a differentiable loss, and (c) a train/eval
loop. This repo provides all three with a small NumPy model so the *entire*
pipeline runs deterministically on a CPU in minutes, and every claim is backed by
a number the code actually produced.

```
instruction text ──▶ HashingVectorizer ──▶ x ∈ R^D
                                              │
                        ┌─────────────────────┴───────────────────────┐
                        ▼                                              ▼
                   W1 (D×H) ── ReLU ── W2 (H×C) ── softmax        (the "LLM" surrogate)
                        ▲
        LoRA:  W1_eff = W1 + (α/r)·B·A   (freeze W1, train B,A + head)
```

| Real LLM stack | This repo's surrogate |
|---|---|
| Token embeddings + attention | Hashing-trick features (`features.py`) |
| Transformer block weight `W ∈ R^{d×d}` | First linear layer `W1 ∈ R^{D×H}` |
| LoRA on q/k/v/o projections | LoRA on `W1` (`lora.py`) |
| Instruction-tuning corpus (FLAN/Alpaca) | Synthetic instruction generator (`data.py`) |
| Teacher = large model / ensemble | Bagged MLP ensemble (`distill.py`) |
| KD soft-label loss (Hinton et al.) | Same loss, same `T²` scaling |
| CI eval gate before deploy | `gate.py` |

## Component decisions

**Featurizer — hashing trick, not a learned vocab.** Stateless and deterministic:
train/val/test and any new text featurize identically with zero fitted state, so
reproducibility is trivial and there is no vocabulary drift. Signed hashing
(Weinberger 2009) halves collision bias. The techniques downstream are
feature-agnostic, so a simple featurizer is the right call.

**Model — 1-hidden-layer MLP with explicit forward/backward.** Written by hand so
the LoRA adapter and the KD student reuse the *exact* same math and differ only in
which parameters get gradients and what loss is applied at the logits. Adam is
shared across all trainers (`optim.py`) so results reflect the technique, not the
optimizer.

**Domain shift for the LoRA story.** `data.py` ships two **disjoint** synonym
vocabularies for the same 9 labels (`source`, `target`). The base model is
pretrained on `source`; it scores near-chance on `target` (different tokens hash
to different buckets). Full fine-tuning and LoRA then both adapt to `target` — and
LoRA matches full fine-tuning while training a fraction of the parameters. Without
a genuine shift, "adaptation" would be meaningless.

**LoRA init & merge.** `A ~ N(0, 0.01²)`, `B = 0`, so `ΔW = 0` at init and the
adapted model *starts identical* to the frozen base (standard LoRA). `merge()`
folds `W1 + (α/r)·B·A` into a plain `MLPClassifier` with **zero** inference
overhead; a test asserts merged logits equal adapter logits to `< 1e-9`.

**Distillation needs headroom.** A tiny model on this task saturates, leaving no
room for KD to help. We create honest, controlled headroom the way it appears in
practice: the **teacher trains on clean labels**, but the **student's transfer set
carries heavy label noise**. With a low hard-label weight (`α=0.2`) the KD student
follows the teacher's clean soft targets and is largely immune to the noise, while
the no-KD student overfits it. This is the well-documented "KD is robust to label
noise / dark knowledge regularizes small models" result — reproduced, not asserted.

**Eval gate.** Three composable checks — absolute accuracy floor, no-regression vs
an incumbent baseline, and worst-class recall floor — returning a structured,
serializable verdict suitable for CI. A deliberately under-trained model is
blocked (test-enforced).

## Determinism

Every RNG is seeded (`numpy.random.default_rng`), the hash is `PYTHONHASHSEED`-
independent (blake2b), and `MPLBACKEND=Agg`. Re-running `make run` reproduces the
same numbers.

## Scaling honestly

- **Data generator** is a chunked generator (`data.stream`) with `O(chunk)` memory;
  `benchmark_generator.py` streams 1e6+ rows aggregating a histogram without OOM
  and extrapolates to 1e9 (embarrassingly parallel across shards/seeds).
- **The model math is BLAS matmuls** — the same code scales to larger `D`, `H`,
  and more data; only wall-clock changes. The single-core OpenBLAS in this
  sandbox runs at ~1 GFLOP/s, so sizes are chosen for a few-minute end-to-end run.

## Scaling the *techniques* to real LLMs

The code is a 1:1 conceptual map onto the production stack:

- **LoRA / QLoRA** — swap `W1` for the transformer's `q/k/v/o` and MLP
  projections; keep base weights frozen (4-bit with QLoRA), train `A`/`B` per
  target module. Same init, same `α/r` scaling, same merge-for-inference. Use
  🤗 PEFT `LoraConfig` + `Trainer`.
- **Distillation** — teacher = a larger checkpoint or an ensemble; student = a
  smaller architecture; identical temperature KD loss on the logits. Precompute
  teacher logits over the transfer set to keep the student loop cheap.
- **Serving** — merge the adapter and serve the single dense model with vLLM /
  TGI for high-throughput, low-latency inference (no adapter overhead at runtime).
- **Eval gate** — wire `gate.py`'s verdict into CI (GitHub Actions) so a
  regression or an under-trained checkpoint blocks promotion automatically.

## Trade-offs & limitations

- The surrogate cannot demonstrate *emergent* LLM behavior — only the mechanics of
  the adaptation/compression techniques. That is stated plainly in the README.
- Hashing collisions are a (small, bounded) source of noise; increasing `D`
  reduces them at linear cost.
- KD headroom here is engineered via label noise; on real models it also comes
  from capacity gaps and genuine aleatoric uncertainty in the data.
