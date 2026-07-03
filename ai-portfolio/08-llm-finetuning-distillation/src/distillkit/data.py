"""Synthetic *instruction-tuning* dataset generator.

Each example is an ``(instruction + input) -> label`` pair, mirroring the shape
of a real instruction-tuning corpus (e.g. FLAN / Alpaca) but small and fully
controllable. Three task *families* share a single 9-way label head:

    sentiment : positive | negative
    topic     : sports | technology | finance | health
    intent    : greeting | complaint | question

The label is determined by task-specific *signal words*; distractor and neutral
words add realistic noise so accuracy tops out below 100 %. Two vocabularies are
provided:

* ``domain="source"`` — the "pretraining" distribution.
* ``domain="target"`` — a **disjoint** synonym vocabulary that simulates a
  domain shift. A model pretrained on ``source`` scores near-chance on
  ``target`` until it is *adapted* (full fine-tune or LoRA). This is what makes
  the LoRA demo meaningful.

The generator is *streaming* (``stream``) with bounded memory, so it scales to
arbitrarily many rows — see ``scripts/benchmark_generator.py``.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# --- Task / label definitions ------------------------------------------------

TASK_INSTRUCTIONS: dict[str, str] = {
    "sentiment": "Classify the sentiment of this review:",
    "topic": "What is the topic of this text:",
    "intent": "Identify the user intent in this message:",
}

# Global 9-way label space (stable order).
LABELS: list[str] = [
    "sentiment/positive",
    "sentiment/negative",
    "topic/sports",
    "topic/technology",
    "topic/finance",
    "topic/health",
    "intent/greeting",
    "intent/complaint",
    "intent/question",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
N_CLASSES = len(LABELS)

_TASK_OF = {name: name.split("/")[0] for name in LABELS}

# Signal vocabularies. SOURCE and TARGET are disjoint synonym sets for the same
# labels — the (deliberate) distribution shift used by the LoRA experiment.
SOURCE_VOCAB: dict[str, list[str]] = {
    "sentiment/positive": "excellent wonderful love great fantastic amazing delightful superb enjoyed pleased".split(),
    "sentiment/negative": "terrible awful hate horrible disappointing worst broken useless angry frustrated".split(),
    "topic/sports": "game team player score match championship coach league tournament goal".split(),
    "topic/technology": "software computer algorithm data chip network processor code server cloud".split(),
    "topic/finance": "stock market invest revenue profit earnings bank trading portfolio dividend".split(),
    "topic/health": "doctor patient medicine hospital treatment disease symptom therapy vaccine diet".split(),
    "intent/greeting": "hello hi hey welcome morning greetings howdy hiya salutations cheers".split(),
    "intent/complaint": "complaint refund defective unacceptable dissatisfied faulty poor damaged annoyed grievance".split(),
    "intent/question": "what when where why how which who wondering explain clarify".split(),
}
TARGET_VOCAB: dict[str, list[str]] = {
    "sentiment/positive": "stellar marvelous adore brilliant magnificent splendid gratifying phenomenal thrilled satisfied".split(),
    "sentiment/negative": "dreadful atrocious despise appalling underwhelming abysmal shattered worthless furious irritated".split(),
    "topic/sports": "athlete stadium referee striker pitch playoff roster fixture scrimmage sprint".split(),
    "topic/technology": "gpu firmware compiler dataset transistor router kernel binary datacenter latency".split(),
    "topic/finance": "equity bond yield merger valuation fund hedge liquidity collateral fiscal".split(),
    "topic/health": "clinician nurse antibiotic clinic diagnosis infection wellness surgery immunity nutrition".split(),
    "intent/greeting": "heya greeting welcoming afternoon evening yo namaste aloha bonjour hola".split(),
    "intent/complaint": "dispute chargeback malfunction intolerable aggrieved shoddy substandard defect vexed protest".split(),
    "intent/question": "whom whence whither inquire elaborate specify wondered puzzled querying ask".split(),
}

NEUTRAL_WORDS: list[str] = (
    "the a an and to of it is this that for with please item thing today really "
    "very just about here there my your our their been was were".split()
)


@dataclass
class Dataset:
    """A featurization-ready split."""

    texts: list[str]
    y: np.ndarray  # int labels, shape (n,)
    domain: str

    def __len__(self) -> int:
        return len(self.texts)


def _vocab_for(domain: str) -> dict[str, list[str]]:
    if domain == "source":
        return SOURCE_VOCAB
    if domain == "target":
        return TARGET_VOCAB
    raise ValueError(f"unknown domain {domain!r} (expected 'source' or 'target')")


def _make_example(
    rng: np.random.Generator,
    label_id: int,
    vocab: dict[str, list[str]],
    *,
    n_signal: int,
    n_noise: int,
    distractor_p: float,
) -> str:
    label = LABELS[label_id]
    task = _TASK_OF[label]
    signal_pool = vocab[label]
    words = list(rng.choice(signal_pool, size=n_signal, replace=True))

    # Distractor: a signal word from a *different* class in the same task family,
    # so the example is ambiguous but the majority signal still wins.
    if rng.random() < distractor_p:
        others = [i for i, name in enumerate(LABELS) if _TASK_OF[name] == task and i != label_id]
        other = int(rng.choice(others))
        words.append(str(rng.choice(vocab[LABELS[other]])))

    words += list(rng.choice(NEUTRAL_WORDS, size=n_noise, replace=True))
    rng.shuffle(words)
    return TASK_INSTRUCTIONS[task] + " " + " ".join(words)


def stream(
    n: int,
    *,
    seed: int = 42,
    domain: str = "source",
    chunk_size: int = 100_000,
    n_signal: int = 4,
    n_noise: int = 5,
    distractor_p: float = 0.18,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(texts, labels)`` chunks with bounded memory.

    Classes are drawn uniformly, so the label distribution is balanced in
    expectation regardless of ``n``. Memory is ``O(chunk_size)``.
    """
    rng = np.random.default_rng(seed)
    vocab = _vocab_for(domain)
    remaining = n
    while remaining > 0:
        m = min(chunk_size, remaining)
        labels = rng.integers(0, N_CLASSES, size=m)
        texts = [
            _make_example(
                rng, int(lbl), vocab,
                n_signal=n_signal, n_noise=n_noise, distractor_p=distractor_p,
            )
            for lbl in labels
        ]
        yield texts, labels.astype(np.int64)
        remaining -= m


def generate(
    n: int,
    *,
    seed: int = 42,
    domain: str = "source",
    **kwargs,
) -> Dataset:
    """Materialize ``n`` examples into a single :class:`Dataset`."""
    texts: list[str] = []
    ys: list[np.ndarray] = []
    for t, y in stream(n, seed=seed, domain=domain, chunk_size=max(n, 1), **kwargs):
        texts.extend(t)
        ys.append(y)
    return Dataset(texts=texts, y=np.concatenate(ys), domain=domain)


def add_label_noise(ds: Dataset, p: float, *, seed: int = 0) -> Dataset:
    """Return a copy of ``ds`` with a fraction ``p`` of labels randomly flipped.

    Applied to *training* labels only, this mimics the noisy human/heuristic
    labels of real instruction data. It caps what a hard-label model can learn
    and creates the headroom that knowledge distillation recovers (the teacher's
    soft targets average the noise away).
    """
    if p <= 0:
        return Dataset(list(ds.texts), ds.y.copy(), ds.domain)
    rng = np.random.default_rng(seed)
    y = ds.y.copy()
    flip = rng.random(len(y)) < p
    y[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))
    return Dataset(list(ds.texts), y, ds.domain)


def split(
    ds: Dataset, *, frac_train: float = 0.7, frac_val: float = 0.15, seed: int = 0
) -> tuple[Dataset, Dataset, Dataset]:
    """Deterministic train/val/test split."""
    n = len(ds)
    idx = np.random.default_rng(seed).permutation(n)
    n_tr = int(n * frac_train)
    n_va = int(n * frac_val)
    parts = {
        "train": idx[:n_tr],
        "val": idx[n_tr : n_tr + n_va],
        "test": idx[n_tr + n_va :],
    }
    out = []
    for key in ("train", "val", "test"):
        sel = parts[key]
        out.append(Dataset([ds.texts[i] for i in sel], ds.y[sel], ds.domain))
    return tuple(out)  # type: ignore[return-value]
