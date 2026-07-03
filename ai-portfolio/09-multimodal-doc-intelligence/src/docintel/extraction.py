"""Field extraction: token classification, key/value pairing, table recovery.

Three real components:

* :class:`TokenClassifier` — a scikit-learn token-classification model. It
  featurizes every token (see :mod:`docintel.features`), standardizes, and
  fits a multinomial logistic-regression head that types each token into one
  of :data:`docintel.LABELS`. Logistic regression is deliberate: it is linear,
  streams to millions of tokens, and exposes calibrated per-class scores, yet
  the engineered layout features let it comfortably beat the majority baseline.
* :func:`pair_key_values` — geometric key→value pairing. For every printed KEY
  entity it links the value entity that is nearest to the right on the same line
  (falling back to directly below), exactly as heuristic KV linkers do over
  LayoutLM entity spans.
* :func:`extract_table` — line-item TABLE extraction. Value tokens are clustered
  into **rows** (1-D clustering on ``y``) and **columns** (on ``x``) to recover
  the 2-D grid without any supervision on the grid itself.

Everything here consumes generic ``(text, x, y, w, h)`` tokens, so a real OCR
backend drops in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import ID_TO_LABEL, LABEL_TO_ID, LABELS, LI_COLUMNS
from .features import extract_features


# ---------------------------------------------------------------------------
# Token classifier
# ---------------------------------------------------------------------------
class TokenClassifier:
    """Layout-aware token → field-type classifier.

    A ``StandardScaler`` + multinomial ``SGDClassifier`` (log-loss) — a linear
    softmax head trained by stochastic gradient descent. SGD is chosen for
    scale: it is O(n) per epoch, supports ``partial_fit`` for out-of-core
    streaming, and trains on millions of tokens in seconds while the engineered
    layout features carry the accuracy.
    """

    def __init__(self, alpha: float = 1e-5, max_iter: int = 50, seed: int = 42) -> None:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss", alpha=alpha, max_iter=max_iter, tol=1e-4,
                early_stopping=True, n_iter_no_change=8, validation_fraction=0.05,
                class_weight="balanced", random_state=seed)),
        ])
        self.classes_: np.ndarray | None = None

    def fit(self, table: dict[str, list]) -> "TokenClassifier":
        X = extract_features(table)
        y = np.array([LABEL_TO_ID[l] for l in table["label"]])
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, table: dict[str, list]) -> np.ndarray:
        X = extract_features(table)
        return self.pipeline.predict(X)

    def predict_labels(self, table: dict[str, list]) -> list[str]:
        return [ID_TO_LABEL[int(i)] for i in self.predict(table)]

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)


def majority_baseline(train_labels: list[str], eval_labels: list[str]) -> float:
    """Accuracy of always predicting the most frequent training label."""
    vals, counts = np.unique(train_labels, return_counts=True)
    maj = vals[int(np.argmax(counts))]
    ev = np.asarray(eval_labels)
    return float(np.mean(ev == maj))


# ---------------------------------------------------------------------------
# Entities (contiguous same-label tokens on one text line)
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    label: str
    text: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def group_entities(tokens: list[dict], labels: list[str],
                   line_tol: float = 6.0, gap_tol: float = 40.0) -> list[Entity]:
    """Merge adjacent same-label tokens on the same line into entities.

    ``tokens`` is a list of dicts with ``x,y,w,h,text``. Tokens are grouped when
    they share a label, sit on the same text line (|Δy|<``line_tol``) and are
    horizontally close (gap<``gap_tol``).
    """
    idx = sorted(range(len(tokens)), key=lambda i: (round(tokens[i]["y"] / line_tol),
                                                     tokens[i]["x"]))
    ents: list[Entity] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        xs = [tokens[i]["x"] for i in cur]
        ys = [tokens[i]["y"] for i in cur]
        x1 = min(tokens[i]["x"] + tokens[i]["w"] for i in cur)
        x0 = min(xs)
        x_end = max(tokens[i]["x"] + tokens[i]["w"] for i in cur)
        y0 = min(ys)
        y_end = max(tokens[i]["y"] + tokens[i]["h"] for i in cur)
        text = " ".join(tokens[i]["text"] for i in cur)
        ents.append(Entity(labels[cur[0]], text, x0, y0, x_end - x0, y_end - y0))

    for i in idx:
        if labels[i] == "O":
            continue
        if not cur:
            cur = [i]
            continue
        j = cur[-1]
        same_line = abs(tokens[i]["y"] - tokens[j]["y"]) <= line_tol
        near = tokens[i]["x"] - (tokens[j]["x"] + tokens[j]["w"]) <= gap_tol
        if labels[i] == labels[j] and same_line and near:
            cur.append(i)
        else:
            flush()
            cur = [i]
    flush()
    return ents


# ---------------------------------------------------------------------------
# Geometric key/value pairing
# ---------------------------------------------------------------------------
@dataclass
class KVPair:
    key_text: str
    value_label: str
    value_text: str
    key_box: tuple[float, float, float, float]
    value_box: tuple[float, float, float, float]


def pair_key_values(entities: list[Entity],
                    right_max_dx: float = 240.0,
                    row_tol: float = 8.0,
                    below_max_dy: float = 34.0) -> list[KVPair]:
    """Link each KEY entity to its value entity by geometry.

    Preference order: nearest value to the **right** on the same line; else the
    nearest value **below** and horizontally overlapping. Non-KEY, non-``O``
    entities are the value candidates.
    """
    keys = [e for e in entities if e.label == "KEY"]
    values = [e for e in entities if e.label not in ("KEY", "O", "HEADER")]
    pairs: list[KVPair] = []
    for k in keys:
        best = None
        best_cost = float("inf")
        for v in values:
            same_line = abs(v.cy - k.cy) <= row_tol
            to_right = v.x >= k.x + k.w - 2
            if same_line and to_right:
                dx = v.x - (k.x + k.w)
                if 0 - 4 <= dx <= right_max_dx and dx < best_cost:
                    best, best_cost = v, dx
        if best is None:
            for v in values:
                below = 0 <= (v.y - (k.y + k.h)) <= below_max_dy
                overlap = abs(v.cx - k.cx) <= 80
                if below and overlap:
                    dy = v.y - (k.y + k.h)
                    if dy < best_cost:
                        best, best_cost = v, dy
        if best is not None:
            pairs.append(KVPair(k.text, best.label, best.text,
                                (k.x, k.y, k.w, k.h),
                                (best.x, best.y, best.w, best.h)))
    return pairs


# ---------------------------------------------------------------------------
# Table extraction via 1-D row / column clustering
# ---------------------------------------------------------------------------
def _cluster_1d(values: np.ndarray, gap: float) -> np.ndarray:
    """Cluster sorted 1-D coordinates: new cluster when gap exceeds ``gap``.

    Returns a cluster id per input element (in original order).
    """
    order = np.argsort(values)
    ids = np.zeros(len(values), dtype=int)
    cid = 0
    prev = None
    for rank, i in enumerate(order):
        if prev is not None and values[i] - prev > gap:
            cid += 1
        ids[i] = cid
        prev = values[i]
    return ids


def extract_table(tokens: list[dict], labels: list[str],
                  row_gap: float = 9.0, col_gap: float = 62.0) -> list[list[str]]:
    """Recover the line-item grid from LI_* tokens via row/col clustering.

    Rows come from clustering token ``y`` centers; columns from clustering ``x``
    centers. The recovered grid is returned as a list of rows, each a list of
    cell strings ordered left-to-right. No supervision on the grid is used.
    """
    li = [(t, lab) for t, lab in zip(tokens, labels) if lab.startswith("LI_")]
    if not li:
        return []
    ys = np.array([t["y"] + t["h"] / 2 for t, _ in li])
    xs = np.array([t["x"] + t["w"] / 2 for t, _ in li])
    row_ids = _cluster_1d(ys, row_gap)
    col_ids = _cluster_1d(xs, col_gap)

    n_rows = row_ids.max() + 1
    n_cols = col_ids.max() + 1
    # order rows top→bottom, cols left→right by mean coordinate
    row_order = np.argsort([ys[row_ids == r].mean() for r in range(n_rows)])
    col_order = np.argsort([xs[col_ids == c].mean() for c in range(n_cols)])
    row_rank = {r: i for i, r in enumerate(row_order)}
    col_rank = {c: i for i, c in enumerate(col_order)}

    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for (t, _), r, c in zip(li, row_ids, col_ids):
        rr, cc = row_rank[r], col_rank[c]
        cell = grid[rr][cc]
        grid[rr][cc] = (cell + " " + t["text"]).strip() if cell else t["text"]
    return grid


def tokens_of_doc(table: dict[str, list], doc_id: int) -> tuple[list[dict], list[str]]:
    """Extract the token dicts + gold labels for one ``doc_id`` from a table."""
    toks, labs = [], []
    for i, d in enumerate(table["doc_id"]):
        if d == doc_id:
            toks.append({"text": table["text"][i], "x": table["x"][i],
                         "y": table["y"][i], "w": table["w"][i],
                         "h": table["h"][i]})
            labs.append(table["label"][i])
    return toks, labs
