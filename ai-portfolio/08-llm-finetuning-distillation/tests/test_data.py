from __future__ import annotations

import numpy as np

from distillkit import data


def test_label_space():
    assert data.N_CLASSES == 9
    assert len(data.LABELS) == 9
    assert data.LABEL_TO_ID[data.LABELS[3]] == 3


def test_generate_shape_and_range():
    ds = data.generate(500, seed=0, domain="source")
    assert len(ds) == 500
    assert ds.y.shape == (500,)
    assert ds.y.min() >= 0 and ds.y.max() < data.N_CLASSES
    # Every text carries a task instruction prefix.
    assert all(any(t in txt for t in data.TASK_INSTRUCTIONS.values()) for txt in ds.texts[:20])


def test_source_and_target_vocab_disjoint():
    src_words = {w for words in data.SOURCE_VOCAB.values() for w in words}
    tgt_words = {w for words in data.TARGET_VOCAB.values() for w in words}
    assert src_words.isdisjoint(tgt_words), "domain shift requires disjoint signal vocab"


def test_labels_roughly_balanced():
    ds = data.generate(4500, seed=1, domain="source")
    counts = np.bincount(ds.y, minlength=data.N_CLASSES)
    # Uniform sampling -> each class within +/-30% of the mean.
    mean = counts.mean()
    assert counts.min() > 0.7 * mean and counts.max() < 1.3 * mean


def test_streaming_is_chunked_and_bounded():
    seen = 0
    max_chunk = 0
    for texts, y in data.stream(2500, seed=2, chunk_size=400):
        assert len(texts) == len(y)
        max_chunk = max(max_chunk, len(texts))
        seen += len(texts)
    assert seen == 2500
    assert max_chunk <= 400  # memory is bounded by chunk_size


def test_stream_matches_generate():
    ds = data.generate(300, seed=5, domain="target")
    texts = []
    ys = []
    for t, y in data.stream(300, seed=5, domain="target", chunk_size=300):
        texts += t
        ys.append(y)
    assert texts == ds.texts
    assert np.array_equal(np.concatenate(ys), ds.y)


def test_split_partitions_without_overlap():
    ds = data.generate(1000, seed=0)
    tr, va, te = data.split(ds, seed=0)
    assert len(tr) + len(va) + len(te) == 1000
    assert len(tr) > len(va) and len(tr) > len(te)


def test_label_noise_flips_expected_fraction():
    ds = data.generate(4000, seed=0)
    noisy = data.add_label_noise(ds, 0.30, seed=1)
    changed = float((noisy.y != ds.y).mean())
    # ~30% flipped, minus the chance a random relabel equals the original.
    assert 0.20 < changed < 0.32
    assert noisy.texts == ds.texts  # texts untouched
