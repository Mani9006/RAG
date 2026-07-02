"""A small synthetic knowledge base for the corpus-search tool.

Each document carries a short text, a canonical numeric fact, its unit, and a
set of keywords.  Facts are common-knowledge constants so ground-truth answers
for benchmark tasks can be computed independently of the retriever.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KBDoc:
    doc_id: str
    text: str
    number: float
    unit: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


KNOWLEDGE_BASE: list[KBDoc] = [
    KBDoc(
        "kb-light",
        "The speed of light in a vacuum is 299792458 metres per second.",
        299792458.0,
        "m/s",
        ("speed", "light", "vacuum", "photon"),
    ),
    KBDoc(
        "kb-boil",
        "The boiling point of water at sea level is 100 degrees Celsius.",
        100.0,
        "C",
        ("boiling", "point", "water", "steam"),
    ),
    KBDoc(
        "kb-freeze",
        "The freezing point of water at sea level is 0 degrees Celsius.",
        0.0,
        "C",
        ("freezing", "point", "water", "ice"),
    ),
    KBDoc(
        "kb-gravity",
        "Standard acceleration due to gravity on Earth is 9.81 metres per second squared.",
        9.81,
        "m/s^2",
        ("gravity", "acceleration", "earth", "gravitational"),
    ),
    KBDoc(
        "kb-year",
        "There are 365 days in a common (non-leap) calendar year.",
        365.0,
        "days",
        ("days", "year", "calendar", "annual"),
    ),
    KBDoc(
        "kb-radius",
        "The mean radius of the Earth is approximately 6371 kilometres.",
        6371.0,
        "km",
        ("radius", "earth", "planet", "mean"),
    ),
    KBDoc(
        "kb-pressure",
        "Standard atmospheric pressure at sea level is 101325 pascals.",
        101325.0,
        "Pa",
        ("atmospheric", "pressure", "standard", "sea"),
    ),
    KBDoc(
        "kb-body",
        "Normal human body temperature is about 37 degrees Celsius.",
        37.0,
        "C",
        ("human", "body", "temperature", "normal"),
    ),
    KBDoc(
        "kb-sound",
        "The speed of sound in dry air at 20 degrees Celsius is about 343 metres per second.",
        343.0,
        "m/s",
        ("speed", "sound", "air", "acoustic"),
    ),
    KBDoc(
        "kb-week",
        "There are 7 days in one week.",
        7.0,
        "days",
        ("days", "week", "calendar", "weekly"),
    ),
]


def kb_by_id(doc_id: str) -> KBDoc:
    for d in KNOWLEDGE_BASE:
        if d.doc_id == doc_id:
            return d
    raise KeyError(doc_id)
