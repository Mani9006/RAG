"""Simulated token / cost accounting and per-tool latency model.

There is no real LLM, so token counts are estimated deterministically from the
size of prompts, arguments and results (a ~4-chars-per-token heuristic, the same
rule of thumb used for real tokenizers).  Costs use a simulated price sheet so
the numbers are stable and comparable across runs.  The latency model gives each
tool a base cost plus a size term, which drives the execution-timeline Gantt.
"""
from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class CostModel:
    """Simulated price sheet and latency model."""

    price_in_per_1k: float = 0.50      # $ per 1k input tokens (simulated)
    price_out_per_1k: float = 1.50     # $ per 1k output tokens (simulated)
    plan_tokens: int = 180             # tokens charged per planning/repair call

    # Base latency (ms) per tool + a per-token term.
    tool_latency_ms: tuple[tuple[str, float], ...] = (
        ("calculator", 8.0),
        ("unit_convert", 6.0),
        ("corpus_search", 22.0),
        ("code_check", 14.0),
    )
    ms_per_token: float = 0.05

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1000.0 * self.price_in_per_1k
            + tokens_out / 1000.0 * self.price_out_per_1k
        )

    def latency_ms(self, tool: str, tokens_out: int) -> float:
        base = dict(self.tool_latency_ms).get(tool, 10.0)
        return base + tokens_out * self.ms_per_token
