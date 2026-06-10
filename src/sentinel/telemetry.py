"""Structured logging and per-run usage accounting.

Every pipeline run gets a run_id; every agent invocation and tool call is
emitted as a structured JSON log line so the platform is greppable and
ship-able to any log aggregator without code changes.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field

# Live-mode pricing (USD per million tokens) used for cost telemetry only.
PRICING_PER_MTOK = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("sentinel")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def log(logger: str, msg: str, **ctx) -> None:
    logging.getLogger(f"sentinel.{logger}").info(msg, extra={"ctx": ctx})


def new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass
class UsageMeter:
    """Accumulates token usage across all agent calls in a run and enforces
    the run-level token budget."""

    model: str
    budget_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    per_agent: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        bucket = self.per_agent.setdefault(agent, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["calls"] += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def budget_exhausted(self) -> bool:
        return self.total_tokens >= self.budget_tokens

    @property
    def estimated_cost_usd(self) -> float:
        in_rate, out_rate = PRICING_PER_MTOK.get(self.model, (5.0, 25.0))
        return round(self.input_tokens / 1e6 * in_rate + self.output_tokens / 1e6 * out_rate, 4)

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "per_agent": self.per_agent,
        }
