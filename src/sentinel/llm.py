"""LLM backends.

AnthropicBackend runs the real agentic loop against the Claude API: the model
plans, calls tools from the ToolBox, observes results, and iterates until it
produces a final JSON answer. The loop is manual (not the SDK tool runner) so
the orchestrator keeps a hook on every tool call for auditing and budgets.

SimulationBackend signals to the agents that they should run their
deterministic, data-grounded fallback logic instead. It exists so the entire
platform — demo, tests, CI — runs offline at zero cost.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sentinel.config import get_settings
from sentinel.telemetry import UsageMeter, log
from sentinel.tools import ToolBox


class TokenBudgetExceeded(RuntimeError):
    pass


@dataclass
class AgentTurn:
    final_text: str
    iterations: int = 0
    tool_calls: list[dict] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Parse the agent's final answer as JSON, tolerating code fences/preamble."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


class SimulationBackend:
    """Marker backend: agents detect it and execute their deterministic logic."""

    name = "simulation"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # imported lazily so simulation mode needs no API key

        self._client = anthropic.Anthropic()
        self._settings = get_settings()

    def run_agent(
        self,
        *,
        agent_name: str,
        system: str,
        user_prompt: str,
        toolbox: ToolBox,
        tool_names: list[str],
        meter: UsageMeter,
    ) -> AgentTurn:
        settings = self._settings
        tools = toolbox.schemas(tool_names) if tool_names else []
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        turn = AgentTurn(final_text="")

        for _ in range(settings.max_agent_iterations):
            if meter.budget_exhausted:
                raise TokenBudgetExceeded(
                    f"run token budget of {meter.budget_tokens} exhausted "
                    f"(used {meter.total_tokens})"
                )

            kwargs: dict = {
                "model": settings.model,
                "max_tokens": settings.max_tokens_per_call,
                "system": system,
                "messages": messages,
                "thinking": {"type": "adaptive"},
            }
            if tools:
                kwargs["tools"] = tools

            response = self._client.messages.create(**kwargs)
            meter.record(agent_name, response.usage.input_tokens, response.usage.output_tokens)
            turn.iterations += 1

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                turn.final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return turn

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                output = toolbox.execute(block.name, dict(block.input))
                turn.tool_calls.append({"tool": block.name, "input": dict(block.input)})
                log("llm", "tool_call", agent=agent_name, tool=block.name, input=dict(block.input))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
            messages.append({"role": "user", "content": results})

        raise RuntimeError(
            f"{agent_name} exceeded max_agent_iterations={settings.max_agent_iterations}"
        )


def build_backend():
    settings = get_settings()
    if settings.llm_backend == "anthropic":
        return AnthropicBackend()
    return SimulationBackend()
