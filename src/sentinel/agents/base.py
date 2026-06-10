"""Agent base class.

Every agent declares a system prompt, the subset of tools it may use, a
prompt builder, and a deterministic `simulate` implementation. `run` dispatches
to the live Claude agentic loop or the simulation depending on the configured
backend — the orchestrator and everything downstream is backend-agnostic.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sentinel.llm import AnthropicBackend, SimulationBackend, extract_json
from sentinel.telemetry import UsageMeter, log
from sentinel.tools import ToolBox


@dataclass
class AgentContext:
    backend: AnthropicBackend | SimulationBackend
    toolbox: ToolBox
    meter: UsageMeter
    run_id: str


class Agent(ABC):
    name: str = "agent"
    tools: list[str] = []
    system: str = ""

    @abstractmethod
    def build_prompt(self, payload: dict) -> str:
        """Render the task payload into the user prompt for the live model."""

    @abstractmethod
    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        """Deterministic, data-grounded fallback used by the simulation backend."""

    def run(self, payload: dict, ctx: AgentContext) -> dict:
        log("agent", "start", run_id=ctx.run_id, agent=self.name)
        if isinstance(ctx.backend, SimulationBackend):
            result = self.simulate(payload, ctx.toolbox)
        else:
            turn = ctx.backend.run_agent(
                agent_name=self.name,
                system=self.system,
                user_prompt=self.build_prompt(payload),
                toolbox=ctx.toolbox,
                tool_names=self.tools,
                meter=ctx.meter,
            )
            result = extract_json(turn.final_text)
            result.setdefault("_meta", {})["tool_calls"] = turn.tool_calls
        log("agent", "done", run_id=ctx.run_id, agent=self.name)
        return result

    @staticmethod
    def tool_json(toolbox: ToolBox, name: str, **kwargs) -> dict | list:
        """Helper for simulate(): execute a tool and parse its JSON result."""
        return json.loads(toolbox.execute(name, kwargs))


JSON_ONLY = (
    "Respond with a single JSON object and nothing else — no prose before or "
    "after it. Use tools as needed to ground every number in real data."
)
