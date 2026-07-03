"""The Agent: the top-level orchestrator object.

An :class:`Agent` owns an :class:`LLMEngine`, a toolbox, a shared
:class:`Blackboard`, a :class:`MessageBus`, and an :class:`Executor`.  Calling
:meth:`solve` runs one task end-to-end and returns its :class:`ExecutionTrace`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .accounting import CostModel
from .blackboard import Blackboard, MessageBus
from .executor import Executor, ExecutionTrace
from .llm import LLMEngine, OfflineRuleEngine
from .tools import Tool, default_toolbox


@dataclass
class AgentConfig:
    self_correct: bool = True
    max_retries: int = 3
    max_workers: int = 4


class Agent:
    def __init__(self, engine: LLMEngine | None = None,
                 tools: dict[str, Tool] | None = None,
                 config: AgentConfig | None = None,
                 cost_model: CostModel | None = None):
        self.engine = engine or OfflineRuleEngine()
        self.tools = tools or default_toolbox()
        self.config = config or AgentConfig()
        self.cost_model = cost_model or CostModel()
        self.blackboard = Blackboard()
        self.bus = MessageBus()
        self.executor = Executor(
            self.engine, self.tools, self.cost_model, self.config.max_workers
        )

    def solve(self, task) -> ExecutionTrace:
        """Run one task end-to-end and return its execution trace."""
        return self.executor.run(
            task,
            self_correct=self.config.self_correct,
            max_retries=self.config.max_retries,
            blackboard=self.blackboard,
            bus=self.bus,
        )
