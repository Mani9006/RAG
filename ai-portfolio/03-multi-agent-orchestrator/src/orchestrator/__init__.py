"""Multi-Agent Orchestrator — a from-scratch agentic framework.

A LangGraph/AutoGPT-class orchestrator built from clean primitives:

  * Tool / Agent / Blackboard / MessageBus core abstractions
  * A Planner that decomposes a task into a DAG of steps
  * An Executor that runs the DAG respecting dependencies, with parallelism
    across independent branches
  * A deterministic, fully-offline "LLM" engine (rule/pattern policy) behind an
    LLMEngine interface, so a real model can drop in unchanged
  * Guardrails: input/output validators, tool-arg schema validation, and a
    retry-with-self-correction loop
  * Token / cost accounting and a full execution trace log

Everything is deterministic and runs end-to-end with zero external API calls.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

from .dag import DAG, Ref, Step, CycleError
from .tools import (
    Tool,
    ToolError,
    CalculatorTool,
    UnitConverterTool,
    CorpusSearchTool,
    CodeCheckTool,
    default_toolbox,
)
from .blackboard import Blackboard, MessageBus
from .guardrails import (
    SchemaError,
    OutputValidationError,
    InputValidationError,
    validate_input,
    validate_args,
)
from .llm import LLMEngine, OfflineRuleEngine, Task
from .planner import Planner
from .executor import Executor, ExecutionTrace, StepTrace
from .agent import Agent, AgentConfig
from .accounting import CostModel

__all__ = [
    "SEED",
    "DAG",
    "Ref",
    "Step",
    "CycleError",
    "Tool",
    "ToolError",
    "CalculatorTool",
    "UnitConverterTool",
    "CorpusSearchTool",
    "CodeCheckTool",
    "default_toolbox",
    "Blackboard",
    "MessageBus",
    "SchemaError",
    "OutputValidationError",
    "InputValidationError",
    "validate_input",
    "validate_args",
    "LLMEngine",
    "OfflineRuleEngine",
    "Task",
    "Planner",
    "Executor",
    "ExecutionTrace",
    "StepTrace",
    "Agent",
    "AgentConfig",
    "CostModel",
]
