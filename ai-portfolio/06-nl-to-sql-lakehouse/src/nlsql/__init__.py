"""NL-to-SQL Lakehouse: DuckDB star schema + semantic layer + offline text-to-SQL."""
from __future__ import annotations

from .lakehouse import build_memory_fixture, connect
from .nl2sql import (DeterministicNLToSQL, GenerationResult, LLMNLToSQL,
                     NLToSQLEngine)
from .schema_linker import LinkResult, SchemaLinker
from .semantic import Dimension, Filter, Metric, Query, SemanticModel
from .validator import SQLValidationError, validate_sql

__all__ = [
    "connect", "build_memory_fixture",
    "SemanticModel", "Query", "Filter", "Metric", "Dimension",
    "SchemaLinker", "LinkResult",
    "DeterministicNLToSQL", "LLMNLToSQL", "NLToSQLEngine", "GenerationResult",
    "validate_sql", "SQLValidationError",
]

__version__ = "0.1.0"
