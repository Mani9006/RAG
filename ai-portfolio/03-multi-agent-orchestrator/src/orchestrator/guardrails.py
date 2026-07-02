"""Guardrails: input validation, tool-arg schema validation, output validation.

These are deliberately separate from the tools so the same validators guard any
tool.  Each raises a distinct exception type so the executor can attribute a
failure to the right stage and drive the self-correction loop accordingly.
"""
from __future__ import annotations

import math
from typing import Any

from .dag import Ref


class InputValidationError(ValueError):
    """The incoming task itself is malformed (empty / too long / unsafe)."""


class SchemaError(ValueError):
    """A tool call's arguments do not satisfy the tool's declared schema."""


class OutputValidationError(ValueError):
    """A step's (or task's) output failed a semantic validator."""


MAX_PROMPT_CHARS = 2_000


def validate_input(prompt: str) -> None:
    """Validate a raw task prompt before any planning happens."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputValidationError("task prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InputValidationError(
            f"task prompt too long ({len(prompt)} > {MAX_PROMPT_CHARS} chars)"
        )


def validate_args(tool_name: str, schema: dict[str, dict[str, Any]],
                  args: dict[str, Any]) -> None:
    """Type-check ``args`` against a tool ``schema``.

    Ref sentinels are accepted for any field: they are resolved to concrete
    values at execution time and re-validated then.  Raises :class:`SchemaError`
    for missing required args, unknown args, or type mismatches.
    """
    schema = schema or {}
    required = {k for k, spec in schema.items() if spec.get("required", True)}
    missing = required - set(args)
    if missing:
        raise SchemaError(
            f"{tool_name}: missing required argument(s): {sorted(missing)}"
        )
    unknown = set(args) - set(schema)
    if unknown:
        raise SchemaError(f"{tool_name}: unknown argument(s): {sorted(unknown)}")
    for name, value in args.items():
        if isinstance(value, Ref):
            continue  # resolved + re-checked at execution time
        expected = schema[name]["type"]
        if not _type_ok(value, expected):
            raise SchemaError(
                f"{tool_name}: argument {name!r} expected "
                f"{_type_name(expected)}, got {type(value).__name__}"
            )


def _type_ok(value: Any, expected: Any) -> bool:
    # bool is a subclass of int but should not satisfy a numeric field silently
    if isinstance(value, bool) and expected not in (bool,):
        return False
    return isinstance(value, expected)


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return "|".join(t.__name__ for t in expected)
    return expected.__name__


def validate_output(value: Any, kind: str) -> None:
    """Validate a resolved output against a named semantic contract.

    Supported ``kind`` values:
      * ``finite``    — a finite real number
      * ``percentage``— a finite number in [0, 100]
      * ``nonempty``  — a non-empty string/collection
    """
    if kind == "finite":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OutputValidationError(f"expected a number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise OutputValidationError("value is not finite")
    elif kind == "percentage":
        validate_output(value, "finite")
        v = float(value)
        if not (0.0 <= abs(v) <= 100.0):
            raise OutputValidationError(
                f"percentage out of range [0, 100]: {value}"
            )
        # Plausibility guard: a value in (0, 1) is almost certainly a bare
        # fraction where the x100 scaling was forgotten (a common LLM slip).
        if 0.0 < abs(v) < 1.0:
            raise OutputValidationError(
                f"value {value} looks like a fraction, not a percentage (expected 0-100)"
            )
    elif kind == "nonempty":
        if value is None or (hasattr(value, "__len__") and len(value) == 0):
            raise OutputValidationError("value is empty")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown output contract: {kind!r}")
