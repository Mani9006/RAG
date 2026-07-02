"""Guardrails: schema validation, input validation, output contracts."""
from __future__ import annotations

import pytest

from orchestrator.dag import Ref
from orchestrator.guardrails import (
    InputValidationError,
    OutputValidationError,
    SchemaError,
    validate_args,
    validate_input,
    validate_output,
)
from orchestrator.tools import CalculatorTool, UnitConverterTool


def test_missing_required_arg_rejected():
    tool = UnitConverterTool()
    with pytest.raises(SchemaError):
        validate_args(tool.name, tool.schema, {"value": 5.0, "from_unit": "km"})


def test_unknown_arg_rejected():
    tool = CalculatorTool()
    with pytest.raises(SchemaError):
        validate_args(tool.name, tool.schema, {"expression": "1+1", "extra": 3})


def test_wrong_type_rejected():
    tool = UnitConverterTool()
    with pytest.raises(SchemaError):
        validate_args(tool.name, tool.schema,
                      {"value": "not-a-number", "from_unit": "km", "to_unit": "mi"})


def test_bool_not_accepted_as_number():
    tool = UnitConverterTool()
    with pytest.raises(SchemaError):
        validate_args(tool.name, tool.schema,
                      {"value": True, "from_unit": "km", "to_unit": "mi"})


def test_valid_args_pass():
    tool = UnitConverterTool()
    validate_args(tool.name, tool.schema,
                  {"value": 5.0, "from_unit": "km", "to_unit": "mi"})


def test_refs_bypass_static_typecheck():
    tool = UnitConverterTool()
    # A Ref stands in for a value resolved at runtime; static check must allow it.
    validate_args(tool.name, tool.schema,
                  {"value": Ref("s1"), "from_unit": "km", "to_unit": "mi"})


def test_input_validation():
    validate_input("Convert 5 km to mi.")
    with pytest.raises(InputValidationError):
        validate_input("   ")
    with pytest.raises(InputValidationError):
        validate_input("x" * 5000)


def test_output_contracts():
    validate_output(42.0, "finite")
    validate_output(37.5, "percentage")
    with pytest.raises(OutputValidationError):
        validate_output(float("inf"), "finite")
    with pytest.raises(OutputValidationError):
        validate_output(0.375, "percentage")   # bare fraction
    with pytest.raises(OutputValidationError):
        validate_output(150.0, "percentage")   # out of range
    with pytest.raises(OutputValidationError):
        validate_output("", "nonempty")
