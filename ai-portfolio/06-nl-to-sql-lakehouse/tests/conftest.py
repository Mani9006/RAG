from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from nlsql import DeterministicNLToSQL, SemanticModel, build_memory_fixture


@pytest.fixture(scope="session")
def model() -> SemanticModel:
    return SemanticModel.load()


@pytest.fixture(scope="session")
def con():
    c = build_memory_fixture(seed=3, rows=5_000)
    yield c
    c.close()


@pytest.fixture()
def engine(model, con) -> DeterministicNLToSQL:
    return DeterministicNLToSQL(model, con=con)
