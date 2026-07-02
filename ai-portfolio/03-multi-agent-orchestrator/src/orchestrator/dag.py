"""DAG primitives: Step, Ref, and a topological-sort executable DAG.

A plan is an ordered list of :class:`Step`.  Each step names a tool, a dict of
arguments, and explicit dependencies.  Arguments may contain :class:`Ref`
sentinels that point at the output of an earlier step; the DAG derives edges
from those references as well as from any explicitly declared ``deps``.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

# Matches a template variable like "$s1" or "$s2.number" embedded in a string arg.
_STR_REF = re.compile(r"\$([A-Za-z]\w*)(?:\.(\w+))?")


class CycleError(ValueError):
    """Raised when a plan's dependency graph contains a cycle."""


@dataclass(frozen=True)
class Ref:
    """A reference to (a field of) another step's result.

    ``Ref("s1")`` resolves to the default ``value`` field of step ``s1``'s
    result; ``Ref("s1", "number")`` resolves to that result's ``number`` field.
    """

    step_id: str
    field: str = "value"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"${self.step_id}.{self.field}"


@dataclass
class Step:
    """A single node in a plan: one tool invocation."""

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    deps: list[str] = field(default_factory=list)
    note: str = ""

    def referenced_steps(self) -> set[str]:
        """Step ids referenced by this step's args (implicit dependencies)."""
        found: set[str] = set()
        for v in self.args.values():
            for ref in _iter_refs(v):
                found.add(ref.step_id)
        return found

    def all_deps(self) -> set[str]:
        return set(self.deps) | self.referenced_steps()


def _iter_refs(value: Any) -> Iterable[Ref]:
    if isinstance(value, Ref):
        yield value
    elif isinstance(value, str):
        for m in _STR_REF.finditer(value):
            yield Ref(m.group(1), m.group(2) or "value")
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_refs(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_refs(v)


class DAG:
    """A directed acyclic graph of steps with a level-based schedule.

    Levels group steps that can run in parallel: every step in level *k* depends
    only on steps in levels ``< k``.  This is Kahn's algorithm, and it doubles as
    the cycle detector.
    """

    def __init__(self, steps: Iterable[Step]):
        self.steps: dict[str, Step] = {}
        for s in steps:
            if s.id in self.steps:
                raise ValueError(f"duplicate step id: {s.id}")
            self.steps[s.id] = s
        self._validate_refs()
        self.levels: list[list[str]] = self._toposort_levels()

    def _validate_refs(self) -> None:
        for s in self.steps.values():
            for dep in s.all_deps():
                if dep not in self.steps:
                    raise ValueError(
                        f"step {s.id!r} depends on unknown step {dep!r}"
                    )
                if dep == s.id:
                    raise CycleError(f"step {s.id!r} depends on itself")

    def _toposort_levels(self) -> list[list[str]]:
        indeg: dict[str, int] = {sid: 0 for sid in self.steps}
        children: dict[str, list[str]] = {sid: [] for sid in self.steps}
        for sid, s in self.steps.items():
            for dep in s.all_deps():
                indeg[sid] += 1
                children[dep].append(sid)

        # Deterministic ordering: sort ids within each level.
        ready = deque(sorted(sid for sid, d in indeg.items() if d == 0))
        levels: list[list[str]] = []
        seen = 0
        while ready:
            level = sorted(ready)
            ready.clear()
            levels.append(level)
            for sid in level:
                seen += 1
                for child in sorted(children[sid]):
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        ready.append(child)
        if seen != len(self.steps):
            raise CycleError("plan dependency graph contains a cycle")
        return levels

    def topo_order(self) -> list[str]:
        """A flat topological ordering (levels concatenated)."""
        return [sid for level in self.levels for sid in level]

    def sinks(self) -> list[str]:
        """Steps that nothing else depends on (candidate final answers)."""
        depended: set[str] = set()
        for s in self.steps.values():
            depended |= s.all_deps()
        return [sid for sid in self.topo_order() if sid not in depended]

    def width(self) -> int:
        """Maximum number of steps runnable in parallel (widest level)."""
        return max((len(level) for level in self.levels), default=0)
