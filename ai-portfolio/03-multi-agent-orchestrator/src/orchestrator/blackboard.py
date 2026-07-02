"""Shared memory (Blackboard) and a lightweight publish/subscribe MessageBus.

Both are thread-safe because the executor runs independent DAG branches on a
thread pool.  The Blackboard is the agents' shared key/value memory with an
append-only write history; the MessageBus carries execution events between
components (and to any external observer/logger).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Write:
    key: str
    value: Any
    author: str
    ts: float


class Blackboard:
    """A thread-safe shared key/value store with write history."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, Any] = {}
        self._history: list[Write] = []

    def write(self, key: str, value: Any, author: str = "system") -> None:
        with self._lock:
            self._store[key] = value
            self._history.append(Write(key, value, author, time.time()))

    def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._store)

    @property
    def history(self) -> list[Write]:
        with self._lock:
            return list(self._history)


@dataclass
class Message:
    topic: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


class MessageBus:
    """A minimal in-process pub/sub bus with a retained event log."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._log: list[Message] = []

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        with self._lock:
            self._subs[topic].append(handler)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        msg = Message(topic, payload)
        with self._lock:
            self._log.append(msg)
            handlers = list(self._subs.get(topic, [])) + list(self._subs.get("*", []))
        for h in handlers:
            h(msg)

    @property
    def log(self) -> list[Message]:
        with self._lock:
            return list(self._log)
