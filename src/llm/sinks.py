"""Gestion du token sink pour le streaming vers l'UI."""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Optional

from src.llm.types import LLMStreamChunk

# Module-level context variable for the token sink.
_token_sink_var: contextvars.ContextVar[Optional[Callable[[str], Any]]] = (
    contextvars.ContextVar("llm_token_sink", default=None)
)


@contextmanager
def token_sink(sink: Optional[Callable[[str], Any]]):
    """Context manager that registers a token sink for the duration of the block.

    The sink is called with each text delta as it streams.
    Supports both sync and async sinks.

    Example:
        with token_sink(lambda t: container.markdown(t)):
            async for event in agent.astream(...): ...
    """
    token = _token_sink_var.set(sink)
    try:
        yield
    finally:
        _token_sink_var.reset(token)


async def _call_token_sink(delta: str) -> None:
    """Await the registered token sink (if any) for a text delta.
    No-ops when no sink is registered or the delta is empty.
    """
    sink = _token_sink_var.get()
    if not sink or not delta:
        return
    result = sink(delta)
    if inspect.iscoroutine(result):
        await result
