from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Literal

ProgressStatus = Literal["running", "completed", "failed"]

_progress_queue: ContextVar[asyncio.Queue[dict[str, Any]] | None] = ContextVar(
    "orchestration_progress_queue",
    default=None,
)


def bind_progress_queue(queue: asyncio.Queue[dict[str, Any]]) -> None:
    _progress_queue.set(queue)


def clear_progress_queue() -> None:
    _progress_queue.set(None)


def emit_agent_progress(
    agent: str,
    status: ProgressStatus,
    message: str,
    *,
    detail: str = "",
) -> None:
    queue = _progress_queue.get()
    if queue is None:
        return
    try:
        queue.put_nowait(
            {
                "event": "on_spoke_event",
                "agent": agent,
                "status": status,
                "message": message,
                "detail": detail,
                "tool_events": [],
            }
        )
    except asyncio.QueueFull:
        pass
