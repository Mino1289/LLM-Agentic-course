from __future__ import annotations

import os

from rag.langsmith_env import ensure_langsmith_env

ensure_langsmith_env()

def _noop_traceable(*_args, **_kwargs):  # type: ignore
    def _decorator(func):
        return func

    return _decorator


if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}:
    try:
        from langsmith import traceable
    except Exception:
        traceable = _noop_traceable
else:
    traceable = _noop_traceable
