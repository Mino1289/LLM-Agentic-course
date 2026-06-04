from __future__ import annotations

from rag.langsmith_env import ensure_langsmith_env

ensure_langsmith_env()

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator
