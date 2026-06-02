from __future__ import annotations


try:
    from langsmith import traceable
except Exception:
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator
