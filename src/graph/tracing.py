"""Tracing LangSmith + decorateur traceable."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.paths import ENV_FILE

LANGSMITH_ENDPOINTS = {
    "us": "https://api.smith.langchain.com",
    "eu": "https://eu.api.smith.langchain.com",
}


def ensure_langsmith_env() -> None:
    load_dotenv(ENV_FILE)
    tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}
    if not tracing:
        return
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
    if not endpoint:
        region = os.getenv("LANGSMITH_REGION", "eu").strip().lower()
        endpoint = LANGSMITH_ENDPOINTS.get(region, LANGSMITH_ENDPOINTS["eu"])
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)


ensure_langsmith_env()


def _noop_traceable(*_args, **_kwargs):
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
