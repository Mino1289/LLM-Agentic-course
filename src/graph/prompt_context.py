"""Contexte d'univers (tickers couverts) pour les invites LLM."""
from __future__ import annotations

from typing import Any

from src.config import TRACKED_TICKERS


def get_known_tickers(agent: Any, max_items: int = 3) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    allowed = set(TRACKED_TICKERS)
    metadata_list = getattr(agent.rag, "doc_metadata", []) or []
    for meta in metadata_list:
        if not isinstance(meta, dict):
            continue
        raw = str(meta.get("ticker", "")).strip().upper()
        if not raw or raw not in allowed:
            continue
        if raw not in seen:
            seen.add(raw)
            tickers.append(raw)
        if len(tickers) >= max_items:
            return tickers
    for fallback in TRACKED_TICKERS:
        if fallback not in seen:
            tickers.append(fallback)
            seen.add(fallback)
        if len(tickers) >= max_items:
            break
    return tickers


def format_universe_hint(agent: Any, max_items: int = 12) -> str:
    return ", ".join(get_known_tickers(agent, max_items=max_items))
