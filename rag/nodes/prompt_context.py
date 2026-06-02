from __future__ import annotations

from typing import Any


DEFAULT_FALLBACK_TICKERS = ["NVDA", "AMD", "INTC", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "ORCL", "PLTR"]


def get_known_tickers(agent: Any, max_items: int = 12) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    metadata_list = getattr(agent.rag, "doc_metadata", []) or []
    for meta in metadata_list:
        if not isinstance(meta, dict):
            continue
        raw = str(meta.get("ticker", "")).strip().upper()
        if not raw:
            continue
        if raw not in seen:
            seen.add(raw)
            tickers.append(raw)
        if len(tickers) >= max_items:
            return tickers

    for fallback in DEFAULT_FALLBACK_TICKERS:
        if fallback not in seen:
            tickers.append(fallback)
            seen.add(fallback)
        if len(tickers) >= max_items:
            break
    return tickers


def format_universe_hint(agent: Any, max_items: int = 12) -> str:
    tickers = get_known_tickers(agent, max_items=max_items)
    return ", ".join(tickers)
