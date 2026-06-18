"""Market price tool — retrieve stock prices via yfinance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.tools.schemas import MarketPriceArgs
from src.tools.descriptions import _normalize_tickers


def _widen_single_day(
    start_date: str, end_date: str, padding_days: int = 5
) -> tuple[str, str]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return start_date, end_date
    span = (end - start).days
    if span < padding_days:
        mid = start + (end - start) / 2
        new_start = mid - timedelta(days=padding_days // 2)
        new_end = mid + timedelta(days=padding_days // 2)
        return new_start.isoformat(), new_end.isoformat()
    return start_date, end_date


def run_market_price_tool(
    args: MarketPriceArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    from src.tools.descriptions import _tracked_tickers_text
    from src.graph.tool_nodes import fetch_price_context

    normalized = _normalize_tickers(args.tickers)
    start_date, end_date = _widen_single_day(args.start_date, args.end_date)
    if not normalized:
        return {
            "text": f"No valid tickers provided. Use {_tracked_tickers_text()}.",
            "price_context": "",
        }
    summary = fetch_price_context(agent, normalized, start_date, end_date)
    if not summary:
        return {
            "text": f"No price data for {normalized} between {start_date} and {end_date}.",
            "price_context": "",
        }
    return {"text": summary, "price_context": summary}
