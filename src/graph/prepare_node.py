"""Noeud de préparation de la requête — normalisation, extraction tickers/metadata."""
from __future__ import annotations

import re
from typing import Any

from src.config import TRACKED_TICKERS
from src.graph.state import GraphState
from src.graph.tracing import traceable


def extract_query_tickers(query: str) -> list[str]:
    allowed_tickers = set(TRACKED_TICKERS)
    tickers: list[str] = []
    for ticker in re.findall(r"\b([A-Z]{2,5})\b", query):
        if ticker in allowed_tickers and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def extract_metadata_filter(query: str) -> dict[str, str]:
    filter_payload: dict[str, str] = {}
    tickers = extract_query_tickers(query)
    if len(tickers) == 1:
        filter_payload["ticker"] = tickers[0]
    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        filter_payload["year"] = year_match.group(1)
    return filter_payload


@traceable(name="prepare_query_node")
async def prepare_query_node(_agent: Any, state: GraphState) -> GraphState:
    raw = state.get("query", "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    tickers = extract_query_tickers(normalized)
    return {
        "normalized_query": normalized,
        "metadata_filter": extract_metadata_filter(normalized),
        "target_tickers": tickers,
        "stats": {"pipeline": "langgraph_finance_v2_agent"},
        "agent_iterations": 0,
        "tool_events": [],
        "report_artifacts": [],
    }
