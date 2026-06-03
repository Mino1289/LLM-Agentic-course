from __future__ import annotations

import re
from typing import Any

from rag.config import TRACKED_TICKERS
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


def extract_metadata_filter(query: str) -> dict[str, str]:
    filter_payload: dict[str, str] = {}
    allowed_tickers = set(TRACKED_TICKERS)

    for ticker in re.findall(r"\b([A-Z]{2,5})\b", query):
        if ticker in allowed_tickers:
            filter_payload["ticker"] = ticker
            break

    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        filter_payload["year"] = year_match.group(1)

    return filter_payload


@traceable(name="prepare_query_node")
def prepare_query_node(_agent: Any, state: GraphState) -> GraphState:
    raw = state.get("query", "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    return {
        "normalized_query": normalized,
        "metadata_filter": extract_metadata_filter(normalized),
        "stats": {"pipeline": "langgraph_finance_v1"},
    }
