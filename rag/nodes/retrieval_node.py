from __future__ import annotations

import re
from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


def _infer_doc_type(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source", "")).lower()
    section = str(metadata.get("section", "")).lower()

    if "10-q" in source or "10q" in source:
        return "10-Q"
    if "10-k" in source or "10k" in source:
        return "10-K"
    if "8-k" in source or "8k" in source or section == "earnings_8k":
        return "8-K"
    if section == "earnings_call" or re.search(r"(transcript|earnings[_\- ]?call|conference[_\- ]?call)", source):
        return "EARNINGS_CALL"
    return "OTHER"


def _match_scope(
    metadata: dict[str, Any],
    scoped_tickers: set[str],
    scoped_doc_types: set[str],
) -> bool:
    if scoped_tickers:
        ticker = str(metadata.get("ticker", "")).upper()
        if ticker not in scoped_tickers:
            return False
    if scoped_doc_types:
        inferred = _infer_doc_type(metadata)
        if inferred not in scoped_doc_types:
            return False
    return True


def _apply_scope_filter(
    agent: Any,
    indices: list[int],
    scoped_tickers: set[str],
    scoped_doc_types: set[str],
) -> list[int]:
    if not scoped_tickers and not scoped_doc_types:
        return indices
    filtered: list[int] = []
    for idx in indices:
        metadata = agent.rag.doc_metadata[idx]
        if _match_scope(metadata, scoped_tickers, scoped_doc_types):
            filtered.append(idx)
    return filtered


def _retrieve_with_fallbacks(
    agent: Any,
    query: str,
    filters_to_try: list[dict[str, str] | None],
    scoped_tickers: set[str],
    scoped_doc_types: set[str],
) -> list[int]:
    for candidate_filter in filters_to_try:
        retrieval = agent.rag.retrieve(
            query,
            search_mode="vector",
            use_reranking=False,
            metadata_filter=candidate_filter,
            top_k=24,
            candidate_pool=40,
        )
        scoped_indices = _apply_scope_filter(
            agent,
            retrieval.chunk_indices,
            scoped_tickers=scoped_tickers,
            scoped_doc_types=scoped_doc_types,
        )
        if scoped_indices:
            return scoped_indices
        if retrieval.chunk_indices and not scoped_tickers and not scoped_doc_types:
            return retrieval.chunk_indices
    return []


@traceable(name="multi_retrieve_node")
def multi_retrieve_node(agent: Any, state: GraphState) -> GraphState:
    queries = state.get("decomposed_queries") or [state["normalized_query"]]
    metadata_filter = state.get("metadata_filter") or {}
    scoped_tickers = {str(t).upper() for t in (state.get("target_tickers") or []) if str(t).strip()}
    scoped_doc_types = {str(dt).upper() for dt in (state.get("doc_type_priority") or []) if str(dt).strip()}

    strict_filter: dict[str, str] = {}
    if metadata_filter.get("year"):
        strict_filter["year"] = metadata_filter["year"]
    if metadata_filter.get("ticker") and (not scoped_tickers or len(scoped_tickers) <= 1):
        strict_filter["ticker"] = metadata_filter["ticker"]

    filters_to_try: list[dict[str, str] | None] = []
    if strict_filter:
        filters_to_try.append(strict_filter)
    if metadata_filter and metadata_filter not in filters_to_try:
        filters_to_try.append(dict(metadata_filter))
    relaxed_filter = dict(metadata_filter)
    relaxed_filter.pop("year", None)
    if relaxed_filter and relaxed_filter not in filters_to_try:
        filters_to_try.append(relaxed_filter)
    filters_to_try.append(None)

    all_indices: list[int] = []
    for query in queries:
        query_indices: list[int] = []

        # Multi-company balancing: force one retrieval pass per ticker when comparing.
        if len(scoped_tickers) > 1:
            for ticker in sorted(scoped_tickers):
                ticker_filters: list[dict[str, str] | None] = []
                for base_filter in filters_to_try:
                    if base_filter is None:
                        ticker_filters.append({"ticker": ticker})
                    else:
                        scoped_filter = dict(base_filter)
                        scoped_filter["ticker"] = ticker
                        ticker_filters.append(scoped_filter)
                per_ticker_indices = _retrieve_with_fallbacks(
                    agent,
                    query,
                    ticker_filters,
                    scoped_tickers={ticker},
                    scoped_doc_types=scoped_doc_types,
                )
                query_indices.extend(per_ticker_indices[:12])

        if not query_indices:
            query_indices = _retrieve_with_fallbacks(
                agent,
                query,
                filters_to_try,
                scoped_tickers=scoped_tickers,
                scoped_doc_types=scoped_doc_types,
            )

        if not query_indices:
            fallback = agent.rag.retrieve(
                query,
                search_mode="vector",
                use_reranking=False,
                metadata_filter=None,
                top_k=18,
                candidate_pool=24,
            )
            query_indices.extend(fallback.chunk_indices)

        all_indices.extend(query_indices)

    dedup_indices = agent.rag._deduplicate_indices(all_indices)
    stats = state.get("stats", {})
    stats.update(
        {
            "decomposed_query_count": len(queries),
            "retrieval_candidate_count": len(dedup_indices),
            "retrieval_scoped_tickers": sorted(scoped_tickers),
            "retrieval_scoped_doc_types": sorted(scoped_doc_types),
        }
    )
    return {
        "candidate_indices": dedup_indices,
        "stats": stats,
    }
