"""SEC Filings RAG tool — query ChromaDB via decomposed multi-query retrieval."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.tools.schemas import SecFilingsRAGArgs
from src.tools.descriptions import (
    _normalize_doc_types,
    _normalize_tickers,
    _normalize_years,
    format_rag_excerpts,
    _tracked_tickers_text,
)

_LOGGER = logging.getLogger("src.tools.sec_filings")


async def run_sec_filings_rag(
    args: SecFilingsRAGArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    from src.graph.decompose_node import decompose_query
    from src.graph.retrieval_node import multi_retrieve_node
    from src.graph.rerank_node import _balanced_rerank_indices, _ticker_counts

    pipeline_t0 = time.perf_counter()
    normalized_tickers = _normalize_tickers(args.tickers)
    normalized_years = _normalize_years(args.years)
    normalized_doc_types = _normalize_doc_types(args.doc_types)
    query = args.query

    metadata_filter: dict[str, str] = {}
    if len(normalized_years) == 1:
        metadata_filter["year"] = normalized_years[0]
    if len(normalized_tickers) == 1:
        metadata_filter["ticker"] = normalized_tickers[0]

    decomposed: list[str]
    decompose_skipped_reason: str | None = None
    if len(normalized_tickers) == 1:
        decomposed = [query]
        decompose_skipped_reason = "single_ticker"
    else:
        decompose_t0 = time.perf_counter()
        decomposed = await decompose_query(agent, query)
        _LOGGER.info("decompose took %.2fs (count=%d)", time.perf_counter() - decompose_t0, len(decomposed))

    rag_state: dict[str, Any] = {
        "normalized_query": query,
        "metadata_filter": metadata_filter,
        "target_tickers": normalized_tickers,
        "doc_type_priority": normalized_doc_types,
        "decomposed_queries": decomposed,
        "stats": {"decomposed_count": len(decomposed)},
    }
    if decompose_skipped_reason:
        rag_state["stats"]["decompose_skipped_reason"] = decompose_skipped_reason

    retrieve_t0 = time.perf_counter()
    retrieve_result = await multi_retrieve_node(agent, rag_state)
    rag_state.update(retrieve_result)
    candidates = rag_state.get("candidate_indices", [])
    if not candidates:
        return {
            "text": format_rag_excerpts([], []),
            "final_chunks": [], "final_metadatas": [],
            "stats": rag_state.get("stats", {}),
        }

    top_indices = await _balanced_rerank_indices(agent, rag_state, candidates)
    final_chunks = [agent.rag.documents[idx] for idx in top_indices]
    final_metadatas = [agent.rag.doc_metadata[idx] for idx in top_indices]
    stats = rag_state.get("stats", {})
    stats.update({
        "rerank_final_ticker_counts": _ticker_counts(final_metadatas),
        "rerank_final_count": len(top_indices),
        "chunks_used": len(final_chunks),
    })
    return {
        "text": format_rag_excerpts(final_chunks, final_metadatas),
        "final_chunks": final_chunks, "final_metadatas": final_metadatas, "stats": stats,
    }
