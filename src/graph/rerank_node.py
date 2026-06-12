"""Noeud de re-ranking — balancé par ticker pour les requêtes multi-entreprises."""
from __future__ import annotations

import asyncio
from typing import Any

from src.graph.prepare_node import extract_query_tickers
from src.graph.state import GraphState
from src.graph.tracing import traceable


def _candidate_ticker(agent: Any, index: int) -> str:
    if index >= len(agent.rag.doc_metadata):
        return ""
    return str(agent.rag.doc_metadata[index].get("ticker", "")).upper()


def _ticker_counts(metadatas: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metadata in metadatas:
        ticker = str(metadata.get("ticker", "UNKNOWN")).upper()
        counts[ticker] = counts.get(ticker, 0) + 1
    return counts


async def _balanced_rerank_indices(agent: Any, state: GraphState, candidates: list[int]) -> list[int]:
    target_tickers = [str(t).upper() for t in (state.get("target_tickers") or []) if str(t).strip()]
    for ticker in extract_query_tickers(state.get("normalized_query", "")):
        if ticker not in target_tickers:
            target_tickers.append(ticker)
    target_tickers = list(dict.fromkeys(target_tickers))
    if len(target_tickers) <= 1:
        return await asyncio.to_thread(agent.rag._rerank, state["normalized_query"], candidates, top_k=agent.max_context_chunks)
    groups = {ticker: [idx for idx in candidates if _candidate_ticker(agent, idx) == ticker] for ticker in target_tickers}
    groups = {t: i for t, i in groups.items() if i}
    if len(groups) <= 1:
        return await asyncio.to_thread(agent.rag._rerank, state["normalized_query"], candidates, top_k=agent.max_context_chunks)
    per_ticker_budget = max(1, agent.max_context_chunks // len(groups))
    selected = []
    for ticker in target_tickers:
        ticker_candidates = groups.get(ticker, [])
        if not ticker_candidates:
            continue
        selected.extend(await asyncio.to_thread(agent.rag._rerank, state["normalized_query"], ticker_candidates, top_k=per_ticker_budget))
    selected = list(dict.fromkeys(selected))
    remaining_budget = max(0, agent.max_context_chunks - len(selected))
    if remaining_budget:
        selected_set = set(selected)
        remaining_candidates = [idx for idx in candidates if idx not in selected_set]
        selected.extend(await asyncio.to_thread(agent.rag._rerank, state["normalized_query"], remaining_candidates, top_k=remaining_budget))
    return selected[: agent.max_context_chunks]


@traceable(name="rerank_node")
async def rerank_node(agent: Any, state: GraphState) -> GraphState:
    candidates = state.get("candidate_indices", [])
    if not candidates:
        return {"final_chunks": [], "final_metadatas": []}
    top_indices = await _balanced_rerank_indices(agent, state, candidates)
    final_chunks = [agent.rag.documents[idx] for idx in top_indices]
    final_meta = [agent.rag.doc_metadata[idx] for idx in top_indices]
    stats = state.get("stats", {})
    stats["estimated_context_tokens"] = agent.rag.count_context_tokens(final_chunks)
    stats.update({"rerank_final_ticker_counts": _ticker_counts(final_meta), "rerank_final_count": len(top_indices)})
    return {"final_chunks": final_chunks, "final_metadatas": final_meta, "stats": stats}
