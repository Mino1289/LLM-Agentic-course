from __future__ import annotations

import os

from rag.langsmith_env import ensure_langsmith_env
from rag.hybrid_rag import HybridRAG
from rag.langgraph_flow import FinanceLangGraphAgent

ensure_langsmith_env()


def build_graph():
    rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
    # Studio should inspect the runnable graph, not trigger a full re-embed.
    rag.load_and_index_data(max_new_embeddings=0)
    agent = FinanceLangGraphAgent(
        rag=rag,
        memory_window_size=int(os.getenv("MEMORY_WINDOW_SIZE", "6")),
        summarize_every_n_turns=int(os.getenv("SUMMARIZE_EVERY_N_TURNS", "6")),
        max_context_chunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "8")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "3500")),
        decompose_query_count=int(os.getenv("QUERY_DECOMPOSE_COUNT", "2")),
        price_max_days=int(os.getenv("PRICE_MAX_DAYS", "180")),
        price_max_points=int(os.getenv("PRICE_MAX_POINTS", "40")),
        price_max_tickers=int(os.getenv("PRICE_MAX_TICKERS", "3")),
        price_default_days=int(os.getenv("PRICE_DEFAULT_DAYS", "90")),
        price_max_attempts=int(os.getenv("PRICE_MAX_ATTEMPTS", "2")),
        max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    )
    return agent.graph


graph = build_graph()
