from __future__ import annotations

from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="multi_retrieve_node")
def multi_retrieve_node(agent: Any, state: GraphState) -> GraphState:
    queries = state.get("decomposed_queries") or [state["normalized_query"]]
    metadata_filter = state.get("metadata_filter") or {}
    all_indices: list[int] = []
    for query in queries:
        retrieval = agent.rag.retrieve(
            query,
            search_mode="vector",
            use_reranking=False,
            metadata_filter=metadata_filter or None,
            top_k=24,
            candidate_pool=24,
        )
        all_indices.extend(retrieval.chunk_indices)

        if not retrieval.chunk_indices and metadata_filter:
            relaxed_filter = dict(metadata_filter)
            relaxed_filter.pop("year", None)
            if relaxed_filter:
                relaxed_retrieval = agent.rag.retrieve(
                    query,
                    search_mode="vector",
                    use_reranking=False,
                    metadata_filter=relaxed_filter,
                    top_k=24,
                    candidate_pool=24,
                )
                all_indices.extend(relaxed_retrieval.chunk_indices)

        if not retrieval.chunk_indices and metadata_filter:
            broad_retrieval = agent.rag.retrieve(
                query,
                search_mode="vector",
                use_reranking=False,
                metadata_filter=None,
                top_k=24,
                candidate_pool=24,
            )
            all_indices.extend(broad_retrieval.chunk_indices)

    dedup_indices = agent.rag._deduplicate_indices(all_indices)
    stats = state.get("stats", {})
    stats.update(
        {
            "decomposed_query_count": len(queries),
            "retrieval_candidate_count": len(dedup_indices),
        }
    )
    return {
        "candidate_indices": dedup_indices,
        "stats": stats,
    }
