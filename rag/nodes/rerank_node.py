from __future__ import annotations

from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="rerank_node")
def rerank_node(agent: Any, state: GraphState) -> GraphState:
    candidates = state.get("candidate_indices", [])
    if not candidates:
        return {"final_chunks": [], "final_metadatas": []}

    top_indices = agent.rag._rerank(state["normalized_query"], candidates, top_k=agent.max_context_chunks)
    final_chunks = [agent.rag.documents[idx] for idx in top_indices]
    final_meta = [agent.rag.doc_metadata[idx] for idx in top_indices]
    return {
        "final_chunks": final_chunks,
        "final_metadatas": final_meta,
    }
