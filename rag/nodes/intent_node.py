from __future__ import annotations

from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


def is_ambiguous_general_query(query: str, metadata_filter: dict[str, str]) -> bool:
    if metadata_filter.get("ticker") or metadata_filter.get("year"):
        return False
    q = query.lower()
    domain_keywords = [
        "risque",
        "catalyseur",
        "croissance",
        "marge",
        "profit",
        "guidance",
        "valorisation",
        "concurrence",
        "opportunit",
        "secteur",
        "gpu",
        "ia",
        "semi",
        "entreprise",
    ]
    has_domain_signal = any(token in q for token in domain_keywords)
    return len(q) < 20 or not has_domain_signal


def route_after_intent_node(state: GraphState) -> str:
    return "clarify" if state.get("ambiguous_query", False) else "continue"


@traceable(name="intent_scope_node")
def intent_scope_node(_agent: Any, state: GraphState) -> GraphState:
    normalized_query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    ambiguous_query = is_ambiguous_general_query(normalized_query, metadata_filter)
    return {"ambiguous_query": ambiguous_query}


@traceable(name="clarify_node")
def clarify_node(_agent: Any, state: GraphState) -> GraphState:
    query = state.get("normalized_query", "")
    clarification_question = (
        "Ta question est encore large. Tu veux une analyse sur quelle entreprise "
        "ou groupe d'entreprises, et sur quelle periode (ex: 2024, 2023-2025) ?\n\n"
        f"Question recue: {query}"
    )
    stats = state.get("stats", {})
    stats.update(
        {
            "chunks_used": 0,
            "gc_applied": False,
            "estimated_context_tokens": 0,
            "clarification_requested": True,
        }
    )
    return {"clarification_question": clarification_question, "stats": stats}
