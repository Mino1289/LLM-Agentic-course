from __future__ import annotations

import json
from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


def parse_query_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass
    lines = [ln.strip("-• \t") for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def decompose_query(agent: Any, query: str) -> list[str]:
    prompt = (
        "Decompose la requete finance suivante en sous-requetes ciblees pour retrieval RAG.\n"
        f"Requete: {query}\n\n"
        f"Rends STRICTEMENT un JSON array de {agent.decompose_query_count} a "
        f"{agent.decompose_query_count + 2} chaines courtes, sans autre texte."
    )
    raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=350)
    parsed = parse_query_list(raw)
    if not parsed:
        parsed = [query]
    if query not in parsed:
        parsed.insert(0, query)
    return parsed[: agent.decompose_query_count + 2]


@traceable(name="decompose_query_node")
def decompose_query_node(agent: Any, state: GraphState) -> GraphState:
    normalized_query = state["normalized_query"]
    metadata_filter = state.get("metadata_filter", {})
    if metadata_filter.get("ticker") and metadata_filter.get("year"):
        decomposed = [normalized_query]
    else:
        decomposed = decompose_query(agent, normalized_query)
    return {"decomposed_queries": decomposed}
