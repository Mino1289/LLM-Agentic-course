"""Décomposition de requête en sous-requêtes pour améliorer le rappel RAG."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.graph.prompt_context import format_universe_hint
from src.graph.state import GraphState
from src.graph.tracing import traceable


def parse_query_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip("-• \t") for ln in text.splitlines()]
    return [
        ln for ln in lines
        if ln and ln not in {"```", "```json", "[", "]"} and not ln.startswith("```")
    ]


async def decompose_query(agent: Any, query: str) -> list[str]:
    universe_hint = format_universe_hint(agent)
    prompt = (
        "You are a query decomposition planner for a finance RAG system.\n"
        "Break the user question into focused sub-queries that improve retrieval coverage.\n"
        "Use only the covered companies when adding ticker-specific queries.\n"
        f"Covered companies (tickers): {universe_hint}\n"
        f"User question: {query}\n\n"
        "Return STRICT JSON only (no prose):\n"
        f"- a JSON array with {agent.decompose_query_count} to {agent.decompose_query_count + 2} short strings."
    )
    raw = await asyncio.to_thread(
        agent.rag.provider.generate, prompt, temperature=0.0, max_tokens=350,
    )
    parsed = parse_query_list(raw)
    if not parsed:
        parsed = [query]
    if query not in parsed:
        parsed.insert(0, query)
    return parsed[: agent.decompose_query_count + 2]


@traceable(name="decompose_query_node")
async def decompose_query_node(agent: Any, state: GraphState) -> GraphState:
    normalized_query = state["normalized_query"]
    metadata_filter = state.get("metadata_filter", {})
    if metadata_filter.get("ticker") and metadata_filter.get("year"):
        decomposed = [normalized_query]
    else:
        decomposed = await decompose_query(agent, normalized_query)
    return {"decomposed_queries": decomposed}
