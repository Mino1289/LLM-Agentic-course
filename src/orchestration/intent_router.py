from __future__ import annotations

import json
import re
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.prompts import INTENT_ROUTER_PROMPT
from src.orchestration.progress import emit_agent_progress
from src.orchestration.state import HubSpokeState
from src.orchestration.tool_domains import detect_tool_domains, resolve_route_from_domains


def route_after_intent_router(state: HubSpokeState) -> str:
    route = state.get("intent_route", "simple")
    return "simple_agent" if route == "simple" else "pm_plan"


def _emit_route_progress(route: str) -> None:
    if route == "complex":
        emit_agent_progress(
            "Intent Router",
            "completed",
            "Route: Hub-and-Spoke (multi-agents)",
        )
    else:
        emit_agent_progress(
            "Intent Router",
            "completed",
            "Route: Agent simple (requête directe)",
        )


@traceable(name="intent_router")
async def intent_router_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    query = (state.get("normalized_query") or state.get("query") or "").strip()
    stats = dict(state.get("stats") or {})

    if not query:
        return {"intent_route": "simple", "answer": "Peux-tu préciser ta question ?", "stats": stats}

    domains = detect_tool_domains(query)
    resolved = resolve_route_from_domains(domains)
    if resolved is not None:
        route, reason = resolved
        stats.update({
            "intent_route": route,
            "intent_reason": reason,
            "tool_domains": sorted(domains),
        })
        _emit_route_progress(route)
        return {
            "intent_route": route,
            "trade_requested": "trade" in domains,
            "stats": stats,
        }

    # --- LLM classifier for ambiguous queries ---
    try:
        messages = [
            {"role": "system", "content": INTENT_ROUTER_PROMPT},
            {"role": "user", "content": f"User query: {query}"},
        ]
        response = await agent.rag.provider.ainvoke_with_tools_stream(
            messages, tools=None, temperature=0.0, max_tokens=120,
        )
        raw = ""
        async for chunk in response:
            if chunk.delta:
                raw += chunk.delta
        parsed = _extract_first_json_object(raw)
        route = str(parsed.get("route", "")).strip().lower() if parsed else ""
        if route in ("simple", "complex"):
            reason = str(parsed.get("reason", "llm_classifier")).strip()
            stats.update({
                "intent_route": route,
                "intent_reason": reason,
                "tool_domains": sorted(domains),
            })
            _emit_route_progress(route)
            return {
                "intent_route": route,
                "trade_requested": "trade" in domains,
                "stats": stats,
            }
    except Exception:
        pass

    stats.update({
        "intent_route": "simple",
        "intent_reason": "fallback_to_simple",
        "tool_domains": sorted(domains),
    })
    _emit_route_progress("simple")
    return {
        "intent_route": "simple",
        "trade_requested": "trade" in domains,
        "stats": stats,
    }


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
