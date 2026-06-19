from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.prompts import INTENT_ROUTER_PROMPT
from src.orchestration.progress import emit_agent_progress
from src.orchestration.state import HubSpokeState
from src.orchestration.tool_domains import (
    detect_tool_domains,
    resolve_route_from_domains,
)

_LOGGER = logging.getLogger("src.orchestration.intent_router")


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
        return {
            "intent_route": "simple",
            "answer": "Peux-tu préciser ta question ?",
            "stats": stats,
        }

    domains = detect_tool_domains(query)
    resolved = resolve_route_from_domains(domains)
    if resolved is not None:
        route, reason = resolved
        stats.update(
            {
                "intent_route": route,
                "intent_reason": reason,
                "tool_domains": sorted(domains),
            }
        )
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
        # ainvoke_with_tools_stream est un async generator : on itère dessus,
        # on ne l'await pas (un await levait une TypeError silencieusement
        # rattrapée => le classifieur LLM ne tournait jamais).
        raw = ""
        async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
            messages,
            tools=None,
            temperature=0.0,
            max_tokens=120,
        ):
            if chunk.delta:
                raw += chunk.delta
        parsed = _extract_first_json_object(raw)
        route = str(parsed.get("route", "")).strip().lower() if parsed else ""
        if route in ("simple", "complex"):
            reason = str(parsed.get("reason", "llm_classifier")).strip()
            # Le LLM tranche is_trade pour les cas ambigus (les mots-clés trade
            # ne couvrent que les ordres explicites). Repli sur le keyword si
            # absent.
            llm_trade = _coerce_bool(parsed.get("is_trade"))
            trade_requested = llm_trade if llm_trade is not None else "trade" in domains
            # Un ordre implique forcément le chemin complexe (PM -> compliance
            # -> approbation humaine).
            if trade_requested and route == "simple":
                route = "complex"
            stats.update(
                {
                    "intent_route": route,
                    "intent_reason": reason,
                    "tool_domains": sorted(domains),
                }
            )
            _emit_route_progress(route)
            return {
                "intent_route": route,
                "trade_requested": trade_requested,
                "stats": stats,
            }
    except Exception:
        _LOGGER.warning("Intent classifier LLM failed; falling back to simple route.", exc_info=True)

    stats.update(
        {
            "intent_route": "simple",
            "intent_reason": "fallback_to_simple",
            "tool_domains": sorted(domains),
        }
    )
    _emit_route_progress("simple")
    return {
        "intent_route": "simple",
        "trade_requested": "trade" in domains,
        "stats": stats,
    }


def _coerce_bool(value: Any) -> bool | None:
    """Interprète is_trade renvoyé par le LLM (true/false, "true"/"yes"/1...).

    Renvoie None si la valeur est absente/inintelligible, pour permettre un
    repli sur la détection par mots-clés.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "oui", "1", "trade"}:
        return True
    if text in {"false", "no", "non", "0", "analysis", "analyse"}:
        return False
    return None


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
