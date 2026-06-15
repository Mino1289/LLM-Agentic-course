from __future__ import annotations

import json
import re
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.prompts import INTENT_ROUTER_PROMPT
from src.orchestration.progress import emit_agent_progress
from src.orchestration.state import HubSpokeState


def route_after_intent_router(state: HubSpokeState) -> str:
    route = state.get("intent_route", "simple")
    return "simple_agent" if route == "simple" else "pm_plan"


@traceable(name="intent_router")
async def intent_router_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    query = (state.get("normalized_query") or state.get("query") or "").strip()
    stats = dict(state.get("stats") or {})

    if not query:
        return {"intent_route": "simple", "answer": "Peux-tu préciser ta question ?", "stats": stats}

    query_lower = query.lower()

    # --- Action (complex) keywords: these need the full Hub-and-Spoke pipeline ---
    action_keywords = [
        "achète", "acheter", "achat", "acheté", "achete", "action achet",
        "buy", "investi", "investis", "investir", "placement",
        "vends", "vendre", "vend", "sell", "vente", "trade", "order", "ordre",
        "rebalance", "rebalancer", "alloue", "allouer", "allocation",
        "place un ordre", "soumet un ordre", "exécute", "exécuter", "execute",
        "close position", "liquid", "couvre", "couverture", "hedge",
        "utilise mon", "prendre position",
    ]
    if any(kw in query_lower for kw in action_keywords):
        stats.update({"intent_route": "complex", "intent_reason": "action_keyword"})
        emit_agent_progress(
            "Intent Router",
            "completed",
            "Route: Hub-and-Spoke (multi-agents)",
        )
        return {"intent_route": "complex", "stats": stats}

    # --- Info (simple) keywords: these can be answered by the Phase 2 agent directly ---
    info_keywords = [
        "prix", "price", "cours", "cotati", "valeur", "combien coûte",
        "portefeuille", "portfolio", "compte", "account", "positions",
        "mon portefeuille", "mon compte", "mes positions", "mes actions",
        "buying power", "equity", "pnl", "solde", "balance",
        "rendement", "performance", "compar",
        "new", "actualité", "article", "news",
        "document", "filings", "sec", "10-k", "10-q", "8-k", "20-f",
        "rapport", "report", "risque", "risk", "section",
    ]
    if any(kw in query_lower for kw in info_keywords):
        stats.update({"intent_route": "simple", "intent_reason": "info_keyword"})
        emit_agent_progress(
            "Intent Router",
            "completed",
            "Route: Agent simple (requête directe)",
        )
        return {"intent_route": "simple", "stats": stats}

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
            stats.update({"intent_route": route, "intent_reason": reason})
            return {"intent_route": route, "stats": stats}
    except Exception:
        pass

    # En cas de doute → simple (info) plutôt que complex (trading)
    stats.update({"intent_route": "simple", "intent_reason": "fallback_to_simple"})
    return {"intent_route": "simple", "stats": stats}


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
