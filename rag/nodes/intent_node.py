from __future__ import annotations

import json
import re
from typing import Any

from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


def _format_recent_dialogue(messages: list[dict[str, str]], keep_last: int = 10) -> str:
    if not messages:
        return "Aucun historique."
    selected = messages[-keep_last:]
    lines = []
    for msg in selected:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_first_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_route(route: str) -> str:
    value = (route or "").strip().lower()
    if value in {"continue", "clarify", "general_chat", "reject_offtopic", "coverage_info"}:
        return value
    return ""


def _minimal_fallback_route(query: str) -> tuple[str, str]:
    q = query.lower().strip()
    if not q:
        return "clarify", "empty_query"
    if any(token in q for token in ["code", "python", "javascript", "java", "c++", "factorielle"]):
        return "reject_offtopic", "fallback_offtopic_code_request"
    if any(token in q for token in ["quelles entreprises", "sur quelles entreprises", "all companies"]):
        return "coverage_info", "fallback_coverage_request"
    if len(q) < 12:
        return "clarify", "fallback_too_short"
    return "continue", "fallback_continue"


def llm_intent_scope_decision(agent: Any, state: GraphState) -> tuple[str, str, str, str]:
    query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    messages = state.get("messages", [])

    if not query.strip():
        return "clarify", "empty_query", "rule", ""

    universe_hint = format_universe_hint(agent, max_items=20)
    recent_dialogue = _format_recent_dialogue(messages, keep_last=10)
    metadata_hint = json.dumps(metadata_filter, ensure_ascii=False)

    prompt = (
        "You are an intent router for a finance RAG assistant.\n"
        "Pick exactly ONE final route.\n"
        "Return STRICT JSON only, with no surrounding text.\n"
        'Schema: {"route":"continue|clarify|general_chat|reject_offtopic|coverage_info",'
        ' "reason":string, "resolved_ticker":string}\n\n'
        "Route definitions:\n"
        "- continue: finance question can be answered now.\n"
        "- clarify: finance scope is insufficient AND cannot be resolved from recent context.\n"
        "- general_chat: acceptable non-finance small talk.\n"
        "- reject_offtopic: out-of-scope request (e.g. coding task).\n"
        "- coverage_info: explicit question about covered companies.\n\n"
        "Priority rules:\n"
        "1) Clarify is the last resort.\n"
        "2) If the question is a referential follow-up (e.g. 'in that case', 'that one'), "
        "resolve the company from recent dialogue and set resolved_ticker.\n"
        "3) If a company can be resolved, prefer continue.\n"
        "4) If this is an investment/comparison question with identifiable companies, prefer continue.\n"
        "5) Use coverage_info only for explicit coverage questions.\n"
        "6) resolved_ticker must be empty or a ticker in the covered universe.\n\n"
        f"Covered companies (tickers): {universe_hint}\n"
        f"Detected metadata (heuristic): {metadata_hint}\n"
        f"User question: {query}\n"
        f"Recent dialogue:\n{recent_dialogue}\n"
    )

    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=220)
        parsed = _extract_first_json_object(raw)
        route = _normalize_route(str(parsed.get("route", ""))) if parsed else ""
        if route:
            reason = str(parsed.get("reason", "llm_decision")).strip() or "llm_decision"
            resolved_ticker = str(parsed.get("resolved_ticker", "")).strip().upper()
            allowed_tickers = {t.strip().upper() for t in universe_hint.split(",") if t.strip()}
            if resolved_ticker and resolved_ticker not in allowed_tickers:
                resolved_ticker = ""
            return route, reason, "llm", resolved_ticker
    except Exception:
        pass

    fallback_route, fallback_reason = _minimal_fallback_route(query)
    if raw.strip():
        fallback_reason = f"{fallback_reason}|fallback_parse_error"
    return fallback_route, fallback_reason, "heuristic", ""


def route_after_intent_node(state: GraphState) -> str:
    route = state.get("intent_route", "")
    if route in {"continue", "clarify", "general_chat", "reject_offtopic", "coverage_info"}:
        return route
    return "clarify" if state.get("ambiguous_query", False) else "continue"


@traceable(name="intent_scope_node")
def intent_scope_node(_agent: Any, state: GraphState) -> GraphState:
    intent_route, reason, source, resolved_ticker = llm_intent_scope_decision(_agent, state)
    ambiguous_query = intent_route == "clarify"
    general_chat = intent_route == "general_chat"
    off_topic_blocked = intent_route == "reject_offtopic"
    metadata_filter = dict(state.get("metadata_filter", {}))
    if not metadata_filter.get("ticker") and resolved_ticker:
        metadata_filter["ticker"] = resolved_ticker
        if intent_route == "clarify":
            intent_route = "continue"
            ambiguous_query = False
            reason = f"{reason}|followup_resolved:{resolved_ticker}"

    stats = state.get("stats", {})
    stats.update(
        {
            "intent_scope_source": source,
            "intent_scope_reason": reason,
            "intent_route": intent_route,
            "ambiguous_query": ambiguous_query,
            "general_chat": general_chat,
            "off_topic_blocked": off_topic_blocked,
            "resolved_ticker": resolved_ticker,
        }
    )
    return {
        "intent_route": intent_route,
        "ambiguous_query": ambiguous_query,
        "general_chat": general_chat,
        "off_topic_blocked": off_topic_blocked,
        "metadata_filter": metadata_filter,
        "stats": stats,
    }


@traceable(name="clarify_node")
def clarify_node(_agent: Any, state: GraphState) -> GraphState:
    query = state.get("normalized_query", "")
    universe_hint = format_universe_hint(_agent, max_items=10)
    clarification_question = (
        "Ta question est encore large. Tu veux une analyse sur quelle entreprise "
        "ou groupe d'entreprises, et sur quelle periode (ex: 2024, 2023-2025) ?\n"
        f"Exemples disponibles dans la base: {universe_hint}\n\n"
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
