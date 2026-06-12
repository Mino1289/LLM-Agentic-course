"""Noeud de garde — classification d'intention avant routage vers l'agent."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.graph.prompt_context import format_universe_hint
from src.graph.state import GraphState
from src.graph.tracing import traceable

_guard_cache: dict[str, tuple[str, str, str]] = {}
_MAX_CACHE_SIZE = 128

GUARD_SYSTEM_PROMPT = """You are an intent guard for a finance RAG assistant.
Return STRICT JSON only, with no surrounding text.

Schema:
{"route":"continue|clarify|coverage_info|reject_offtopic|general_chat","reason":"short reason"}

Routes:
- continue: the finance agent should handle the request using RAG/tools if useful.
- clarify: the request is finance-related but too vague to answer safely.
- coverage_info: the user asks what companies/tickers are covered.
- reject_offtopic: the request is clearly outside finance/RAG/market analysis.
- general_chat: harmless greeting or small talk that does not require tools.

Priority:
1) Prefer continue for finance, market, company, portfolio, report, SEC, risk, or price questions.
2) Use clarify only when a finance request lacks enough scope and no reasonable assumption is possible.
3) Use coverage_info only for explicit coverage/universe questions.
4) Use reject_offtopic only when the request is clearly non-finance.
"""

_VALID_ROUTES = {"continue", "clarify", "coverage_info", "reject_offtopic", "general_chat"}


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


def _format_recent_dialogue(messages: list[dict[str, str]], keep_last: int = 6) -> str:
    if not messages:
        return "Aucun historique."
    lines: list[str] = []
    for msg in messages[-keep_last:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "Aucun historique."


def _build_guard_prompt(agent: Any, state: GraphState, query: str) -> str:
    universe = format_universe_hint(agent, max_items=12)
    metadata_filter = json.dumps(state.get("metadata_filter") or {}, ensure_ascii=False)
    recent = _format_recent_dialogue(state.get("messages", []))
    memory_summary = state.get("memory_summary") or "Aucun résumé mémoire."
    memory_window = _format_recent_dialogue(state.get("memory_window", []), keep_last=6)
    return (
        f"Covered tickers: {universe}\n"
        f"Heuristic metadata: {metadata_filter}\n"
        f"Memory summary:\n{memory_summary}\n"
        f"Memory recent turns:\n{memory_window}\n"
        f"Recent dialogue:\n{recent}\n\n"
        f"User query:\n{query}"
    )


async def _llm_guard_decision(agent: Any, state: GraphState, query: str) -> tuple[str, str, str]:
    if cached := _guard_cache.get(query):
        return cached

    prompt = _build_guard_prompt(agent, state, query)
    raw = ""
    try:
        raw = await asyncio.to_thread(
            agent.rag.provider.generate,
            prompt,
            system_prompt=GUARD_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=180,
        )
        parsed = _extract_first_json_object(raw)
        route = str(parsed.get("route", "")).strip().lower() if parsed else ""
        if route in _VALID_ROUTES:
            reason = str(parsed.get("reason", "llm_guard")).strip() or "llm_guard"
            result = (route, reason, "llm")
            _guard_cache[query] = result
            if len(_guard_cache) > _MAX_CACHE_SIZE:
                _guard_cache.pop(next(iter(_guard_cache)))
            return result
        if raw.strip():
            result = ("continue", "fallback_parse_error", "fallback")
            _guard_cache[query] = result
            if len(_guard_cache) > _MAX_CACHE_SIZE:
                _guard_cache.pop(next(iter(_guard_cache)))
            return result
    except Exception:
        return "continue", "fallback_provider_error", "fallback"
    result = ("continue", "fallback_empty_response", "fallback")
    _guard_cache[query] = result
    if len(_guard_cache) > _MAX_CACHE_SIZE:
        _guard_cache.pop(next(iter(_guard_cache)))
    return result


def _answer_for_route(agent: Any, route: str, query: str) -> str:
    universe = format_universe_hint(agent, max_items=12)
    if route == "coverage_info":
        return f"Entreprises couvertes en mode test/debug : {universe}."
    if route == "reject_offtopic":
        return (
            "Je suis spécialisé en analyse financière avec RAG SEC, prix de marché, "
            f"validation et rapports. Pose une question sur : {universe}."
        )
    if route == "clarify":
        return (
            "Ta question est trop large ou manque de contexte. Précise l'entreprise, "
            f"la période ou le type d'analyse. Entreprises disponibles : {universe}."
        )
    if route == "general_chat":
        return (
            "Je peux t'aider sur l'analyse financière, les documents SEC, les prix, "
            f"la validation d'affirmations et les rapports pour : {universe}."
        )
    return ""


@traceable(name="guard_node")
async def guard_node(agent: Any, state: GraphState) -> GraphState:
    query = (state.get("normalized_query") or state.get("query") or "").strip()
    stats = dict(state.get("stats") or {})

    if not query:
        stats.update(
            {
                "guard_route": "clarify",
                "guard_source": "rule",
                "guard_reason": "empty_query",
            }
        )
        return {
            "answer": "Peux-tu préciser ta question finance (entreprise, période, ou type de document) ?",
            "tool_calls_pending": False,
            "stats": stats,
        }

    route, reason, source = await _llm_guard_decision(agent, state, query)
    stats.update(
        {
            "guard_route": route,
            "guard_source": source,
            "guard_reason": reason,
        }
    )

    answer = _answer_for_route(agent, route, query)
    if answer:
        return {
            "answer": answer,
            "tool_calls_pending": False,
            "stats": stats,
        }
    return {"stats": stats}


def route_after_guard(state: GraphState) -> str:
    if state.get("answer"):
        return "finalize"
    return "agent"
