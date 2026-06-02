from __future__ import annotations

import json
import re
from typing import Any

from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


COMPANY_ALIASES = {
    "nvidia": "NVDA",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "intel": "INTC",
    "palantir": "PLTR",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "broadcom": "AVGO",
    "oracle": "ORCL",
}


def _extract_company_mentions(query: str, universe_hint: str) -> list[str]:
    q = query.lower()
    mentions: list[str] = []

    known_tickers = [t.strip().upper() for t in universe_hint.split(",") if t.strip()]
    for token in re.findall(r"\b[A-Za-z]{2,5}\b", query):
        up = token.upper()
        if up in known_tickers and up not in mentions:
            mentions.append(up)

    for alias, ticker in COMPANY_ALIASES.items():
        if alias in q and ticker in known_tickers and ticker not in mentions:
            mentions.append(ticker)
    return mentions


def _format_recent_dialogue(messages: list[dict[str, str]], keep_last: int = 8) -> str:
    if not messages:
        return "Aucun historique."
    selected = messages[-keep_last:]
    lines = []
    for m in selected:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _is_followup_reference_query(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    reference_markers = [
        "dedans",
        "la-dessus",
        "là-dessus",
        "ce cas",
        "cette boite",
        "cette boîte",
        "cette entreprise",
        "celle-ci",
        "celui-ci",
        "ça",
        "ca",
    ]
    return any(marker in q for marker in reference_markers)


def _resolve_recent_company_from_messages(messages: list[dict[str, str]], universe_hint: str) -> str:
    known_tickers = [t.strip().upper() for t in universe_hint.split(",") if t.strip()]
    # Search recent user+assistant turns to resolve references like "dedans".
    for msg in reversed(messages[-10:]):
        content = msg.get("content", "")
        mentions = _extract_company_mentions(content, universe_hint)
        for mention in mentions:
            if mention in known_tickers:
                return mention
    return ""


def _is_investment_or_comparison_query(query: str) -> bool:
    q = query.lower()
    signals = [
        "investir",
        "investissement",
        "acheter",
        "buy",
        "sell",
        "vs",
        "versus",
        "comparer",
        "compare",
        "meilleur",
        "better",
        "overweight",
        "underweight",
    ]
    return any(token in q for token in signals)


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
        "investir",
        "investissement",
        "acheter",
        "comparer",
        "compare",
        "vs",
    ]
    has_domain_signal = any(token in q for token in domain_keywords)
    return len(q) < 20 or not has_domain_signal


def _is_general_chat_query(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    greetings = [
        "bonjour",
        "salut",
        "hello",
        "hey",
        "bonsoir",
        "coucou",
        "comment vas",
        "ca va",
        "ça va",
        "merci",
    ]
    finance_terms = [
        "action",
        "stock",
        "prix",
        "cours",
        "earnings",
        "10-k",
        "8-k",
        "sec",
        "guidance",
        "marge",
        "risque",
        "catalyseur",
        "ticker",
        "valorisation",
        "revenu",
    ]
    has_greeting = any(token in q for token in greetings)
    has_finance_signal = any(token in q for token in finance_terms)
    return has_greeting and not has_finance_signal


def _is_obviously_off_topic_request(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    off_topic_patterns = [
        "code python",
        "python",
        "javascript",
        "java",
        "c++",
        "factorielle",
        "algo",
        "recette",
        "cuisine",
        "poeme",
        "poème",
        "traduis",
        "traduction",
        "blague",
        "devine",
    ]
    finance_terms = [
        "action",
        "stock",
        "prix",
        "cours",
        "earnings",
        "10-k",
        "8-k",
        "sec",
        "guidance",
        "marge",
        "risque",
        "catalyseur",
        "ticker",
        "valorisation",
        "revenu",
        "entreprise",
    ]
    has_off_topic = any(token in q for token in off_topic_patterns)
    has_finance_signal = any(token in q for token in finance_terms)
    return has_off_topic and not has_finance_signal


def _is_coverage_question(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    patterns = [
        "quelles entreprises",
        "quelle entreprise",
        "entreprises couvert",
        "entreprises disponibles",
        "univers couvert",
        "tu connais quelles entreprises",
        "sur quelles entreprises",
        "which companies",
        "covered companies",
    ]
    return any(p in q for p in patterns)


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


def llm_intent_scope_decision(agent: Any, state: GraphState) -> tuple[str, str, str]:
    query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    messages = state.get("messages", [])

    if not query.strip():
        return "clarify", "empty_query", "rule"

    universe_hint = format_universe_hint(agent)
    recent_dialogue = _format_recent_dialogue(messages, keep_last=8)
    resolved_followup_ticker = ""
    if _is_followup_reference_query(query):
        resolved_followup_ticker = _resolve_recent_company_from_messages(messages, universe_hint)
    company_mentions = _extract_company_mentions(query, universe_hint)
    heuristic_route = "continue"
    if _is_coverage_question(query):
        heuristic_route = "coverage_info"
    elif _is_obviously_off_topic_request(query):
        heuristic_route = "reject_offtopic"
    elif _is_general_chat_query(query):
        heuristic_route = "general_chat"
    elif company_mentions and _is_investment_or_comparison_query(query):
        heuristic_route = "continue"
    elif resolved_followup_ticker:
        heuristic_route = "continue"
    elif is_ambiguous_general_query(query, metadata_filter):
        heuristic_route = "clarify"

    recent_user_turns = [
        m.get("content", "").strip()
        for m in messages[-4:]
        if m.get("role") == "user" and m.get("content", "").strip()
    ]
    prompt = (
        "Tu es un routeur d'intention pour un agent RAG finance.\n"
        "Choisis la route la plus adaptee pour cette requete.\n"
        "Retourne STRICTEMENT un objet JSON, sans texte autour.\n"
        'Schema: {"route": "continue|clarify|general_chat|reject_offtopic|coverage_info", "reason": string}\n'
        "- continue: question finance exploitable avec retrieval/reponse finance.\n"
        "- clarify: question finance mais scope insuffisant. IMPORTANT: "
        "si au moins une entreprise identifiable est presente (ticker ou nom), "
        "privilegie continue meme sans periode explicite. "
        "Pareil si la question est un suivi referentiel (ex: 'c'est risque dedans ?') "
        "et que le contexte recent permet d'identifier l'entreprise.\n"
        "- coverage_info: question sur les entreprises/tickers couverts par la base.\n"
        "- general_chat: question hors finance mais repondable en conversation generale "
        "(salutation, small talk, question generale).\n\n"
        "- reject_offtopic: demande hors perimetre finance a bloquer poliment "
        "(ex: generation de code, recette, tache dev generale).\n\n"
        f"Univers couvert (tickers disponibles): {universe_hint}\n"
        f"Question: {query}\n"
        f"Metadata detectee: {metadata_filter}\n"
        f"Entreprises detectees (heuristique): {company_mentions}\n"
        f"Entreprise resolue depuis le contexte (si follow-up): {resolved_followup_ticker or 'none'}\n"
        f"Dialogue recent:\n{recent_dialogue}\n"
        f"Derniers tours utilisateur: {recent_user_turns}\n"
    )
    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=120)
        parsed = _extract_first_json_object(raw)
        route = str(parsed.get("route", "")).strip().lower() if parsed is not None else ""
        if route in {"continue", "clarify", "general_chat", "reject_offtopic", "coverage_info"}:
            reason = str(parsed.get("reason", "llm_decision")).strip() or "llm_decision"
            return route, reason, "llm"
    except Exception:
        pass

    fallback_reason = "fallback_heuristic_invalid_llm_output"
    if raw.strip():
        fallback_reason = "fallback_heuristic_parse_error"
    return heuristic_route, fallback_reason, "heuristic"


def route_after_intent_node(state: GraphState) -> str:
    route = state.get("intent_route", "")
    if route in {"continue", "clarify", "general_chat", "reject_offtopic", "coverage_info"}:
        return route
    return "clarify" if state.get("ambiguous_query", False) else "continue"


@traceable(name="intent_scope_node")
def intent_scope_node(_agent: Any, state: GraphState) -> GraphState:
    intent_route, reason, source = llm_intent_scope_decision(_agent, state)
    ambiguous_query = intent_route == "clarify"
    general_chat = intent_route == "general_chat"
    off_topic_blocked = intent_route == "reject_offtopic"
    metadata_filter = dict(state.get("metadata_filter", {}))
    # Deterministic safety net for follow-up references like "dedans".
    if not metadata_filter.get("ticker") and _is_followup_reference_query(state.get("normalized_query", "")):
        universe_hint = format_universe_hint(_agent)
        resolved_ticker = _resolve_recent_company_from_messages(state.get("messages", []), universe_hint)
        if resolved_ticker:
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
