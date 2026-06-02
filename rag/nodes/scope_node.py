from __future__ import annotations

import json
import re
from typing import Any

from rag.nodes.prompt_context import get_known_tickers
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


ALLOWED_DOC_TYPES = ["10-K", "10-Q", "8-K", "EARNINGS_CALL"]
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


def _normalize_doc_types(raw: Any) -> list[str]:
    if isinstance(raw, str):
        candidates = re.split(r"[,\s]+", raw.strip())
    elif isinstance(raw, list):
        candidates = [str(item).strip() for item in raw]
    else:
        candidates = []

    normalized: list[str] = []
    for item in candidates:
        up = item.upper().replace("_", "-")
        if up in {"10K", "10-K"}:
            value = "10-K"
        elif up in {"10Q", "10-Q"}:
            value = "10-Q"
        elif up in {"8K", "8-K"}:
            value = "8-K"
        elif up in {"EARNINGSCALL", "EARNINGS-CALL", "EARNINGS_CALL", "TRANSCRIPT"}:
            value = "EARNINGS_CALL"
        else:
            continue
        if value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_tickers(raw: Any, known_tickers: list[str], max_items: int = 4) -> list[str]:
    if isinstance(raw, str):
        candidates = re.split(r"[,\s]+", raw.strip())
    elif isinstance(raw, list):
        candidates = [str(item).strip() for item in raw]
    else:
        candidates = []

    known_set = set(known_tickers)
    normalized: list[str] = []
    for item in candidates:
        ticker = item.upper()
        if ticker in known_set and ticker not in normalized:
            normalized.append(ticker)
        if len(normalized) >= max_items:
            break
    return normalized


def _extract_company_mentions(query: str, known_tickers: list[str], max_items: int = 4) -> list[str]:
    q = query.lower()
    known_set = set(known_tickers)
    mentions: list[str] = []

    for token in re.findall(r"\b[A-Za-z]{2,5}\b", query):
        up = token.upper()
        if up in known_set and up not in mentions:
            mentions.append(up)
        if len(mentions) >= max_items:
            return mentions

    for alias, ticker in COMPANY_ALIASES.items():
        if alias in q and ticker in known_set and ticker not in mentions:
            mentions.append(ticker)
        if len(mentions) >= max_items:
            break
    return mentions


def _heuristic_scope(agent: Any, state: GraphState) -> tuple[list[str], list[str], str]:
    query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    known_tickers = get_known_tickers(agent, max_items=25)

    tickers: list[str] = []
    if metadata_filter.get("ticker"):
        tickers.append(metadata_filter["ticker"].upper())
    for up in _extract_company_mentions(query, known_tickers, max_items=4):
        if up not in tickers:
            tickers.append(up)
    tickers = tickers[:4]

    q = query.lower()
    doc_scope: list[str] = []
    if any(token in q for token in ["recent", "récent", "news", "annonce", "event", "catalyseur", "catalyst"]):
        doc_scope = ["8-K", "10-Q", "10-K"]
    elif any(token in q for token in ["trimestre", "quarter", "q1", "q2", "q3", "q4"]):
        doc_scope = ["10-Q", "8-K", "10-K"]
    elif any(token in q for token in ["transcript", "earnings call", "conference call", "call"]):
        doc_scope = ["EARNINGS_CALL", "10-Q", "10-K"]
    else:
        doc_scope = ["10-K", "10-Q", "8-K", "EARNINGS_CALL"]

    if not tickers and "semi" in q:
        for fallback in ["NVDA", "AMD", "INTC"]:
            if fallback in known_tickers and fallback not in tickers:
                tickers.append(fallback)

    return tickers, doc_scope, "heuristic_default"


def _llm_scope_decision(agent: Any, state: GraphState) -> tuple[list[str], list[str], str, str]:
    query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    known_tickers = get_known_tickers(agent, max_items=25)

    heuristic_tickers, heuristic_docs, heuristic_reason = _heuristic_scope(agent, state)
    if not query.strip():
        return heuristic_tickers, heuristic_docs, "empty_query", "rule"

    prompt = (
        "Tu es un routeur de scope pour un agent RAG finance.\n"
        "Choisis les entreprises et types de documents SEC/transcripts a prioriser pour le retrieval.\n"
        "Retourne STRICTEMENT un objet JSON sans texte autour.\n"
        'Schema: {"target_tickers": string[], "doc_type_priority": string[], "reason": string}\n'
        "Doc types autorises: 10-K, 10-Q, 8-K, EARNINGS_CALL.\n"
        "N'utilise QUE des tickers presents dans l'univers fourni.\n\n"
        f"Question: {query}\n"
        f"Metadata detectee: {metadata_filter}\n"
        f"Univers tickers disponible: {known_tickers}\n"
    )

    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=180)
        parsed = _extract_first_json_object(raw)
        if parsed is not None:
            tickers = _normalize_tickers(parsed.get("target_tickers"), known_tickers, max_items=4)
            doc_types = _normalize_doc_types(parsed.get("doc_type_priority"))
            if not doc_types:
                doc_types = heuristic_docs
            reason = str(parsed.get("reason", "llm_scope")).strip() or "llm_scope"
            explicit_mentions = _extract_company_mentions(query, known_tickers, max_items=4)
            for ticker in explicit_mentions:
                if ticker not in tickers:
                    tickers.append(ticker)
            tickers = tickers[:4]
            return tickers, doc_types, reason, "llm"
    except Exception:
        pass

    fallback_reason = "fallback_heuristic_invalid_llm_output"
    if raw.strip():
        fallback_reason = "fallback_heuristic_parse_error"
    return heuristic_tickers, heuristic_docs, f"{heuristic_reason}|{fallback_reason}", "heuristic"


@traceable(name="query_scope_node")
def query_scope_node(agent: Any, state: GraphState) -> GraphState:
    tickers, doc_types, reason, source = _llm_scope_decision(agent, state)
    stats = state.get("stats", {})
    stats.update(
        {
            "scope_source": source,
            "scope_reason": reason,
            "scope_tickers": tickers,
            "scope_doc_types": doc_types,
        }
    )
    return {
        "target_tickers": tickers,
        "doc_type_priority": [dt for dt in doc_types if dt in ALLOWED_DOC_TYPES],
        "stats": stats,
    }
