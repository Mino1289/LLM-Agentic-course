from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


TRACKED_TICKERS = ["NVDA", "INTC", "AMD", "PLTR", "GOOGL", "META", "AMZN", "MSFT", "AVGO", "ORCL"]


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


def _normalize_tickers(raw_tickers: Any, max_items: int) -> list[str]:
    if isinstance(raw_tickers, str):
        candidates = re.split(r"[,\s]+", raw_tickers.strip())
    elif isinstance(raw_tickers, list):
        candidates = [str(t).strip() for t in raw_tickers]
    else:
        candidates = []

    deduped: list[str] = []
    for token in candidates:
        t = token.upper()
        if t in TRACKED_TICKERS and t not in deduped:
            deduped.append(t)
    return deduped[:max_items]


def has_sufficient_price_context(price_context: str) -> bool:
    if not price_context:
        return False
    return "Fenetre prix:" in price_context and "points[" in price_context


def should_fetch_price_context(
    query: str,
    metadata_filter: dict[str, str],
    messages: list[dict[str, str]],
) -> bool:
    if os.getenv("PRICE_TOOL_ENABLED", "true").lower() in {"0", "false", "no"}:
        return False

    lower_q = query.lower()
    explicit_price_keywords = [
        "prix",
        "cours",
        "performance",
        "rendement",
        "drawdown",
        "volatilit",
        "return",
        "returns",
        "stock price",
        "chart",
        "graphique",
    ]
    contextual_finance_keywords = [
        "risque",
        "catalyseur",
        "these",
        "allocation",
        "compar",
        "opportunit",
        "investissement",
        "positionn",
        "court terme",
        "momentum",
        "sentiment",
    ]

    has_ticker = bool(metadata_filter.get("ticker")) or bool(re.search(r"\b[A-Z]{2,5}\b", query))
    explicit_price = any(k in lower_q for k in explicit_price_keywords)
    contextual_finance = any(k in lower_q for k in contextual_finance_keywords)

    if explicit_price:
        return True
    if has_ticker and contextual_finance:
        return True
    if "semiconduct" in lower_q or "semi-conduct" in lower_q or "semi conduct" in lower_q:
        return True

    if messages:
        last_user_turns = [m.get("content", "").lower() for m in messages[-3:] if m.get("role") == "user"]
        if any("compare" in t or "compar" in t for t in last_user_turns) and has_ticker:
            return True
    return False


def extract_tickers_for_price_tool(agent: Any, query: str, metadata_filter: dict[str, str]) -> list[str]:
    found = []
    if metadata_filter.get("ticker"):
        found.append(metadata_filter["ticker"].upper())

    for ticker in re.findall(r"\b[A-Z]{2,5}\b", query):
        candidate = ticker.upper()
        if candidate in TRACKED_TICKERS and candidate not in found:
            found.append(candidate)

    if not found:
        q = query.lower()
        if "semiconduct" in q or "semi-conduct" in q or "semi conduct" in q:
            found = ["NVDA", "AMD", "INTC"]

    return found[: agent.price_max_tickers]


def llm_tool_decision(
    agent: Any,
    query: str,
    metadata_filter: dict[str, str],
    messages: list[dict[str, str]],
    enough_price_context: bool,
    attempts: int,
    fallback_tickers: list[str],
) -> tuple[bool, str, list[str], str]:
    heuristic_decision = should_fetch_price_context(query, metadata_filter, messages)
    if enough_price_context:
        return False, "price_context_ready", fallback_tickers, "rule"

    recent_user_turns = [
        m.get("content", "").strip()
        for m in messages[-4:]
        if m.get("role") == "user" and m.get("content", "").strip()
    ]
    universe_hint = format_universe_hint(agent, max_items=14)
    prompt = (
        "Tu es un routeur d'outil pour assistant finance.\n"
        "Decide si l'outil prix de marche (series de prix) doit etre appele AVANT la reponse.\n"
        "Retourne STRICTEMENT un JSON object, sans texte autour.\n"
        'Schema: {"use_price_tool": bool, "reason": string, "tickers": string[]}\n'
        "Mets use_price_tool=true si les prix/perf/volatilite/momentum/comparaison peuvent materially "
        "ameliorer la reponse. Sinon false.\n"
        f"Question: {query}\n"
        f"Metadata detectee: {metadata_filter}\n"
        f"Derniers tours utilisateur: {recent_user_turns}\n"
        f"Contexte prix deja present: {enough_price_context}\n"
        f"Tentatives outil deja faites: {attempts}\n"
        f"Tickers candidats heuristiques: {fallback_tickers}\n"
        f"Tickers disponibles dans la base: {universe_hint}\n"
        f"Univers suivi: {TRACKED_TICKERS}\n"
    )
    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=180)
        parsed = _extract_first_json_object(raw)
        if parsed is not None and isinstance(parsed.get("use_price_tool"), bool):
            llm_tickers = _normalize_tickers(parsed.get("tickers"), agent.price_max_tickers)
            if not llm_tickers:
                llm_tickers = fallback_tickers
            reason = str(parsed.get("reason", "llm_decision")).strip() or "llm_decision"
            return parsed["use_price_tool"], reason, llm_tickers, "llm"
    except Exception:
        pass

    fallback_reason = "fallback_heuristic_invalid_llm_output"
    if raw.strip():
        fallback_reason = "fallback_heuristic_parse_error"
    return heuristic_decision, fallback_reason, fallback_tickers, "heuristic"


def extract_price_date_window(agent: Any, query: str) -> tuple[str, str]:
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=agent.price_default_days)
    start_date = default_start
    end_date = today

    explicit_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    if len(explicit_dates) >= 2:
        try:
            d1 = datetime.strptime(explicit_dates[0], "%Y-%m-%d").date()
            d2 = datetime.strptime(explicit_dates[1], "%Y-%m-%d").date()
            start_date, end_date = (d1, d2) if d1 <= d2 else (d2, d1)
        except Exception:
            pass
    elif re.search(r"\b20\d{2}\b", query):
        year = int(re.search(r"\b(20\d{2})\b", query).group(1))
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()

    rel_match = re.search(r"\b(\d+)\s*(jour|jours|day|days|mois|month|months|an|ans|year|years)\b", query.lower())
    if rel_match:
        value = int(rel_match.group(1))
        unit = rel_match.group(2)
        if "jour" in unit or "day" in unit:
            delta_days = value
        elif "mois" in unit or "month" in unit:
            delta_days = value * 30
        else:
            delta_days = value * 365
        delta_days = min(delta_days, agent.price_max_days)
        start_date = today - timedelta(days=delta_days)
        end_date = today

    if (end_date - start_date).days > agent.price_max_days:
        start_date = end_date - timedelta(days=agent.price_max_days)

    return start_date.isoformat(), end_date.isoformat()


def widen_price_window(agent: Any, start_date: str, end_date: str) -> tuple[str, str]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except Exception:
        return start_date, end_date

    widened_start = start - timedelta(days=agent.price_default_days)
    max_start = end - timedelta(days=agent.price_max_days)
    if widened_start < max_start:
        widened_start = max_start
    return widened_start.isoformat(), end.isoformat()


def format_price_index(idx: Any) -> str:
    if hasattr(idx, "date"):
        return idx.date().isoformat()
    return str(idx)


def fetch_price_context(agent: Any, tickers: list[str], start_date: str, end_date: str) -> str:
    lines: list[str] = []
    per_ticker_point_budget = max(5, agent.price_max_points // max(1, len(tickers)))
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"]
        if getattr(close, "ndim", 1) == 2:
            if close.shape[1] == 0:
                continue
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) == 0:
            continue

        returns = close.pct_change().dropna()
        perf = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 1 else 0.0
        vol = returns.std() * (252**0.5) * 100 if not returns.empty else 0.0
        rolling_max = close.cummax()
        drawdown = ((close / rolling_max) - 1).min() * 100 if len(close) > 1 else 0.0

        lines.append(
            f"- {ticker}: perf={perf:.2f}%, vol_ann={vol:.2f}%, max_drawdown={drawdown:.2f}%, "
            f"close_min={close.min():.2f}, close_max={close.max():.2f}, close_last={close.iloc[-1]:.2f}"
        )

        step = max(1, len(close) // per_ticker_point_budget)
        sampled = close.iloc[::step].tail(per_ticker_point_budget)
        points = ", ".join(f"{format_price_index(idx)}={float(val):.2f}" for idx, val in sampled.items())
        lines.append(f"  points[{ticker}]: {points}")

    if not lines:
        return ""
    return f"Fenetre prix: {start_date} -> {end_date}\n" + "\n".join(lines)


def route_after_tool_orchestrator_node(state: GraphState) -> str:
    return state.get("price_tool_decision", "continue")


@traceable(name="tool_orchestrator_node")
def tool_orchestrator_node(agent: Any, state: GraphState) -> GraphState:
    query = state.get("normalized_query", "")
    metadata_filter = state.get("metadata_filter", {})
    messages = state.get("messages", [])
    attempts = state.get("price_tool_attempts", 0)
    current_price_context = state.get("price_context", "")

    enough_price_context = has_sufficient_price_context(current_price_context)
    tickers = extract_tickers_for_price_tool(agent, query, metadata_filter)
    should_try_price, decision_reason, llm_tickers, decision_source = llm_tool_decision(
        agent=agent,
        query=query,
        metadata_filter=metadata_filter,
        messages=messages,
        enough_price_context=enough_price_context,
        attempts=attempts,
        fallback_tickers=tickers,
    )
    tickers = llm_tickers
    start_date, end_date = extract_price_date_window(agent, query)
    if attempts > 0 and not enough_price_context:
        start_date, end_date = widen_price_window(agent, start_date, end_date)

    decision = "continue"
    reason = decision_reason if decision_reason else "not_needed"

    if should_try_price and not enough_price_context:
        if not tickers:
            reason = "no_ticker"
        elif attempts < agent.price_max_attempts:
            decision = "call_price_tool"
            reason = "need_more_price_context"
        else:
            reason = "max_attempts_reached"
    elif should_try_price and enough_price_context:
        reason = "price_context_ready"

    stats = state.get("stats", {})
    stats.update(
        {
            "price_tool_attempts": attempts,
            "price_tool_decision": reason,
            "price_tool_decision_source": decision_source,
            "price_context_ready": enough_price_context,
            "price_tickers": tickers,
            "price_window_start": start_date,
            "price_window_end": end_date,
        }
    )

    return {
        "price_tool_decision": decision,
        "price_tickers": tickers,
        "price_window_start": start_date,
        "price_window_end": end_date,
        "stats": stats,
    }


@traceable(name="price_data_node")
def price_data_node(agent: Any, state: GraphState) -> GraphState:
    attempts = state.get("price_tool_attempts", 0) + 1
    tickers = state.get("price_tickers", [])
    start_date = state.get("price_window_start")
    end_date = state.get("price_window_end")

    if not tickers or not start_date or not end_date:
        stats = state.get("stats", {})
        stats.update(
            {
                "price_tool_used": False,
                "price_tool_reason": "orchestrator_missing_inputs",
                "price_tool_attempts": attempts,
            }
        )
        return {
            "price_tool_used": False,
            "price_context": state.get("price_context", ""),
            "price_tool_attempts": attempts,
            "stats": stats,
        }

    summary = fetch_price_context(agent, tickers, start_date, end_date)
    stats = state.get("stats", {})
    stats.update(
        {
            "price_tool_used": bool(summary),
            "price_tickers": tickers,
            "price_window_start": start_date,
            "price_window_end": end_date,
            "price_tool_attempts": attempts,
        }
    )
    return {
        "price_tool_used": bool(summary),
        "price_context": summary or state.get("price_context", ""),
        "price_tool_attempts": attempts,
        "stats": stats,
    }
