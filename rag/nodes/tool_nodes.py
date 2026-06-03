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


def _allowed_tickers_from_universe(agent: Any, max_items: int = 20) -> list[str]:
    universe_hint = format_universe_hint(agent, max_items=max_items)
    return [token.strip().upper() for token in universe_hint.split(",") if token.strip()]


def _extract_explicit_tickers(query: str, allowed_tickers: list[str], max_items: int) -> list[str]:
    allowed = set(allowed_tickers)
    found: list[str] = []
    for token in re.findall(r"\b[A-Za-z]{2,5}\b", query):
        up = token.upper()
        if up in allowed and up not in found:
            found.append(up)
        if len(found) >= max_items:
            break
    return found


def _normalize_tickers(raw_tickers: Any, allowed_tickers: list[str], max_items: int) -> list[str]:
    if isinstance(raw_tickers, str):
        candidates = re.split(r"[,\s]+", raw_tickers.strip())
    elif isinstance(raw_tickers, list):
        candidates = [str(t).strip() for t in raw_tickers]
    else:
        candidates = []

    allowed = set(allowed_tickers)
    deduped: list[str] = []
    for token in candidates:
        t = token.upper()
        if t in allowed and t not in deduped:
            deduped.append(t)
    return deduped[:max_items]


def has_sufficient_price_context(price_context: str) -> bool:
    if not price_context:
        return False
    return "Fenetre prix:" in price_context and "points[" in price_context


def _minimal_tool_fallback(
    query: str,
    metadata_filter: dict[str, str],
    fallback_tickers: list[str],
) -> tuple[bool, str, list[str]]:
    # Keep deterministic fallback small: only explicit price intent.
    if not query.strip():
        return False, "fallback_empty_query", fallback_tickers
    lower_q = query.lower()
    explicit_price_keywords = ["prix", "cours", "performance", "volatil", "drawdown", "return", "chart"]
    if any(token in lower_q for token in explicit_price_keywords):
        return True, "fallback_explicit_price_keyword", fallback_tickers
    if metadata_filter.get("ticker") and "compare" in lower_q:
        return True, "fallback_compare_with_ticker", fallback_tickers
    return False, "fallback_not_needed", fallback_tickers


def llm_tool_decision(
    agent: Any,
    query: str,
    metadata_filter: dict[str, str],
    messages: list[dict[str, str]],
    enough_price_context: bool,
    attempts: int,
    fallback_tickers: list[str],
) -> tuple[bool, str, list[str], str]:
    fallback_decision, fallback_reason, fallback_tickers = _minimal_tool_fallback(
        query=query,
        metadata_filter=metadata_filter,
        fallback_tickers=fallback_tickers,
    )
    if enough_price_context:
        return False, "price_context_ready", fallback_tickers, "rule"

    recent_user_turns = [
        m.get("content", "").strip()
        for m in messages[-4:]
        if m.get("role") == "user" and m.get("content", "").strip()
    ]
    allowed_tickers = _allowed_tickers_from_universe(agent, max_items=20)
    universe_hint = ", ".join(allowed_tickers[:14])
    prompt = (
        "You are a tool router for a finance assistant.\n"
        "Decide whether the market-price tool (price time series) should be called BEFORE answering.\n"
        "Return STRICT JSON only, with no surrounding text.\n"
        'Schema: {"use_price_tool": bool, "reason": string, "tickers": string[]}\n'
        "Decision rules:\n"
        "1) Set use_price_tool=true only if price/performance/volatility/momentum/comparison "
        "materially improves answer quality.\n"
        "2) If the question is purely fundamental and not market-timing oriented, prefer false.\n"
        "3) Prioritize tickers explicitly mentioned or resolved from metadata/context.\n"
        "4) Use ONLY tickers from the covered universe.\n"
        "5) If no relevant ticker can be identified, return use_price_tool=false and tickers=[].\n\n"
        f"User question: {query}\n"
        f"Detected metadata: {metadata_filter}\n"
        f"Recent user turns: {recent_user_turns}\n"
        f"Price context already available: {enough_price_context}\n"
        f"Tool attempts already used: {attempts}\n"
        f"Heuristic candidate tickers: {fallback_tickers}\n"
        f"Covered companies (tickers): {universe_hint}\n"
    )
    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=180)
        parsed = _extract_first_json_object(raw)
        if parsed is not None and isinstance(parsed.get("use_price_tool"), bool):
            llm_tickers = _normalize_tickers(parsed.get("tickers"), allowed_tickers, agent.price_max_tickers)
            if not llm_tickers:
                llm_tickers = fallback_tickers
            reason = str(parsed.get("reason", "llm_decision")).strip() or "llm_decision"
            return parsed["use_price_tool"], reason, llm_tickers, "llm"
    except Exception:
        pass

    fallback_reason = f"fallback_heuristic_invalid_llm_output|{fallback_reason}"
    if raw.strip():
        fallback_reason = f"{fallback_reason}|fallback_parse_error"
    return fallback_decision, fallback_reason, fallback_tickers, "heuristic"


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

    if os.getenv("PRICE_TOOL_ENABLED", "true").lower() in {"0", "false", "no"}:
        stats = state.get("stats", {})
        stats.update(
            {
                "price_tool_attempts": attempts,
                "price_tool_decision": "disabled_by_config",
                "price_tool_decision_source": "config",
                "price_context_ready": False,
                "price_tickers": [],
            }
        )
        return {
            "price_tool_decision": "continue",
            "price_tickers": [],
            "stats": stats,
        }

    enough_price_context = has_sufficient_price_context(current_price_context)
    tickers = []
    allowed_tickers = _allowed_tickers_from_universe(agent, max_items=20)
    if metadata_filter.get("ticker"):
        tickers.append(metadata_filter["ticker"].upper())
    for ticker in (state.get("target_tickers") or []):
        up = str(ticker).upper()
        if up in allowed_tickers and up not in tickers:
            tickers.append(up)
    explicit_mentions = _extract_explicit_tickers(query, allowed_tickers, max_items=agent.price_max_tickers)
    for up in explicit_mentions:
        if up not in tickers:
            tickers.append(up)
    tickers = tickers[: agent.price_max_tickers]
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
