"""Noeud outils — décision LLM d'appel market_price, extraction fenêtre de prix, yfinance."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from src.graph.prompt_context import format_universe_hint


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
    found = []
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
    deduped = []
    for token in candidates:
        t = token.upper()
        if t in allowed and t not in deduped:
            deduped.append(t)
    return deduped[:max_items]


def has_sufficient_price_context(price_context: str) -> bool:
    if not price_context:
        return False
    return "Fenetre prix:" in price_context and "points[" in price_context


def _minimal_tool_fallback(query: str, metadata_filter: dict[str, str], fallback_tickers: list[str]) -> tuple[bool, str, list[str]]:
    if not query.strip():
        return False, "fallback_empty_query", fallback_tickers
    lower_q = query.lower()
    explicit_price_keywords = ["prix", "cours", "performance", "volatil", "drawdown", "return", "chart"]
    if any(token in lower_q for token in explicit_price_keywords):
        return True, "fallback_explicit_price_keyword", fallback_tickers
    if metadata_filter.get("ticker") and "compare" in lower_q:
        return True, "fallback_compare_with_ticker", fallback_tickers
    return False, "fallback_not_needed", fallback_tickers


def llm_tool_decision(agent: Any, query: str, metadata_filter: dict[str, str],
                       messages: list[dict[str, str]], enough_price_context: bool,
                       attempts: int, fallback_tickers: list[str]) -> tuple[bool, str, list[str], str]:
    fallback_decision, fallback_reason, fallback_tickers = _minimal_tool_fallback(query, metadata_filter, fallback_tickers)
    if enough_price_context:
        return False, "price_context_ready", fallback_tickers, "rule"
    recent_user_turns = [m.get("content", "").strip() for m in messages[-4:] if m.get("role") == "user" and m.get("content", "").strip()]
    allowed_tickers = _allowed_tickers_from_universe(agent, max_items=20)
    universe_hint = ", ".join(allowed_tickers[:14])
    prompt = (
        "You are a tool router for a finance assistant.\n"
        "Decide whether the market-price tool should be called BEFORE answering.\n"
        'Schema: {"use_price_tool": bool, "reason": string, "tickers": string[]}\n'
        "Set use_price_tool=true only if price/performance/volatility/momentum/comparison "
        "materially improves answer quality. "
        f"User question: {query}\nDetected metadata: {metadata_filter}\n"
        f"Price context available: {enough_price_context}\nAttempts: {attempts}\n"
        f"Candidate tickers: {fallback_tickers}\nCovered: {universe_hint}\n"
        "Return STRICT JSON only."
    )
    raw = ""
    try:
        raw = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=180)
        parsed = _extract_first_json_object(raw)
        if parsed and isinstance(parsed.get("use_price_tool"), bool):
            llm_tickers = _normalize_tickers(parsed.get("tickers"), allowed_tickers, agent.price_max_tickers)
            if not llm_tickers:
                llm_tickers = fallback_tickers
            return parsed["use_price_tool"], str(parsed.get("reason", "llm_decision")), llm_tickers, "llm"
    except Exception:
        pass
    return fallback_decision, f"fallback_heuristic|{fallback_reason}", fallback_tickers, "heuristic"


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
        delta = min(value * (1 if "jour" in unit or "day" in unit else 30 if "mois" in unit or "month" in unit else 365), agent.price_max_days)
        start_date = today - timedelta(days=delta)
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
    lines = []
    per_ticker_point_budget = max(5, agent.price_max_points // max(1, len(tickers)))
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        except Exception:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"]
        if getattr(close, "ndim", 1) == 2 and close.shape[1] > 0:
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) == 0:
            continue
        returns = close.pct_change().dropna()
        perf = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 1 else 0.0
        vol = returns.std() * (252**0.5) * 100 if not returns.empty else 0.0
        drawdown = ((close / close.cummax()) - 1).min() * 100 if len(close) > 1 else 0.0
        lines.append(f"- {ticker}: perf={perf:.2f}%, vol_ann={vol:.2f}%, max_drawdown={drawdown:.2f}%, close_min={close.min():.2f}, close_max={close.max():.2f}, close_last={close.iloc[-1]:.2f}")
        step = max(1, len(close) // per_ticker_point_budget)
        sampled = close.iloc[::step].tail(per_ticker_point_budget)
        points = ", ".join(f"{format_price_index(idx)}={float(val):.2f}" for idx, val in sampled.items())
        lines.append(f"  points[{ticker}]: {points}")
    if not lines:
        return ""
    return f"Fenetre prix: {start_date} -> {end_date}\n" + "\n".join(lines)
