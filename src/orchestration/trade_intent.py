from __future__ import annotations

from typing import Any

from src.orchestration.pm_decision import enrich_pm_decision
from src.orchestration.tool_domains import detect_tool_domains

_NON_TRADE_SIDES = frozenset(
    {
        "none",
        "n/a",
        "na",
        "hold",
        "neutral",
        "neutre",
        "wait",
        "no action",
        "no trade",
        "informational",
        "analyse",
        "analysis",
        "compare",
        "comparison",
        "monitor",
        "surveiller",
    }
)

_SIDE_ALIASES = {
    "achat": "buy",
    "acheter": "buy",
    "vente": "sell",
    "vendre": "sell",
    "long": "buy",
    "short": "sell",
}


def _normalize_side(raw: str) -> str | None:
    side = (raw or "").strip().lower()
    if not side or side in _NON_TRADE_SIDES:
        return None
    side = _SIDE_ALIASES.get(side, side)
    if side in ("buy", "sell"):
        return side
    return None


def _invalid_ticker(raw: str) -> bool:
    ticker = (raw or "").strip().lower()
    return not ticker or ticker in {"n/a", "na", "none", "null", "-", "—", "?"}


def is_trade_requested(state: dict[str, Any]) -> bool:
    if state.get("trade_requested") is True:
        return True
    if state.get("trade_requested") is False:
        return False

    stats = state.get("stats") or {}
    domains = stats.get("tool_domains")
    if domains is None:
        query = state.get("normalized_query") or state.get("query") or ""
        domains = sorted(detect_tool_domains(query))
    if "trade" in domains:
        return True
    return stats.get("intent_reason") == "action_keyword"


def has_trade_proposal(state: dict[str, Any]) -> bool:
    decision = enrich_pm_decision(state)
    if decision.get("action") == "none":
        return False

    ticker = str(decision.get("ticker", "")).strip()
    if _invalid_ticker(ticker):
        return False

    side = _normalize_side(str(decision.get("side", "")))
    if not side:
        return False

    return True


def route_after_pm_synthesis(state: dict[str, Any]) -> str:
    if has_trade_proposal(state):
        return "compliance_validator"
    return "__end__"


def route_after_compliance(state: dict[str, Any]) -> str:
    if not has_trade_proposal(state):
        return "__end__"
    verdict = state.get("compliance_verdict", "FAIL")
    if verdict in ("PASS", "OVERRULED"):
        return "human_review"
    # FAIL après épuisement des tentatives : on arrête au lieu de reboucler à l'infini.
    if state.get("compliance_exhausted"):
        return "__end__"
    return "pm_plan"
