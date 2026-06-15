from __future__ import annotations

from typing import Any


def parse_pm_response(text: str) -> dict[str, Any]:
    decision: dict[str, Any] = {"response": text}
    lines = text.split("\n")
    for line in lines:
        raw = line.strip()
        lower = raw.lower()
        for prefix in ("- ", "* ", "• "):
            if lower.startswith(prefix):
                lower = lower[len(prefix):]
                raw = raw[len(prefix):]
                break
        if lower.startswith("ticker:"):
            decision["ticker"] = raw.split(":", 1)[1].strip()
        elif lower.startswith("side:"):
            decision["side"] = raw.split(":", 1)[1].strip().lower()
        elif lower.startswith("quantity") or lower.startswith("qty") or lower.startswith("amount"):
            val = raw.split(":", 1)[1].strip() if ":" in raw else ""
            decision["qty"] = val
        elif lower.startswith("order type"):
            decision["order_type"] = raw.split(":", 1)[1].strip().lower() if ":" in raw else "market"
        elif lower.startswith("limit price"):
            val = raw.split(":", 1)[1].strip() if ":" in raw else ""
            decision["limit_price"] = val
    return decision


def enrich_pm_decision(state: dict[str, Any]) -> dict[str, Any]:
    decision = dict(state.get("pm_decision") or {})
    response_text = str(decision.get("response") or state.get("answer") or "")
    if response_text:
        parsed = parse_pm_response(response_text)
        for key in ("ticker", "side", "qty", "order_type", "limit_price"):
            if not decision.get(key) and parsed.get(key):
                decision[key] = parsed[key]
    return decision
