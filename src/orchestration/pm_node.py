from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.prompts import PM_SYSTEM_PROMPT
from src.orchestration.state import HubSpokeState

_LOGGER = logging.getLogger("src.orchestration.pm_node")


@traceable(name="pm_plan_node")
async def pm_plan_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    query = state.get("normalized_query") or state.get("query", "")
    compliance_reasons = state.get("compliance_reasons") or []
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append({
        "agent": "Portfolio Manager",
        "status": "running",
        "message": "Création du plan d'action...",
        "tool_events": [],
    })

    context = f"User request: {query}\n"
    if compliance_reasons:
        context += f"\nPrevious Compliance rejection reasons:\n" + "\n".join(f"- {r}" for r in compliance_reasons)
        context += "\n\nAdjust the plan accordingly."

    messages = [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {"role": "system", "content": f"Today (UTC): {datetime.now(UTC).date().isoformat()}."},
        {"role": "user", "content": context},
    ]

    try:
        full_response = ""
        async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
            messages, tools=None, temperature=0.2, max_tokens=4096,
        ):
            if chunk.delta:
                full_response += chunk.delta

        plan = _parse_pm_response(full_response)

        spoke_events.append({
            "agent": "Portfolio Manager",
            "status": "completed",
            "message": "Plan d'action créé avec succès.",
            "tool_events": [],
        })

        merged_stats = dict(state.get("stats") or {})
        merged_stats["pm_plan_done"] = True
        return {
            "analysis_plan": full_response,
            "pm_decision": plan,
            "spoke_events": spoke_events,
            "answer": plan.get("response", ""),
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("PM plan failed")
        spoke_events.append({
            "agent": "Portfolio Manager",
            "status": "failed",
            "message": f"Erreur: {e}",
            "tool_events": [],
        })
        return {
            "answer": f"Erreur lors de la planification: {e}",
            "spoke_events": spoke_events,
        }


@traceable(name="pm_synthesis_node")
async def pm_synthesis_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    fundamental_report = _truncate_report(state.get("fundamental_report", "No fundamental report available."))
    quantitative_report = _truncate_report(state.get("quantitative_report", "No quantitative report available."))
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append({
        "agent": "Portfolio Manager",
        "status": "running",
        "message": "Synthèse des rapports des analystes...",
        "tool_events": [],
    })

    today = datetime.now(UTC).date().isoformat()
    synthesis_prompt = f"""{PM_SYSTEM_PROMPT}

Current date: {today}
You are now in the SYNTHESIS phase. Read the analyst reports and make the final investment decision.

IMPORTANT — For every factual claim in your Justification, cite the specific source (filing type and year from the Fundamental Report, or price/date from the Quantitative Report). This allows Compliance to verify your claims.

FUNDAMENTAL REPORT:
{fundamental_report}

QUANTITATIVE REPORT:
{quantitative_report}

Based on these reports, produce a final investment decision in this format:
DECISION:
- Ticker: ...
- Side: buy/sell
- Quantity or amount: ...
- Order type: market/limit/stop
- Justification: ... (cite specific sources for each claim)

RESPONSE (French, human-readable): ..."""

    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": "Synthesize and make the final decision."},
    ]

    try:
        full_response = ""
        async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
            messages, tools=None, temperature=0.2, max_tokens=4096,
        ):
            if chunk.delta:
                full_response += chunk.delta

        decision = _parse_pm_response(full_response)

        spoke_events.append({
            "agent": "Portfolio Manager",
            "status": "completed",
            "message": "Décision d'investissement prise.",
            "tool_events": [],
        })

        merged_stats = dict(state.get("stats") or {})
        merged_stats["pm_synthesis_done"] = True
        return {
            "pm_decision": decision,
            "answer": decision.get("response", full_response),
            "spoke_events": spoke_events,
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("PM synthesis failed")
        spoke_events.append({
            "agent": "Portfolio Manager",
            "status": "failed",
            "message": f"Erreur: {e}",
            "tool_events": [],
        })
        return {"answer": f"Erreur lors de la synthèse: {e}", "spoke_events": spoke_events}


def _parse_pm_response(text: str) -> dict[str, Any]:
    decision: dict[str, Any] = {"response": text}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        raw = line.strip()
        lower = raw.lower()
        # Enlève les prefixes comme "- ", "* ", "• " pour matcher les champs
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


def _truncate_report(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[... tronqué - voir les rapports complets plus haut ...]\n\n{text[-half:]}"
