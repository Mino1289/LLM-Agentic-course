from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.pm_decision import (
    extract_user_response,
    parse_dollar_amount,
    parse_pm_response,
)
from src.orchestration.progress import emit_token
from src.orchestration.prompts import PM_SYSTEM_PROMPT
from src.orchestration.state import HubSpokeState
from src.orchestration.trade_intent import is_trade_requested

_LOGGER = logging.getLogger("src.orchestration.pm_node")


@traceable(name="pm_plan_node")
async def pm_plan_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    query = state.get("normalized_query") or state.get("query", "")
    compliance_reasons = state.get("compliance_reasons") or []
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Portfolio Manager",
            "status": "running",
            "message": "Création du plan d'action...",
            "tool_events": [],
        }
    )

    context = f"User request: {query}\n"
    budget = parse_dollar_amount(query)
    if budget is not None and is_trade_requested(state):
        context += (
            f"\nUser budget constraint: ${budget:,.2f} maximum notional for this trade. "
            "Convert to share quantity using the latest price and do not exceed this amount.\n"
        )
    if compliance_reasons:
        context += f"\nPrevious Compliance rejection reasons:\n" + "\n".join(
            f"- {r}" for r in compliance_reasons
        )
        context += "\n\nAdjust the plan accordingly."

    if is_trade_requested(state):
        mode_instruction = (
            "The user explicitly requested a trade or investment action. "
            "Include a DECISION block with ticker, side, quantity/amount, and order type."
        )
    else:
        mode_instruction = (
            "This is an ANALYSIS-ONLY request (comparison, research, risks, performance). "
            "Do NOT propose any trade or order. Provide only PLAN and RESPONSE — no DECISION block."
        )

    messages = [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {"role": "system", "content": mode_instruction},
        {
            "role": "system",
            "content": f"Today (UTC): {datetime.now(UTC).date().isoformat()}.",
        },
        {"role": "user", "content": context},
    ]

    try:
        full_response = ""
        async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
            messages,
            tools=None,
            temperature=0.2,
            # Le plan (délégation aux analystes) est court et interne.
            max_tokens=2048,
        ):
            if chunk.delta:
                full_response += chunk.delta

        plan = parse_pm_response(full_response)
        if not is_trade_requested(state):
            plan["action"] = "none"
            for key in ("ticker", "side", "qty", "order_type", "limit_price"):
                plan.pop(key, None)

        spoke_events.append(
            {
                "agent": "Portfolio Manager",
                "status": "completed",
                "message": "Plan d'action créé avec succès.",
                "tool_events": [],
            }
        )

        merged_stats = dict(state.get("stats") or {})
        merged_stats["pm_plan_done"] = True
        return {
            "analysis_plan": full_response,
            "pm_decision": plan,
            "spoke_events": spoke_events,
            "answer": extract_user_response(full_response),
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("PM plan failed")
        spoke_events.append(
            {
                "agent": "Portfolio Manager",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {
            "answer": f"Erreur lors de la planification: {e}",
            "spoke_events": spoke_events,
        }


@traceable(name="pm_synthesis_node")
async def pm_synthesis_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    fundamental_report = _truncate_report(
        state.get("fundamental_report", "No fundamental report available.")
    )
    quantitative_report = _truncate_report(
        state.get("quantitative_report", "No quantitative report available.")
    )
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Portfolio Manager",
            "status": "running",
            "message": "Synthèse des rapports des analystes...",
            "tool_events": [],
        }
    )

    today = datetime.now(UTC).date().isoformat()
    trade_requested = is_trade_requested(state)

    if trade_requested:
        synthesis_instructions = """You are now in the SYNTHESIS phase. Read the analyst reports and make the final investment decision.

IMPORTANT — For every factual claim in your Justification, cite the specific source (filing type and year from the Fundamental Report, or price/date from the Quantitative Report). This allows Compliance to verify your claims.

Based on these reports, produce a final investment decision in this format:
DECISION:
- Ticker: ...
- Side: buy/sell
- Quantity or amount: ...
- Order type: market/limit/stop
- Justification: ... (cite specific sources for each claim)

RESPONSE: ...
The RESPONSE section is the ONLY part shown to the user. Write it last, in the
user's language, self-contained (understandable without the DECISION block):
state the recommendation clearly, then the key reasons grounded in the reports."""
    else:
        synthesis_instructions = """You are now in the SYNTHESIS phase for an ANALYSIS-ONLY request.
The user did NOT ask to place a trade. Do NOT propose any buy/sell order or DECISION block.

Synthesize the analyst reports into a clear comparative answer in this format:
SYNTHESIS:
- Fundamental highlights: ...
- Quantitative highlights: ...
- Comparative conclusion: ...

RESPONSE: ...
The RESPONSE section is the ONLY part shown to the user. Write it last, in the
user's language, and make it self-contained (understandable without the
SYNTHESIS section): lead with the direct answer, then the key supporting facts
grounded in the analyst data. Use short paragraphs or bullet points."""

    synthesis_prompt = f"""{PM_SYSTEM_PROMPT}

Current date: {today}
{synthesis_instructions}
"""
    budget = parse_dollar_amount(state.get("normalized_query") or state.get("query", ""))
    if budget is not None and trade_requested:
        synthesis_prompt += f"""
User budget constraint: ${budget:,.2f} maximum notional. Quantity must respect this cap.
"""

    synthesis_prompt += f"""
FUNDAMENTAL REPORT:
{fundamental_report}

QUANTITATIVE REPORT:
{quantitative_report}"""

    messages = [
        {"role": "system", "content": synthesis_prompt},
        {
            "role": "user",
            "content": "Synthesize the analyst reports for the user."
            if not trade_requested
            else "Synthesize and make the final decision.",
        },
    ]

    try:
        full_response = ""
        # Pour une requête d'analyse (pas de trade), la synthèse EST la réponse
        # finale : on stream la portion RESPONSE en temps réel vers l'UI.
        # Pour un trade, la réponse finale vient de l'approbation/exécution, donc
        # on n'émet pas de tokens ici (on évite d'afficher puis remplacer).
        from src.orchestration.pm_decision import _RESPONSE_RE

        stream_to_user = not trade_requested
        emitted = 0
        response_started = False
        async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
            messages,
            tools=None,
            temperature=0.2,
            max_tokens=4096,
        ):
            if chunk.delta:
                full_response += chunk.delta
            if stream_to_user:
                if not response_started:
                    marker = _RESPONSE_RE.search(full_response)
                    if marker:
                        response_started = True
                        emitted = marker.end()
                if response_started and len(full_response) > emitted:
                    emit_token(full_response[emitted:])
                    emitted = len(full_response)

        decision = parse_pm_response(full_response)
        if not trade_requested:
            decision["action"] = "none"
            for key in ("ticker", "side", "qty", "order_type", "limit_price"):
                decision.pop(key, None)

        spoke_events.append(
            {
                "agent": "Portfolio Manager",
                "status": "completed",
                "message": "Décision d'investissement prise."
                if trade_requested
                else "Synthèse analytique terminée.",
                "tool_events": [],
            }
        )

        merged_stats = dict(state.get("stats") or {})
        merged_stats["pm_synthesis_done"] = True
        return {
            "pm_decision": decision,
            "answer": extract_user_response(full_response),
            "spoke_events": spoke_events,
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("PM synthesis failed")
        spoke_events.append(
            {
                "agent": "Portfolio Manager",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {
            "answer": f"Erreur lors de la synthèse: {e}",
            "spoke_events": spoke_events,
        }


def _truncate_report(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[... tronqué - voir les rapports complets plus haut ...]\n\n{text[-half:]}"
