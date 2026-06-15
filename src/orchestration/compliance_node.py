from __future__ import annotations

import logging
from typing import Any

from src.graph.tracing import traceable
from src.orchestration._spoke_helpers import run_spoke_agent
from src.orchestration.prompts import COMPLIANCE_PROMPT
from src.orchestration.state import HubSpokeState
from src.orchestration.trade_intent import route_after_compliance as _route_after_compliance

_LOGGER = logging.getLogger("src.orchestration.compliance_node")

COMPLIANCE_TOOLS = ["validate_claims_tool", "portfolio_info_tool", "account_activity_tool"]


@traceable(name="compliance_validator")
async def compliance_validator_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    decision = state.get("pm_decision", {})
    decision_text = decision.get("response", str(decision))
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append({
        "agent": "Compliance Validator",
        "status": "running",
        "message": "Vérification de la conformité de l'ordre...",
        "tool_events": [],
    })

    task = f"""Décision d'investissement à valider :
{decision_text}

Utilise portfolio_info_tool pour vérifier le pouvoir d'achat et les positions.
Utilise account_activity_tool pour vérifier l'activité récente.
Utilise validate_claims_tool pour vérifier les affirmations si des données RAG sont disponibles.

Retourne PASS ou FAIL avec des raisons spécifiques et actionnables."""

    # Compter le nombre de passages dans Compliance via spoke_events
    compliance_count = sum(1 for e in spoke_events if e.get("agent") == "Compliance Validator" and e.get("status") == "running") + 1

    try:
        result, spoke_stats = await run_spoke_agent(
            agent, COMPLIANCE_PROMPT, task, COMPLIANCE_TOOLS, dict(state), max_iterations=3,
        )
        result_lower = result.strip().lower()
        first_word = result_lower.split(maxsplit=1)[0] if result_lower else ""
        verdict = "PASS" if first_word == "pass" else "FAIL"
        reasons = _extract_reasons(result)

        if verdict == "FAIL" and compliance_count >= 2:
            verdict = "OVERRULED"
            reasons = ["Maximum de tentatives atteint. Décision forcée après révision.", *reasons]

        summary = reasons[0][:2000] if reasons else "Aucune raison détaillée."
        spoke_events.append({
            "agent": "Compliance Validator",
            "status": "completed",
            "message": f"Verdict: {verdict}",
            "detail": summary,
            "tool_events": [],
        })

        merged_stats = dict(state.get("stats") or {})
        merged_stats.update(spoke_stats)
        return {
            "compliance_verdict": verdict,
            "compliance_reasons": reasons,
            "compliance_detail": result,
            "spoke_events": spoke_events,
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("Compliance validator failed")
        spoke_events.append({
            "agent": "Compliance Validator",
            "status": "failed",
            "message": f"Erreur: {e}",
            "tool_events": [],
        })
        return {
            "compliance_verdict": "FAIL",
            "compliance_reasons": [f"Validation error: {e}"],
            "compliance_detail": f"Error: {e}",
            "spoke_events": spoke_events,
        }


def route_after_compliance(state: HubSpokeState) -> str:
    return _route_after_compliance(state)


def _extract_reasons(result: str) -> list[str]:
    lines = result.split("\n")
    reasons = []
    for line in lines:
        lower = line.strip().lower()
        if any(kw in lower for kw in ["reason", "because", "issue", "error", "fail", "insufficient", "exceeds", "missing"]):
            reasons.append(line.strip())
    if not reasons:
        reasons = [result.strip()]
    return reasons
