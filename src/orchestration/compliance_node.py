from __future__ import annotations

import logging
import os
import re
from typing import Any

from src.graph.tracing import traceable
from src.orchestration._spoke_helpers import run_spoke_agent
from src.orchestration.prompts import COMPLIANCE_PROMPT
from src.orchestration.state import HubSpokeState
from src.orchestration.trade_intent import (
    route_after_compliance as _route_after_compliance,
)

_LOGGER = logging.getLogger("src.orchestration.compliance_node")

COMPLIANCE_TOOLS = [
    "validate_claims_tool",
    "portfolio_info_tool",
    "account_activity_tool",
]

# Nombre de passages Compliance avant d'arrêter la boucle PM↔Compliance.
_MAX_COMPLIANCE_ATTEMPTS = 2


def _overrule_enabled() -> bool:
    """Forcer le trade malgré un FAIL après N tentatives (désactivé par défaut)."""
    return os.getenv("COMPLIANCE_ALLOW_OVERRULE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@traceable(name="compliance_validator")
async def compliance_validator_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    decision = state.get("pm_decision", {})
    decision_text = decision.get("response", str(decision))
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Compliance Validator",
            "status": "running",
            "message": "Vérification de la conformité de l'ordre...",
            "tool_events": [],
        }
    )

    task = f"""Décision d'investissement à valider :
{decision_text}

Utilise portfolio_info_tool pour vérifier le pouvoir d'achat et les positions.
Utilise account_activity_tool pour vérifier l'activité récente.
Utilise validate_claims_tool pour vérifier les affirmations si des données RAG sont disponibles.

Retourne PASS ou FAIL avec des raisons spécifiques et actionnables."""

    # Compter le nombre de passages dans Compliance via spoke_events
    compliance_count = (
        sum(
            1
            for e in spoke_events
            if e.get("agent") == "Compliance Validator" and e.get("status") == "running"
        )
        + 1
    )

    try:
        result, spoke_stats = await run_spoke_agent(
            agent,
            COMPLIANCE_PROMPT,
            task,
            COMPLIANCE_TOOLS,
            dict(state),
            max_iterations=3,
        )
        result_lower = result.strip().lower()
        first_word = result_lower.split(maxsplit=1)[0] if result_lower else ""
        verdict = "PASS" if first_word == "pass" else "FAIL"
        reasons = _extract_reasons(result)
        exhausted = False
        answer: str | None = None

        if verdict == "FAIL" and compliance_count >= _MAX_COMPLIANCE_ATTEMPTS:
            if _overrule_enabled():
                verdict = "OVERRULED"
                reasons = [
                    "Maximum de tentatives atteint. Décision forcée après révision "
                    "(COMPLIANCE_ALLOW_OVERRULE actif).",
                    *reasons,
                ]
            else:
                # Garde-fou strict : on ne force pas, on arrête la boucle proprement.
                exhausted = True
                answer = (
                    "❌ **Ordre bloqué par la conformité**\n\n"
                    f"Après {compliance_count} tentatives, le Compliance Validator "
                    "maintient son refus. Aucun ordre n'a été soumis.\n\n"
                    f"Raison principale : {reasons[0] if reasons else 'non précisée'}"
                )

        summary = reasons[0][:2000] if reasons else "Aucune raison détaillée."
        spoke_events.append(
            {
                "agent": "Compliance Validator",
                "status": "completed",
                "message": f"Verdict: {verdict}",
                "detail": summary,
                "tool_events": [],
            }
        )

        merged_stats = dict(state.get("stats") or {})
        merged_stats.update(spoke_stats)
        result_state: dict[str, Any] = {
            "compliance_verdict": verdict,
            "compliance_reasons": reasons,
            "compliance_detail": result,
            "compliance_exhausted": exhausted,
            "spoke_events": spoke_events,
            "stats": merged_stats,
        }
        if answer is not None:
            result_state["answer"] = answer
        return result_state
    except Exception as e:
        _LOGGER.exception("Compliance validator failed")
        spoke_events.append(
            {
                "agent": "Compliance Validator",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {
            "compliance_verdict": "FAIL",
            "compliance_reasons": [f"Validation error: {e}"],
            "compliance_detail": f"Error: {e}",
            "spoke_events": spoke_events,
        }


def route_after_compliance(state: HubSpokeState) -> str:
    return _route_after_compliance(state)


_REASON_KEYWORDS = (
    "reason",
    "because",
    "issue",
    "insufficient",
    "exceeds",
    "exceed",
    "missing",
    "concentration",
    "buying power",
    "reduce",
    "limit",
    "contradict",
    "unsupported",
)

# Préfixe verdict imposé par COMPLIANCE_PROMPT ("PASS"/"FAIL" en premier mot).
_VERDICT_PREFIX = re.compile(r"^\W*(pass|fail)\b[\s:.\-–—]*", re.IGNORECASE)


def _is_bare_verdict(line: str) -> bool:
    """True si la ligne n'est QUE le verdict (ex. "FAIL", "**FAIL**", "FAIL.")."""
    return re.sub(r"[^a-z]", "", line.lower()) in {"pass", "fail"}


def _strip_verdict_prefix(line: str) -> str:
    """Retire un "FAIL."/"PASS:" collé en tête de ligne, garde l'explication."""
    return _VERDICT_PREFIX.sub("", line, count=1).strip()


def _extract_reasons(result: str) -> list[str]:
    # Lignes utiles : non vides et pas seulement le verdict PASS/FAIL, qui sinon
    # devenait à tort la "raison principale" affichée à l'utilisateur.
    useful: list[str] = []
    for line in result.split("\n"):
        stripped = line.strip()
        if not stripped or _is_bare_verdict(stripped):
            continue
        useful.append(_strip_verdict_prefix(stripped) or stripped)
    if not useful:
        return ["Aucune raison détaillée fournie par la conformité."]
    # On privilégie les lignes qui ressemblent à une raison actionnable, mais on
    # ne renvoie JAMAIS un simple "FAIL".
    keyworded = [
        line for line in useful if any(kw in line.lower() for kw in _REASON_KEYWORDS)
    ]
    return keyworded or useful
