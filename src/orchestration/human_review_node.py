from __future__ import annotations

import logging
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.state import HubSpokeState

_LOGGER = logging.getLogger("src.orchestration.human_review_node")


@traceable(name="human_review")
async def human_review_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append({
        "agent": "Human Review",
        "status": "running",
        "message": "En attente de l'approbation humaine...",
        "tool_events": [],
    })

    decision = state.get("pm_decision", {})
    detail = state.get("compliance_detail", "")

    summary_lines = [
        "## Résumé de l'ordre proposé",
        "",
        f"**Ticker:** {decision.get('ticker', 'N/A')}",
        f"**Side:** {decision.get('side', 'N/A')}",
        f"**Quantité:** {decision.get('qty', 'N/A')}",
        f"**Type d'ordre:** {decision.get('order_type', 'market')}",
    ]
    if decision.get("limit_price"):
        summary_lines.append(f"**Prix limite:** {decision['limit_price']}")
    summary_lines.extend([
        "",
        "**Justification:**",
        decision.get("response", "N/A")[:500],
        "",
        "**Vérification Compliance:**",
        detail[:300],
    ])

    spoke_events.append({
        "agent": "Human Review",
        "status": "completed",
        "message": "En attente d'approbation...",
        "tool_events": [],
    })

    merged_stats = dict(state.get("stats") or {})
    return {
        "human_review_pending": True,
        "answer": "\n".join(summary_lines),
        "spoke_events": spoke_events,
        "stats": merged_stats,
    }


@traceable(name="human_approve")
async def human_approve_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    spoke_events = list(state.get("spoke_events") or [])
    spoke_events.append({
        "agent": "Human Review",
        "status": "completed",
        "message": "Trade approuvé par l'humain.",
        "tool_events": [],
    })
    return {
        "human_approved": True,
        "human_review_pending": False,
        "spoke_events": spoke_events,
    }


@traceable(name="human_reject")
async def human_reject_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    spoke_events = list(state.get("spoke_events") or [])
    spoke_events.append({
        "agent": "Human Review",
        "status": "completed",
        "message": "Trade refusé par l'humain.",
        "tool_events": [],
    })
    return {
        "human_approved": False,
        "human_review_pending": False,
        "answer": "Ordre annulé par l'utilisateur.",
        "spoke_events": spoke_events,
    }


def route_after_human_review(state: HubSpokeState) -> str:
    if state.get("human_approved"):
        return "executor_trader"
    return "__end__"
