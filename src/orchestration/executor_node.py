from __future__ import annotations

import json
import logging
from typing import Any

from src.graph.tracing import traceable
from src.orchestration._spoke_helpers import run_spoke_agent
from src.orchestration.prompts import EXECUTOR_PROMPT
from src.orchestration.state import HubSpokeState

_LOGGER = logging.getLogger("src.orchestration.executor_node")

EXECUTOR_TOOLS = ["place_trade_tool", "close_position_tool"]


@traceable(name="executor_trader")
async def executor_trader_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    decision = state.get("pm_decision", {})
    decision_text = decision.get("response", str(decision))
    compliance_detail = state.get("compliance_detail", "No compliance detail")
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Executor Trader",
            "status": "running",
            "message": "Exécution de l'ordre sur le marché...",
            "tool_events": [],
        }
    )

    task = f"""Compliance approved this decision:
{decision_text}

Compliance detail: {compliance_detail}

Execute the trade exactly as specified. Use place_trade_tool or close_position_tool.
Confirm the execution result: ticker, side, quantity, filled price, status."""

    try:
        result, spoke_stats = await run_spoke_agent(
            agent,
            EXECUTOR_PROMPT,
            task,
            EXECUTOR_TOOLS,
            dict(state),
        )

        spoke_events.append(
            {
                "agent": "Executor Trader",
                "status": "completed",
                "message": "Ordre exécuté avec succès.",
                "tool_events": [],
            }
        )

        merged_stats = dict(state.get("stats") or {})
        merged_stats.update(spoke_stats)
        return {
            "trade_result": {"status": "executed", "detail": result},
            "answer": result,
            "spoke_events": spoke_events,
            "stats": merged_stats,
        }
    except Exception as e:
        _LOGGER.exception("Executor trader failed")
        spoke_events.append(
            {
                "agent": "Executor Trader",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {
            "trade_result": {"status": "failed", "error": str(e)},
            "answer": f"Erreur lors de l'exécution: {e}",
            "spoke_events": spoke_events,
        }
