from __future__ import annotations

import logging
from typing import Any

from src.graph.tracing import traceable
from src.orchestration.progress import emit_agent_progress
from src.orchestration.state import HubSpokeState

_LOGGER = logging.getLogger("src.orchestration.simple_agent_node")


@traceable(name="simple_agent")
async def simple_agent_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    query = state.get("normalized_query") or state.get("query", "")
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Simple Agent (Phase 2)",
            "status": "running",
            "message": "Traitement de la requête simple...",
            "tool_events": [],
        }
    )
    emit_agent_progress(
        "Simple Agent",
        "running",
        "Agent simple — traitement en cours...",
    )

    try:
        result_state = await agent.arun(
            query, state.get("conversation_id"), state.get("messages", [])
        )

        answer = result_state.get("answer", "")
        tool_events = result_state.get("tool_events", [])

        spoke_events.append(
            {
                "agent": "Simple Agent (Phase 2)",
                "status": "completed",
                "message": "Requête traitée.",
                "tool_events": [dict(te) for te in (tool_events or [])],
            }
        )

        return {
            "answer": answer,
            "tool_events": tool_events,
            "stats": result_state.get("stats", {}),
            "spoke_events": spoke_events,
            "conversation_id": result_state.get("conversation_id"),
            "lc_messages": list(result_state.get("lc_messages") or []),
            "messages": list(result_state.get("messages") or []),
        }
    except Exception as e:
        _LOGGER.exception("Simple agent failed")
        spoke_events.append(
            {
                "agent": "Simple Agent (Phase 2)",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {
            "answer": f"Erreur lors du traitement: {e}",
            "spoke_events": spoke_events,
        }
