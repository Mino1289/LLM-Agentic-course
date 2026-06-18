from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from api.schemas.chat import ChatResponse, HumanReviewPayload
from api.schemas.settings import AgentSettings
from api.services.agent_factory import agent_factory
from api.services.artifact_mapper import (
    message_dto_from_assistant,
    state_to_human_review,
    state_to_response,
)
from api.services.run_store import run_store


def _format_time() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")


async def stream_chat(
    conversation_id: str,
    message: str,
    settings: AgentSettings,
    locale: str = "fr",
) -> AsyncIterator[dict[str, Any]]:
    record = run_store.get_conversation(conversation_id)
    if record is None:
        yield {"type": "error", "message": "Conversation not found"}
        return

    run_store.append_message(conversation_id, {"role": "user", "content": message})
    hub = agent_factory.get_hub_graph(settings)
    run_id = str(uuid.uuid4())

    yield {
        "type": "spoke",
        "agent": "Orchestration",
        "status": "running",
        "message": "Pipeline démarré...",
    }

    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in record.messages
    ]

    final_state: dict[str, Any] | None = None

    try:
        async for event in hub.astream(message, conversation_id, api_messages):
            kind = event.get("event")

            if kind == "on_spoke_event":
                yield {
                    "type": "spoke",
                    "agent": event.get("agent"),
                    "status": event.get("status"),
                    "message": event.get("message"),
                }
            elif kind == "on_tool_start":
                tool = event.get("name", "?")
                yield {"type": "tool_start", "name": tool}
            elif kind == "on_tool_end":
                yield {"type": "tool_end", "name": event.get("name")}
            elif kind == "on_llm_token":
                token = event.get("token", "")
                if token:
                    yield {"type": "token", "token": token}
            elif kind == "on_graph_end":
                final_state = event.get("state") or {}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    if final_state is None:
        yield {"type": "error", "message": "No final state from graph"}
        return

    if final_state.get("human_review_pending"):
        run_store.save_pending_run(run_id, conversation_id, settings, final_state)
        payload = state_to_human_review(final_state, conversation_id, run_id, locale)
        assistant_msg = message_dto_from_assistant(
            final_state,
            message_id=str(uuid.uuid4()),
            timestamp=_format_time(),
            locale=locale,
        )
        run_store.append_message(
            conversation_id,
            {
                "role": "assistant",
                "content": assistant_msg.content,
                "artifacts": assistant_msg.artifacts.model_dump(by_alias=True)
                if assistant_msg.artifacts
                else None,
            },
        )
        yield {
            "type": "human_review_required",
            "payload": json.loads(payload.model_dump_json()),
        }
        return

    response = state_to_response(final_state, conversation_id, run_id, locale)
    assistant_msg = message_dto_from_assistant(
        final_state,
        message_id=str(uuid.uuid4()),
        timestamp=_format_time(),
        locale=locale,
    )
    response.message = assistant_msg
    run_store.append_message(
        conversation_id,
        {
            "role": "assistant",
            "content": assistant_msg.content,
            "artifacts": assistant_msg.artifacts.model_dump(by_alias=True)
            if assistant_msg.artifacts
            else None,
        },
    )
    yield {"type": "done", "payload": json.loads(response.model_dump_json())}


async def resume_chat(run_id: str, approved: bool, locale: str = "fr") -> ChatResponse:
    pending = run_store.pop_pending_run(run_id)
    if pending is None:
        raise ValueError("Pending run not found")

    hub = agent_factory.get_hub_graph(pending.settings)
    final_state = await hub.resume_after_human(pending.state, approved=approved)
    response = state_to_response(
        final_state,
        pending.conversation_id,
        run_id,
        locale,
    )
    assistant_msg = message_dto_from_assistant(
        final_state,
        message_id=str(uuid.uuid4()),
        timestamp=_format_time(),
        locale=locale,
    )
    response.message = assistant_msg
    run_store.append_message(
        pending.conversation_id,
        {
            "role": "assistant",
            "content": assistant_msg.content,
            "artifacts": assistant_msg.artifacts.model_dump(by_alias=True)
            if assistant_msg.artifacts
            else None,
        },
    )
    return response
