from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.chat import (
    ChatMessageDTO,
    ChatResumeRequest,
    ChatResponse,
    ChatStreamRequest,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationListResponse,
)
from api.schemas.settings import AgentSettings
from api.services.hub_runner import resume_chat, stream_chat
from api.services.run_store import run_store
from api.streaming.sse import sse_event_stream

router = APIRouter(prefix="/api", tags=["chat"])


def _format_time() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")


def _messages_to_dto(messages: list[dict]) -> list[ChatMessageDTO]:
    result: list[ChatMessageDTO] = []
    for idx, msg in enumerate(messages):
        artifacts = msg.get("artifacts")
        result.append(
            ChatMessageDTO(
                id=msg.get("id") or f"msg-{idx}",
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp") or _format_time(),
                artifacts=artifacts,
            )
        )
    return result


@router.post("/conversations", response_model=ConversationDetail)
async def create_conversation(body: ConversationCreateRequest) -> ConversationDetail:
    settings = body.settings or AgentSettings.from_env_defaults()
    record = run_store.create_conversation(locale=body.locale, settings=settings)
    return ConversationDetail(
        id=record.id,
        title=record.title,
        messages=_messages_to_dto(record.messages),
        settings=record.settings,
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(locale: str = "fr") -> ConversationListResponse:
    records = run_store.list_conversations()
    today_label = "Aujourd'hui" if locale == "fr" else "Today"
    items = [
        {"id": r.id, "title": r.title, "active": idx == 0}
        for idx, r in enumerate(records[:10])
    ]
    groups = [{"label": today_label, "items": items}] if items else []
    return ConversationListResponse(groups=groups)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    record = run_store.get_conversation(conversation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=record.id,
        title=record.title,
        messages=_messages_to_dto(record.messages),
        settings=record.settings,
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatStreamRequest) -> StreamingResponse:
    settings = body.settings or AgentSettings.from_env_defaults()
    conversation_id = body.conversation_id
    if not conversation_id:
        record = run_store.create_conversation(locale=body.locale, settings=settings)
        conversation_id = record.id
    elif run_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    events = stream_chat(conversation_id, body.message, settings, body.locale)
    return StreamingResponse(
        sse_event_stream(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/resume", response_model=ChatResponse)
async def chat_resume(body: ChatResumeRequest, locale: str = "fr") -> ChatResponse:
    try:
        return await resume_chat(body.run_id, body.approved, locale)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
