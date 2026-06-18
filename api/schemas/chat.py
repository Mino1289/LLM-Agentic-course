from __future__ import annotations

from pydantic import BaseModel, Field

from api.schemas.artifacts import MessageArtifacts, TradeProposal
from api.schemas.settings import AgentSettings


class ChatMessageDTO(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    artifacts: MessageArtifacts | None = None


class ChatStreamRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    locale: str = "fr"
    settings: AgentSettings | None = None


class ChatResumeRequest(BaseModel):
    run_id: str
    approved: bool
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    run_id: str | None = None
    answer: str
    human_review_pending: bool = False
    artifacts: MessageArtifacts = Field(default_factory=MessageArtifacts)
    message: ChatMessageDTO | None = None


class HumanReviewPayload(BaseModel):
    run_id: str
    conversation_id: str
    answer: str
    trade: TradeProposal
    artifacts: MessageArtifacts = Field(default_factory=MessageArtifacts)


class ConversationCreateRequest(BaseModel):
    locale: str = "fr"
    settings: AgentSettings | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[ChatMessageDTO]
    settings: AgentSettings


class ConversationListResponse(BaseModel):
    groups: list[dict]
