from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from api.schemas.chat import ChatMessageDTO
from api.schemas.settings import AgentSettings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


WELCOME_MESSAGES = {
    "fr": (
        "Bonjour. Je peux analyser **NVDA, AMD, MSFT, ARM, ASML** — interroger les filings SEC, "
        "les transcripts earnings, valider des affirmations, simuler une allocation, récupérer les "
        "prix et générer des rapports."
    ),
    "en": (
        "Hello. I can analyze **NVDA, AMD, MSFT, ARM, ASML** — query SEC filings, earnings transcripts, "
        "validate claims, simulate allocation, fetch prices, and generate reports."
    ),
}


@dataclass
class ConversationRecord:
    id: str
    title: str
    locale: str
    settings: AgentSettings
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


@dataclass
class PendingRun:
    run_id: str
    conversation_id: str
    settings: AgentSettings
    state: dict[str, Any]


class RunStore:
    def __init__(self) -> None:
        self.conversations: dict[str, ConversationRecord] = {}
        self.pending_runs: dict[str, PendingRun] = {}

    def create_conversation(
        self,
        locale: str = "fr",
        settings: AgentSettings | None = None,
    ) -> ConversationRecord:
        conv_id = str(uuid.uuid4())
        settings = settings or AgentSettings.from_env_defaults()
        welcome = WELCOME_MESSAGES.get(locale, WELCOME_MESSAGES["fr"])
        title = "Nouvelle conversation" if locale == "fr" else "New conversation"
        record = ConversationRecord(
            id=conv_id,
            title=title,
            locale=locale,
            settings=settings,
            messages=[{"role": "assistant", "content": welcome}],
        )
        self.conversations[conv_id] = record
        return record

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        return self.conversations.get(conversation_id)

    def list_conversations(self) -> list[ConversationRecord]:
        return sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )

    def append_message(self, conversation_id: str, message: dict[str, Any]) -> None:
        record = self.conversations[conversation_id]
        record.messages.append(message)
        record.updated_at = _now_iso()
        if message.get("role") == "user" and record.title in {
            "Nouvelle conversation",
            "New conversation",
        }:
            content = str(message.get("content", ""))
            record.title = content[:60] + ("..." if len(content) > 60 else "")

    def save_pending_run(
        self,
        run_id: str,
        conversation_id: str,
        settings: AgentSettings,
        state: dict[str, Any],
    ) -> None:
        self.pending_runs[run_id] = PendingRun(
            run_id=run_id,
            conversation_id=conversation_id,
            settings=settings,
            state=state,
        )

    def pop_pending_run(self, run_id: str) -> PendingRun | None:
        return self.pending_runs.pop(run_id, None)

    def get_pending_run(self, run_id: str) -> PendingRun | None:
        return self.pending_runs.get(run_id)


run_store = RunStore()
