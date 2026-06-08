from __future__ import annotations

from dataclasses import dataclass, field

from toon_format import encode as toon_encode


@dataclass
class ConversationMemory:
    summary: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)
    last_chunk_fingerprints: set[str] = field(default_factory=set)


class MemoryStore:
    def __init__(self, window_size: int = 6):
        self.window_size = max(2, window_size)
        self._store: dict[str, ConversationMemory] = {}

    def get_or_create(self, conversation_id: str) -> ConversationMemory:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationMemory()
        return self._store[conversation_id]

    def get_window(self, conversation_id: str) -> list[dict[str, str]]:
        memory = self.get_or_create(conversation_id)
        return memory.turns[-self.window_size :]

    def append_turn(self, conversation_id: str, role: str, content: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.turns.append({"role": role, "content": content.strip()})

    def update_summary(self, conversation_id: str, new_summary: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.summary = new_summary.strip()

    def get_summary(self, conversation_id: str) -> str:
        return self.get_or_create(conversation_id).summary

    def trim_turns(self, conversation_id: str, keep_last: int) -> None:
        memory = self.get_or_create(conversation_id)
        memory.turns = memory.turns[-max(1, keep_last) :]

    def is_duplicate_chunk(self, conversation_id: str, chunk: str) -> bool:
        memory = self.get_or_create(conversation_id)
        fingerprint = chunk[:220].strip()
        return fingerprint in memory.last_chunk_fingerprints

    def remember_chunk(self, conversation_id: str, chunk: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.last_chunk_fingerprints.add(chunk[:220].strip())
        if len(memory.last_chunk_fingerprints) > 60:
            memory.last_chunk_fingerprints = set(list(memory.last_chunk_fingerprints)[-40:])


def format_memory_context(summary: str, window: list[dict[str, str]]) -> str:
    """Serialize memory context (summary + recent turns) as a TOON payload.

    Saves ~30% tokens vs the legacy "Resume memoire: ...\\n\\nDerniers echanges: ..."
    text format. The structure is roundtrippable: toon_format.decode() returns
    {"summary": str, "turns": [{"role", "content"}, ...]}.

    Empty input (no summary, no window) returns a French fallback verbatim
    (it's a prompt-engineering signal, not a payload).
    """
    if not summary and not window:
        return "Aucun contexte memorise."
    return toon_encode({"summary": summary, "turns": list(window)})


def format_chat_context(messages: list[dict[str, str]], keep_last: int = 6) -> str:
    """Serialize recent chat history as a TOON payload.

    Same shape as format_memory_context but for raw multi-turn chat (no summary).
    Empty history returns a French fallback verbatim.
    """
    if not messages:
        return "Aucun historique de chat."
    selected = messages[-keep_last:]
    return toon_encode({"turns": [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in selected
    ]})
