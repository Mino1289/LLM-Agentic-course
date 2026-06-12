"""Types fondamentaux pour les providers LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    thought_signature: Optional[str] = None


@dataclass
class LLMConfig:
    provider: str
    chat_model: str
    embedding_model: str
    api_key: str
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_base_url: Optional[str] = None
    embedding_api_version: Optional[str] = None


@dataclass
class LLMToolResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LLMStreamChunk:
    """Chunk incrémental d'un stream LLM.

    Soit un delta texte (delta != ""), soit un delta de tool_call
    (tool_call_delta != None), soit les deux. Un chunk final peut
    contenir finish_reason set ("stop", "tool_calls", "length").
    """

    delta: str = ""
    tool_call_delta: Optional[list[dict[str, Any]]] = None
    finish_reason: Optional[str] = None
