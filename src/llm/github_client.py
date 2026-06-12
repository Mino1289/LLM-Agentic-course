"""Client GitHub Models pour les appels LLM."""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from src.llm.types import LLMToolResponse, LLMStreamChunk
from src.llm.openai_client import OpenAIClient


class GitHubModelsClient(OpenAIClient):
    """GitHub Models utilise le même protocole OpenAI, mais avec une base URL différente."""

    def __init__(self, config):
        super().__init__(config)

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMToolResponse:
        return super().invoke_with_tools(messages, tools, temperature, max_tokens)

    async def ainvoke_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> AsyncIterator[LLMStreamChunk]:
        async for chunk in super().ainvoke_with_tools_stream(messages, tools, temperature, max_tokens):
            yield chunk
