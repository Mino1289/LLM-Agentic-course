"""Client Azure OpenAI pour les appels LLM."""

from __future__ import annotations

from typing import Any, Optional

from openai import AzureOpenAI, AsyncAzureOpenAI

from src.llm.types import LLMToolResponse, LLMStreamChunk
from src.llm.openai_client import OpenAIClient
from src.llm.messages import _openai_messages_to_api
from src.llm.sinks import _call_token_sink
from src.llm.parser import _parse_openai_tool_calls


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI utilise le même protocole OpenAI mais avec AzureOpenAI / AsyncAzureOpenAI."""

    def _init_sync(self):
        if self.client is None:
            self.client = AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version or "2024-02-01",
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

    def _init_async(self):
        if self.async_client is None:
            self.async_client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version or "2024-02-01",
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
