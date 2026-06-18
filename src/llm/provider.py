"""Agrégateur de providers LLM - Point d'entrée unique."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional, Generator

from src.llm.types import LLMConfig, LLMToolResponse, LLMStreamChunk
from src.llm.config_builder import build_llm_config_from_env
from src.llm.openai_client import OpenAIClient
from src.llm.github_client import GitHubModelsClient
from src.llm.azure_client import AzureOpenAIClient
from src.llm.nvidia_nim_client import NvidiaNIMClient
from src.llm.gemini_client import GeminiClient


class LLMProvider:
    """
    Unified provider wrapper for OpenAI, GitHub Models, Azure OpenAI, NVIDIA NIM, and Gemini.
    Embedding and chat providers can be selected independently.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or build_llm_config_from_env()
        self._client = None
        self._gemini_client = None
        self.async_client = None

        # Request timeout + max retries for all OpenAI-compatible clients.
        self.request_timeout = self._parse_request_timeout()
        self.max_retries = self._parse_max_retries()

        # Initialize embedding client
        if self.config.embedding_provider == "azure_openai":
            from openai import AzureOpenAI

            self.embedding_client = AzureOpenAI(
                api_key=self.config.embedding_api_key or self.config.api_key,
                azure_endpoint=self.config.embedding_base_url,
                api_version=self.config.embedding_api_version or "2024-02-01",
            )
        else:
            embed_kwargs = {
                "api_key": self.config.embedding_api_key or self.config.api_key
            }
            if self.config.embedding_base_url:
                embed_kwargs["base_url"] = self.config.embedding_base_url
            embed_kwargs["timeout"] = self.request_timeout
            embed_kwargs["max_retries"] = self.max_retries
            from openai import OpenAI

            self.embedding_client = OpenAI(**embed_kwargs)

        if self.config.provider == "openai":
            self._client = OpenAIClient(self.config)
        elif self.config.provider == "github_models":
            self._client = GitHubModelsClient(self.config)
        elif self.config.provider == "azure_openai":
            self._client = AzureOpenAIClient(self.config)
        elif self.config.provider == "nvidia_nim":
            self._client = NvidiaNIMClient(self.config)
        else:
            self._client = None

        self._init_async_client()

    @staticmethod
    def _parse_request_timeout() -> float:
        raw = os.getenv("OPENAI_REQUEST_TIMEOUT", "30.0")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 30.0
        if value <= 0:
            return 30.0
        return value

    @staticmethod
    def _parse_max_retries() -> int:
        raw = os.getenv("OPENAI_MAX_RETRIES", "2")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 2
        if value < 0:
            return 2
        return value

    def _init_async_client(self) -> None:
        """Initialize async client if needed."""
        if (
            self.config.provider
            in {"openai", "github_models", "azure_openai", "nvidia_nim"}
            and self._client
        ):
            self._client._init_async()

    def _get_gemini_client(self):
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(self.config)
        return self._gemini_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the configured embedding model."""
        payload = [text for text in texts if text and text.strip()]
        if not payload:
            return []
        cache = self._get_embedding_cache()
        unique_payload = list(dict.fromkeys(payload))
        cached: dict[str, list[float]] = {}
        uncached: list[str] = []
        for text in unique_payload:
            hit = cache.get(text)
            if hit is not None:
                cached[text] = hit
            else:
                uncached.append(text)
        if uncached:
            response = self.embedding_client.embeddings.create(
                model=self.config.embedding_model,
                input=uncached,
            )
            new_vectors = [item.embedding for item in response.data]
            cache.put_many(list(zip(uncached, new_vectors)))
            for text, vec in zip(uncached, new_vectors):
                cached[text] = vec
        return [cached[text] for text in payload]

    def _get_embedding_cache(self):
        """Lazy-load the persistent query-embedding cache."""
        cache = getattr(self, "_embedding_cache", None)
        if cache is None:
            from src.embeddings.cache import build_embedding_cache_from_env

            try:
                cache = build_embedding_cache_from_env()
            except Exception:
                from src.embeddings.cache import EmbeddingCache
                from src.paths import DATA_DIR

                cache = EmbeddingCache(DATA_DIR / "embedding_query_cache.json")
            self._embedding_cache = cache
        return cache

    def embedding_cache_stats(self) -> dict[str, int | str]:
        """Return hit/miss counters for the persistent query-embedding cache."""
        return self._get_embedding_cache().stats()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.invoke_with_tools(
            messages, tools=None, temperature=temperature, max_tokens=max_tokens
        )
        return response.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> Generator[str, None, None]:
        yield self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMToolResponse:
        if self.config.provider == "gemini":
            return self._get_gemini_client().invoke_with_tools(
                messages, tools, temperature, max_tokens
            )
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        return self._client.invoke_with_tools(messages, tools, temperature, max_tokens)

    async def ainvoke_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> AsyncIterator[LLMStreamChunk]:
        if self.config.provider == "gemini":
            async for chunk in self._get_gemini_client().ainvoke_with_tools_stream(
                messages, tools, temperature, max_tokens
            ):
                yield chunk
            return
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        async for chunk in self._client.ainvoke_with_tools_stream(
            messages, tools, temperature, max_tokens
        ):
            yield chunk

    async def agenerate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async for chunk in self.ainvoke_with_tools_stream(
            messages, tools=None, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.delta:
                yield chunk.delta

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)
