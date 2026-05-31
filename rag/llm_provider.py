from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

from openai import OpenAI


SUPPORTED_PROVIDERS = {"openai", "github_models"}
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"


@dataclass
class LLMConfig:
    provider: str
    chat_model: str
    embedding_model: str
    api_key: str
    base_url: Optional[str] = None


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_llm_config_from_env() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. Expected one of: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        return LLMConfig(
            provider=provider,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        )

    api_key = _require_env("GITHUB_MODELS_API_KEY")
    return LLMConfig(
        provider=provider,
        chat_model=os.getenv("GITHUB_CHAT_MODEL", "openai/gpt-4o-mini"),
        embedding_model=os.getenv("GITHUB_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key,
        base_url=os.getenv("GITHUB_MODELS_BASE_URL", GITHUB_MODELS_BASE_URL).strip(),
    )


class LLMProvider:
    """
    Unified provider wrapper for OpenAI and GitHub Models endpoint.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or build_llm_config_from_env()
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(**client_kwargs)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        payload = [text for text in texts if text and text.strip()]
        if not payload:
            return []
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=payload,
        )
        return [item.embedding for item in response.data]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Lightweight estimate to avoid a paid token counting call on every turn.
        return max(1, len(text) // 4)
