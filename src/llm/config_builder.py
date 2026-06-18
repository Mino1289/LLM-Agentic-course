"""Construction de la configuration LLM depuis les variables d'environnement."""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.llm.types import LLMConfig

_LOGGER = logging.getLogger("src.llm.config_builder")

SUPPORTED_CHAT_PROVIDERS = {
    "openai",
    "github_models",
    "gemini",
    "azure_openai",
    "nvidia_nim",
}
SUPPORTED_EMBEDDING_PROVIDERS = {
    "openai",
    "github_models",
    "azure_openai",
    "nvidia_nim",
}
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
GITHUB_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
AZURE_DEFAULT_API_VERSION = "2024-02-01"
AZURE_DEFAULT_CHAT_DEPLOYMENT = "gpt-4o-mini"
AZURE_DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_DEFAULT_CHAT_MODEL = "deepseek-ai/deepseek-v4-flash"
NVIDIA_NIM_DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
# Backward-compatible alias
SUPPORTED_PROVIDERS = SUPPORTED_CHAT_PROVIDERS


def _normalize_github_model_id(model: str) -> str:
    """GitHub Models inference uses IDs like gpt-4.1-mini, not openai/gpt-4.1-mini."""
    model = model.strip()
    for prefix in ("openai/", "azure-openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def mask_api_key(value: str | None, visible_suffix: int = 6) -> str:
    raw = (value or "").strip()
    if not raw:
        return "(unset)"
    if len(raw) <= visible_suffix:
        return "***"
    return f"...{raw[-visible_suffix:]}"


def _log_effective_config(provider: str, chat_model: str, api_key: str) -> None:
    key_label = {
        "openai": "OPENAI_API_KEY",
        "github_models": "GITHUB_MODELS_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
    }.get(provider, "API_KEY")
    _LOGGER.info(
        "LLM config: provider=%s model=%s %s=%s",
        provider,
        chat_model,
        key_label,
        mask_api_key(api_key),
    )


def _build_openai_style_embedding_config(
    provider: str,
) -> tuple[str, str, Optional[str]]:
    if provider == "openai":
        return (
            os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            _require_env("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL", "").strip() or OPENAI_DEFAULT_BASE_URL,
        )
    if provider == "nvidia_nim":
        return (
            os.getenv("NVIDIA_NIM_EMBEDDING_MODEL", NVIDIA_NIM_DEFAULT_EMBEDDING_MODEL),
            _require_env("NVIDIA_NIM_API_KEY"),
            os.getenv("NVIDIA_NIM_BASE_URL", "").strip() or NVIDIA_NIM_BASE_URL,
        )
    return (
        _normalize_github_model_id(
            os.getenv("GITHUB_EMBEDDING_MODEL", GITHUB_DEFAULT_EMBEDDING_MODEL)
        ),
        _require_env("GITHUB_MODELS_API_KEY"),
        os.getenv("GITHUB_MODELS_BASE_URL", GITHUB_MODELS_BASE_URL).strip(),
    )


def _build_azure_openai_embedding_config() -> tuple[str, str, str, str]:
    return (
        os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", AZURE_DEFAULT_EMBEDDING_DEPLOYMENT
        ),
        _require_env("AZURE_OPENAI_API_KEY"),
        _require_env("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_API_VERSION", AZURE_DEFAULT_API_VERSION),
    )


def build_llm_config_from_env() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_CHAT_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            f"Expected one of: {sorted(SUPPORTED_CHAT_PROVIDERS)}"
        )

    default_embedding = "openai" if provider == "gemini" else provider
    embedding_provider = (
        os.getenv("EMBEDDING_PROVIDER", default_embedding).strip().lower()
    )
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER='{embedding_provider}'. "
            f"Expected one of: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
        )

    if embedding_provider == "azure_openai":
        embed_model, embed_key, embed_base, embed_api_ver = (
            _build_azure_openai_embedding_config()
        )
    else:
        embed_model, embed_key, embed_base = _build_openai_style_embedding_config(
            embedding_provider
        )
        embed_api_ver = None

    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        _log_effective_config(provider, chat_model, api_key)
        raw_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        return LLMConfig(
            provider=provider,
            chat_model=chat_model,
            embedding_model=embed_model,
            api_key=api_key,
            base_url=raw_base_url or OPENAI_DEFAULT_BASE_URL,
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
            embedding_api_version=embed_api_ver,
        )

    if provider == "github_models":
        api_key = _require_env("GITHUB_MODELS_API_KEY")
        chat_model = _normalize_github_model_id(
            os.getenv("GITHUB_CHAT_MODEL", GITHUB_DEFAULT_CHAT_MODEL)
        )
        _log_effective_config(provider, chat_model, api_key)
        return LLMConfig(
            provider=provider,
            chat_model=chat_model,
            embedding_model=embed_model,
            api_key=api_key,
            base_url=os.getenv(
                "GITHUB_MODELS_BASE_URL", GITHUB_MODELS_BASE_URL
            ).strip(),
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
            embedding_api_version=embed_api_ver,
        )

    if provider == "azure_openai":
        api_key = _require_env("AZURE_OPENAI_API_KEY")
        endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
        api_ver = os.getenv("AZURE_OPENAI_API_VERSION", AZURE_DEFAULT_API_VERSION)
        chat_model = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", AZURE_DEFAULT_CHAT_DEPLOYMENT
        )
        _log_effective_config(provider, chat_model, api_key)
        return LLMConfig(
            provider=provider,
            chat_model=chat_model,
            embedding_model=embed_model,
            api_key=api_key,
            base_url=endpoint,
            api_version=api_ver,
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
            embedding_api_version=embed_api_ver,
        )

    if provider == "nvidia_nim":
        api_key = _require_env("NVIDIA_NIM_API_KEY")
        chat_model = os.getenv("NVIDIA_NIM_CHAT_MODEL", NVIDIA_NIM_DEFAULT_CHAT_MODEL)
        _log_effective_config(provider, chat_model, api_key)
        return LLMConfig(
            provider=provider,
            chat_model=chat_model,
            embedding_model=embed_model,
            api_key=api_key,
            base_url=os.getenv("NVIDIA_NIM_BASE_URL", "").strip()
            or NVIDIA_NIM_BASE_URL,
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
            embedding_api_version=embed_api_ver,
        )

    api_key = _require_env("GEMINI_API_KEY")
    chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    _log_effective_config(provider, chat_model, api_key)
    return LLMConfig(
        provider=provider,
        chat_model=chat_model,
        embedding_model=embed_model,
        api_key=api_key,
        base_url=None,
        embedding_provider=embedding_provider,
        embedding_api_key=embed_key,
        embedding_base_url=embed_base,
        embedding_api_version=embed_api_ver,
    )
