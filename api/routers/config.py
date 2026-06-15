from __future__ import annotations

from fastapi import APIRouter

from api.schemas.settings import AgentSettings, ConfigResponse
from src.llm import build_llm_config_from_env

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    llm_config = build_llm_config_from_env()
    return ConfigResponse(
        chat_provider=llm_config.provider,
        chat_model=llm_config.chat_model,
        embedding_provider=llm_config.embedding_provider,
        embedding_model=llm_config.embedding_model,
        defaults=AgentSettings.from_env_defaults(),
    )
