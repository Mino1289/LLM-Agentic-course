from __future__ import annotations

from fastapi import APIRouter

from api.schemas.settings import AgentSettings, ConfigResponse
from src.llm import build_llm_config_from_env
from src.llm.config_builder import mask_api_key

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    llm_config = build_llm_config_from_env()
    return ConfigResponse(
        chat_provider=llm_config.provider,
        chat_model=llm_config.chat_model,
        embedding_provider=llm_config.embedding_provider,
        embedding_model=llm_config.embedding_model,
        chat_api_key_suffix=mask_api_key(llm_config.api_key),
        defaults=AgentSettings.from_env_defaults(),
    )
