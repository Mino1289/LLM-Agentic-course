from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field


class AgentSettings(BaseModel):
    max_chunks: int = Field(default=8, alias="maxChunks", ge=4, le=12)
    sub_queries: int = Field(default=2, alias="subQueries", ge=1, le=8)
    price_max_days: int = Field(default=180, alias="priceMaxDays", ge=30, le=365)
    price_max_points: int = Field(default=40, alias="priceMaxPoints", ge=10, le=120)
    price_max_tickers: int = Field(default=3, alias="priceMaxTickers", ge=1, le=5)
    price_default_window: int = Field(
        default=90, alias="priceDefaultWindow", ge=15, le=180
    )
    max_iterations: int = Field(default=6, alias="maxIterations", ge=2, le=10)
    max_spoke_iterations: int = Field(default=3, alias="maxSpokeIterations", ge=1, le=5)

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    @classmethod
    def from_env_defaults(cls) -> "AgentSettings":
        return cls(
            maxChunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "8")),
            subQueries=int(os.getenv("QUERY_DECOMPOSE_COUNT", "2")),
            priceMaxDays=int(os.getenv("PRICE_MAX_DAYS", "180")),
            priceMaxPoints=int(os.getenv("PRICE_MAX_POINTS", "40")),
            priceMaxTickers=int(os.getenv("PRICE_MAX_TICKERS", "3")),
            priceDefaultWindow=int(os.getenv("PRICE_DEFAULT_DAYS", "90")),
            maxIterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
        )

    def to_agent_kwargs(self) -> dict:
        return {
            "max_context_chunks": self.max_chunks,
            "decompose_query_count": self.sub_queries,
            "price_max_days": self.price_max_days,
            "price_max_points": self.price_max_points,
            "price_max_tickers": self.price_max_tickers,
            "price_default_days": self.price_default_window,
            "max_tool_iterations": self.max_iterations,
        }

    def cache_key(self) -> str:
        return self.model_dump_json(by_alias=True)


class ConfigResponse(BaseModel):
    chat_provider: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    chat_api_key_suffix: str | None = Field(default=None, alias="chatApiKeySuffix")
    defaults: AgentSettings

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
