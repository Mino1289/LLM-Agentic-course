"""Pydantic schemas for tool arguments."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class SecFilingsRAGArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for SEC filings / earnings calls.")
    tickers: list[str] | None = Field(default=None, description="Tracked tickers filter.")
    years: list[str] | None = Field(default=None, description="Filing years filter, e.g. ['2024'].")
    doc_types: list[str] | None = Field(default=None, description="Allowed: 10-K, 10-Q, 8-K, 20-F, 6-K, EARNINGS_CALL.")

    @field_validator("tickers", "years", "doc_types", mode="before")
    @classmethod
    def _coerce_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class MarketPriceArgs(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class ValidateClaimsLLMArgs(BaseModel):
    claims: list[str] = Field(..., min_length=1)


class ValidateClaimsArgs(BaseModel):
    claims: list[str] = Field(..., min_length=1)
    chunks: list[str] = Field(default_factory=list)
    metadatas: list[dict[str, Any]] = Field(default_factory=list)


class PortfolioInfoArgs(BaseModel):
    pass


class PlaceTradeArgs(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker to trade.")
    side: Literal["buy", "sell"] = Field(..., description="Buy or sell.")
    qty: float = Field(..., gt=0, description="Number of shares.")
    order_type: Literal["market", "limit", "stop", "stop_limit"] | None = Field(default="market")
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)


class ClosePositionArgs(BaseModel):
    ticker: str | None = Field(default=None, min_length=1)
    all: bool = Field(default=False, description="Close all positions.")


class GetNewsArgs(BaseModel):
    symbols: list[str] = Field(..., min_length=1)
    start: str | None = Field(default=None)
    end: str | None = Field(default=None)
    limit: int = Field(default=10, gt=0, le=50)
    include_content: bool = Field(default=False)


class PortfolioHistoryArgs(BaseModel):
    period: str | None = Field(default="1M")
    timeframe: str | None = Field(default=None)
    extended_hours: bool = Field(default=False)
    start: str | None = Field(default=None)
    end: str | None = Field(default=None)


class AccountActivityArgs(BaseModel):
    activity_types: list[str] | None = Field(default=None)
    date: str | None = Field(default=None)
    after: str | None = Field(default=None)
    until: str | None = Field(default=None)
    page_size: int = Field(default=20, gt=0, le=100)
    direction: Literal["asc", "desc"] | None = Field(default="desc")

    @field_validator("activity_types", mode="before")
    @classmethod
    def _coerce_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class ExportReportArgs(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    format: Literal["md", "pdf"] = "md"


__all__ = [
    "SecFilingsRAGArgs", "MarketPriceArgs", "ValidateClaimsLLMArgs", "ValidateClaimsArgs",
    "PortfolioInfoArgs", "PlaceTradeArgs", "ClosePositionArgs", "GetNewsArgs",
    "PortfolioHistoryArgs", "AccountActivityArgs", "ExportReportArgs",
]
