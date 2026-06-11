"""Pydantic BaseModel schemas for tool arguments (PRD etape 4 §3.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _split_csv(value: Any) -> Any:
    """Accept a comma-separated string in addition to list for LLM tolerance."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class SecFilingsRAGArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for SEC filings / earnings calls.")
    tickers: list[str] | None = Field(
        default=None, description="Tracked tickers filter: NVDA, ASML, AMD, ARM, MSFT."
    )
    years: list[str] | None = Field(
        default=None, description="Filing years filter, e.g. ['2024']."
    )
    doc_types: list[str] | None = Field(
        default=None, description="Allowed: 10-K, 10-Q, 8-K, 20-F, 6-K, EARNINGS_CALL."
    )

    @field_validator("tickers", "years", "doc_types", mode="before")
    @classmethod
    def _coerce_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class MarketPriceArgs(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class ValidateClaimsLLMArgs(BaseModel):
    """Schéma PUBLIC envoyé à OpenAI : le LLM ne fournit que 'claims'."""

    claims: list[str] = Field(..., min_length=1)


class ValidateClaimsArgs(BaseModel):
    """Modèle RUNTIME complet : chunks/metadatas sont injectés par tools_node."""

    claims: list[str] = Field(..., min_length=1)
    chunks: list[str] = Field(default_factory=list)
    metadatas: list[dict[str, Any]] = Field(default_factory=list)


class PortfolioInfoArgs(BaseModel):
    pass


class PlaceTradeArgs(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker to trade (NVDA, ASML, AMD, ARM, MSFT).")
    side: Literal["buy", "sell"] = Field(..., description="Buy or sell.")
    qty: float = Field(..., gt=0, description="Number of shares (can be fractional for paper).")
    order_type: Literal["market", "limit", "stop", "stop_limit"] | None = Field(
        default="market", description="Order type. Default market."
    )
    limit_price: float | None = Field(
        default=None, gt=0, description="Limit price (required for limit / stop_limit)."
    )
    stop_price: float | None = Field(
        default=None, gt=0, description="Stop price (required for stop / stop_limit)."
    )


class ClosePositionArgs(BaseModel):
    ticker: str | None = Field(default=None, min_length=1, description="Ticker to close.")
    all: bool = Field(default=False, description="Close all positions.")


class GetNewsArgs(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Tickers to get news for.")
    start: str | None = Field(default=None, description="Start date YYYY-MM-DD (optional).")
    end: str | None = Field(default=None, description="End date YYYY-MM-DD (optional).")
    limit: int = Field(default=10, gt=0, le=50, description="Max articles (default 10, max 50).")
    include_content: bool = Field(default=False, description="Include full article content.")


class PortfolioHistoryArgs(BaseModel):
    period: str | None = Field(default="1M", description="Duration: 1D, 1W, 1M (default), 1A.")
    timeframe: str | None = Field(default=None, description="Resolution: 1Min, 5Min, 15Min, 1H, 1D.")
    extended_hours: bool = Field(default=False, description="Include extended hours.")
    start: str | None = Field(default=None, description="Start timestamp (RFC3339).")
    end: str | None = Field(default=None, description="End timestamp (RFC3339).")


class AccountActivityArgs(BaseModel):
    activity_types: list[str] | None = Field(
        default=None,
        description="Filter: FILL, DIV, CSD, CSW, INT, FEE, etc. Comma-separated or list.",
    )
    date: str | None = Field(default=None, description="Date filter YYYY-MM-DD.")
    after: str | None = Field(default=None, description="Activities after this date YYYY-MM-DD.")
    until: str | None = Field(default=None, description="Activities before this date YYYY-MM-DD.")
    page_size: int = Field(default=20, gt=0, le=100, description="Max entries (default 20, max 100).")
    direction: Literal["asc", "desc"] | None = Field(default="desc", description="Sort direction.")

    @field_validator("activity_types", mode="before")
    @classmethod
    def _coerce_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class ExportReportArgs(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    format: Literal["md", "pdf"] = "md"


__all__ = [
    "SecFilingsRAGArgs",
    "MarketPriceArgs",
    "ValidateClaimsLLMArgs",
    "ValidateClaimsArgs",
    "PortfolioInfoArgs",
    "PlaceTradeArgs",
    "ClosePositionArgs",
    "GetNewsArgs",
    "PortfolioHistoryArgs",
    "AccountActivityArgs",
    "ExportReportArgs",
]
