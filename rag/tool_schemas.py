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


class SimulatePortfolioArgs(BaseModel):
    allocations: dict[str, float] = Field(..., min_length=1)
    notional_usd: float = Field(default=100_000, gt=0, le=1_000_000)


class ExportReportArgs(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    format: Literal["md", "pdf"] = "md"


__all__ = [
    "SecFilingsRAGArgs",
    "MarketPriceArgs",
    "ValidateClaimsLLMArgs",
    "ValidateClaimsArgs",
    "SimulatePortfolioArgs",
    "ExportReportArgs",
]
