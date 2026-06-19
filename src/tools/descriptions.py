"""Descriptions et normalisation pour les outils SEC / marchés / Alpaca."""

from __future__ import annotations

import re
from typing import Any

from src.config import TRACKED_TICKERS
from toon_format import encode as toon_encode

ALLOWED_DOC_TYPES = ["10-K", "10-Q", "8-K", "20-F", "6-K", "EARNINGS_CALL"]


def _tracked_tickers_text() -> str:
    return ", ".join(TRACKED_TICKERS)


def _normalize_doc_types(doc_types: Any) -> list[str]:
    if doc_types is None:
        return []
    if isinstance(doc_types, str):
        raw = [part.strip() for part in re.split(r"[,|]", doc_types) if part.strip()]
    elif isinstance(doc_types, list):
        raw = [str(item).strip() for item in doc_types]
    else:
        return []
    normalized: list[str] = []
    for item in raw:
        up = item.upper().replace(" ", "_")
        if up in {"EARNINGSCALL", "EARNINGS-CALL"}:
            up = "EARNINGS_CALL"
        if up in ALLOWED_DOC_TYPES and up not in normalized:
            normalized.append(up)
    return normalized


def _normalize_tickers(tickers: Any) -> list[str]:
    allowed = set(TRACKED_TICKERS)
    if tickers is None:
        return []
    if isinstance(tickers, str):
        raw = re.split(r"[,|\s]+", tickers.strip())
    elif isinstance(tickers, list):
        raw = [str(t).strip() for t in tickers]
    else:
        return []
    result: list[str] = []
    for token in raw:
        up = token.upper()
        if up in allowed and up not in result:
            result.append(up)
    return result


def _normalize_years(years: Any) -> list[str]:
    if years is None:
        return []
    if isinstance(years, (int, float)):
        return [str(int(years))]
    if isinstance(years, str):
        raw = re.split(r"[,|\s]+", years.strip())
    elif isinstance(years, list):
        raw = [str(y) for y in years]
    else:
        return []
    result: list[str] = []
    for token in raw:
        match = re.search(r"(20\d{2})", token)
        if match and match.group(1) not in result:
            result.append(match.group(1))
    return result


SEC_FILINGS_RAG_DESCRIPTION = (
    "Search SEC filings and earnings call transcripts in ChromaDB. "
    "Filter by document type: 10-K (annual), 10-Q (quarterly), 8-K (events), "
    "20-F (foreign annual), 6-K (foreign interim), EARNINGS_CALL (conference call transcripts). "
    "Examples: 'MSFT 10-K risk factors 2024', 'ASML 20-F risk factors 2024'."
)

MARKET_PRICE_DESCRIPTION = (
    f"Fetch stock price performance for tracked tickers ({_tracked_tickers_text()}) "
    "between start_date and end_date (YYYY-MM-DD). "
    "Use a WIDER date range (at least 5 trading days) to ensure data is returned."
)

EXPORT_REPORT_DESCRIPTION = (
    "Save an investment report to the reports/ folder. "
    "Supported formats: md (default) and pdf. Returns the file path."
)

VALIDATE_CLAIMS_DESCRIPTION = (
    "Check whether factual claims are supported by RAG excerpts already retrieved. "
    "Call after sec_filings_rag_tool. Returns supported/partial/unsupported per claim with source refs."
)

PORTFOLIO_INFO_DESCRIPTION = (
    "View your Alpaca paper trading account: cash balance, equity, buying power, "
    f"unrealized P&L, and all open positions. Restricted to {_tracked_tickers_text()}."
)

PLACE_TRADE_DESCRIPTION = (
    "Submit a real paper trade on Alpaca. "
    f"Support tickers: {_tracked_tickers_text()}. "
    "Supports market, limit, stop, and stop-limit orders. "
    "Max notional per order: $10,000."
)

CLOSE_POSITION_DESCRIPTION = (
    "Close a specific position by ticker or liquidate all positions "
    "on your Alpaca paper trading account."
)

GET_NEWS_DESCRIPTION = (
    "Fetch latest news articles for one or more tickers from Alpaca News API. "
    f"Supports {_tracked_tickers_text()} and other US-traded symbols. "
    "Returns headlines, source, summary, date, and URL."
)

PORTFOLIO_HISTORY_DESCRIPTION = (
    "Get equity and P&L history for your Alpaca paper account over a period. "
    "Defaults to 1 month. Supports 1D, 1W, 1M, 1A periods with configurable timeframe."
)

ACCOUNT_ACTIVITY_DESCRIPTION = (
    "Retrieve account activity (fills, dividends, deposits, withdrawals, fees) "
    "from your Alpaca paper account. Filter by activity type, date range, and sort direction."
)


def format_rag_excerpts(chunks: list[str], metadatas: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No matching SEC or earnings-call excerpts found for the given filters."
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        rows.append(
            {
                "i": idx + 1,
                "ticker": str(meta.get("ticker", "UNKNOWN")),
                "year": str(meta.get("year", "unknown")),
                "file_type": str(meta.get("file_type", "unknown")),
                "section": str(meta.get("section", "unknown")),
                "source": str(meta.get("source", "unknown")),
                "text": chunk[:1200],
            }
        )
    return toon_encode({"excerpts": rows})
