"""Tool definitions for LLM function-calling — maps names → schemas + descriptions."""

from __future__ import annotations

from typing import Any

from src.tools.schemas import ValidateClaimsLLMArgs
from src.tools.descriptions import (
    SEC_FILINGS_RAG_DESCRIPTION,
    MARKET_PRICE_DESCRIPTION,
    EXPORT_REPORT_DESCRIPTION,
    VALIDATE_CLAIMS_DESCRIPTION,
    PORTFOLIO_INFO_DESCRIPTION,
    PLACE_TRADE_DESCRIPTION,
    CLOSE_POSITION_DESCRIPTION,
    GET_NEWS_DESCRIPTION,
    PORTFOLIO_HISTORY_DESCRIPTION,
    ACCOUNT_ACTIVITY_DESCRIPTION,
    _tracked_tickers_text,
)


def get_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "sec_filings_rag_tool",
                "description": SEC_FILINGS_RAG_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for filings/transcripts.",
                        },
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Optional tickers: {_tracked_tickers_text()}.",
                        },
                        "years": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional filing years, e.g. ['2024'].",
                        },
                        "doc_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: 10-K, 10-Q, 8-K, 20-F, 6-K, EARNINGS_CALL.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "market_price_tool",
                "description": MARKET_PRICE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tickers to fetch.",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date YYYY-MM-DD.",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date YYYY-MM-DD.",
                        },
                    },
                    "required": ["tickers", "start_date", "end_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_investment_report_tool",
                "description": EXPORT_REPORT_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Report title."},
                        "content": {
                            "type": "string",
                            "description": "Full report body (Markdown).",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["md", "pdf"],
                            "description": "Output format.",
                        },
                    },
                    "required": ["title", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_claims_tool",
                "description": VALIDATE_CLAIMS_DESCRIPTION,
                "parameters": ValidateClaimsLLMArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_info_tool",
                "description": PORTFOLIO_INFO_DESCRIPTION,
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "place_trade_tool",
                "description": PLACE_TRADE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": f"Ticker: {_tracked_tickers_text()}.",
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Buy or sell.",
                        },
                        "qty": {"type": "number", "description": "Number of shares."},
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop", "stop_limit"],
                            "description": "Default: market.",
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "Limit price.",
                        },
                        "stop_price": {"type": "number", "description": "Stop price."},
                    },
                    "required": ["ticker", "side", "qty"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_position_tool",
                "description": CLOSE_POSITION_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker to close (empty if all=true).",
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Liquidate all positions.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_news_tool",
                "description": GET_NEWS_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tickers.",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date YYYY-MM-DD.",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date YYYY-MM-DD.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max articles (default 10, max 50).",
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Include full content.",
                        },
                    },
                    "required": ["symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_history_tool",
                "description": PORTFOLIO_HISTORY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "description": "1D, 1W, 1M (default), 1A.",
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "1Min, 5Min, 15Min, 1H, 1D.",
                        },
                        "extended_hours": {
                            "type": "boolean",
                            "description": "Include extended hours.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "account_activity_tool",
                "description": ACCOUNT_ACTIVITY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "activity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter: FILL, DIV, CSD, CSW, INT, FEE, etc.",
                        },
                        "date": {"type": "string", "description": "Date YYYY-MM-DD."},
                        "after": {"type": "string", "description": "After YYYY-MM-DD."},
                        "until": {
                            "type": "string",
                            "description": "Before YYYY-MM-DD.",
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Max entries (default 20, max 100).",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "description": "Sort direction.",
                        },
                    },
                },
            },
        },
    ]
