from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from contextlib import contextmanager, redirect_stdout
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from rag.hybrid_rag import HybridRAG
from rag.langgraph_flow import FinanceLangGraphAgent
from rag.llm_provider import ToolCall
from rag.paths import ENV_FILE
from rag.tool_executor import ToolExecutor
from rag.tools import (
    ACCOUNT_ACTIVITY_DESCRIPTION,
    CLOSE_POSITION_DESCRIPTION,
    EXPORT_REPORT_DESCRIPTION,
    GET_NEWS_DESCRIPTION,
    MARKET_PRICE_DESCRIPTION,
    PLACE_TRADE_DESCRIPTION,
    PORTFOLIO_HISTORY_DESCRIPTION,
    PORTFOLIO_INFO_DESCRIPTION,
    SEC_FILINGS_RAG_DESCRIPTION,
    VALIDATE_CLAIMS_DESCRIPTION,
)


SERVER_NAME = "finance-rag-mcp"


@contextmanager
def app_stdout_to_stderr():
    """Keep MCP stdio clean.

    MCP stdio reserves stdout for JSON-RPC frames. The existing RAG/indexing
    code prints progress messages, so MCP clients fail to parse responses if
    those messages leak to stdout. Redirect application prints to stderr while
    tools run.
    """
    with redirect_stdout(sys.stderr):
        yield


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_agent() -> FinanceLangGraphAgent:
    """Build the shared runtime used by MCP tools.

    The vector store is opened read-only for practical purposes:
    ``max_new_embeddings=0`` loads existing Chroma data and removes stale
    entries, but never starts costly embedding calls from the MCP server.
    """
    with app_stdout_to_stderr():
        load_dotenv(ENV_FILE)
        rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
        rag.load_and_index_data(max_new_embeddings=0)
    return FinanceLangGraphAgent(
        rag=rag,
        memory_window_size=_env_int("MEMORY_WINDOW_SIZE", 6),
        summarize_every_n_turns=_env_int("SUMMARIZE_EVERY_N_TURNS", 6),
        max_context_chunks=_env_int("MAX_CONTEXT_CHUNKS", 8),
        max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 3500),
        decompose_query_count=_env_int("QUERY_DECOMPOSE_COUNT", 2),
        price_max_days=_env_int("PRICE_MAX_DAYS", 180),
        price_max_points=_env_int("PRICE_MAX_POINTS", 40),
        price_max_tickers=_env_int("PRICE_MAX_TICKERS", 3),
        price_default_days=_env_int("PRICE_DEFAULT_DAYS", 90),
        price_max_attempts=_env_int("PRICE_MAX_ATTEMPTS", 2),
        max_tool_iterations=_env_int("MAX_TOOL_ITERATIONS", 6),
    )


async def execute_mcp_tool(
    name: str,
    args: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute an existing project tool through the same ToolExecutor as LangGraph."""
    tool_call = ToolCall(
        id=f"mcp-{uuid.uuid4()}",
        name=name,
        arguments=json.dumps(args, ensure_ascii=False),
    )
    executor = ToolExecutor(agent=get_agent(), state=state or {})
    with app_stdout_to_stderr():
        outcome = await executor.execute(tool_call)
    payload: dict[str, Any] = {
        "ok": outcome.event.get("status") == "completed",
        "tool": name,
        "event": outcome.event,
    }
    if outcome.result:
        payload.update(outcome.result)
    else:
        payload["text"] = outcome.message.get("content", "")
    return payload


def create_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Le package MCP n'est pas installé. Lance: "
            ".venv/bin/pip install -r requirements.txt"
        ) from exc

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool(description=SEC_FILINGS_RAG_DESCRIPTION)
    async def sec_filings_rag_tool(
        query: str,
        tickers: list[str] | None = None,
        years: list[str] | None = None,
        doc_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search the local finance RAG over SEC filings and earnings calls."""
        return await execute_mcp_tool(
            "sec_filings_rag_tool",
            {
                "query": query,
                "tickers": tickers,
                "years": years,
                "doc_types": doc_types,
            },
        )

    @mcp.tool(description=MARKET_PRICE_DESCRIPTION)
    async def market_price_tool(
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Fetch market-price context for tracked tickers."""
        return await execute_mcp_tool(
            "market_price_tool",
            {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    @mcp.tool(description=VALIDATE_CLAIMS_DESCRIPTION)
    async def validate_claims_tool(
        claims: list[str],
        chunks: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate claims against provided RAG chunks.

        MCP calls are stateless, so callers should pass chunks/metadatas from a
        previous ``sec_filings_rag_tool`` response when they want grounded
        validation.
        """
        return await execute_mcp_tool(
            "validate_claims_tool",
            {"claims": claims},
            state={
                "final_chunks": chunks or [],
                "final_metadatas": metadatas or [],
            },
        )

    @mcp.tool(description=PORTFOLIO_INFO_DESCRIPTION)
    async def portfolio_info_tool() -> dict[str, Any]:
        """View Alpaca paper account info and open positions."""
        return await execute_mcp_tool("portfolio_info_tool", {})

    @mcp.tool(description=PLACE_TRADE_DESCRIPTION)
    async def place_trade_tool(
        ticker: str,
        side: str,
        qty: float,
        order_type: str | None = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        """Submit a paper trade on Alpaca."""
        return await execute_mcp_tool(
            "place_trade_tool",
            {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "limit_price": limit_price,
                "stop_price": stop_price,
            },
        )

    @mcp.tool(description=CLOSE_POSITION_DESCRIPTION)
    async def close_position_tool(
        ticker: str | None = None,
        all: bool = False,
    ) -> dict[str, Any]:
        """Close a position or liquidate all on Alpaca paper."""
        return await execute_mcp_tool(
            "close_position_tool",
            {
                "ticker": ticker,
                "all": all,
            },
        )

    @mcp.tool(description=GET_NEWS_DESCRIPTION)
    async def get_news_tool(
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Fetch news articles for one or more tickers."""
        return await execute_mcp_tool(
            "get_news_tool",
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "limit": limit,
                "include_content": include_content,
            },
        )

    @mcp.tool(description=PORTFOLIO_HISTORY_DESCRIPTION)
    async def portfolio_history_tool(
        period: str | None = "1M",
        timeframe: str | None = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """Get equity and P&L history for your Alpaca paper account."""
        return await execute_mcp_tool(
            "portfolio_history_tool",
            {
                "period": period,
                "timeframe": timeframe,
                "extended_hours": extended_hours,
            },
        )

    @mcp.tool(description=ACCOUNT_ACTIVITY_DESCRIPTION)
    async def account_activity_tool(
        activity_types: list[str] | None = None,
        date: str | None = None,
        after: str | None = None,
        until: str | None = None,
        page_size: int = 20,
        direction: str | None = "desc",
    ) -> dict[str, Any]:
        """Retrieve account activity (fills, dividends, transfers, fees)."""
        return await execute_mcp_tool(
            "account_activity_tool",
            {
                "activity_types": activity_types,
                "date": date,
                "after": after,
                "until": until,
                "page_size": page_size,
                "direction": direction,
            },
        )

    @mcp.tool(description=EXPORT_REPORT_DESCRIPTION)
    async def export_investment_report_tool(
        title: str,
        content: str,
        format: str = "md",
    ) -> dict[str, Any]:
        """Save an investment report in reports/."""
        return await execute_mcp_tool(
            "export_investment_report_tool",
            {
                "title": title,
                "content": content,
                "format": format,
            },
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG MCP server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse"],
        help="MCP transport. stdio is the default and most portable option.",
    )
    args = parser.parse_args()

    server = create_mcp_server()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
