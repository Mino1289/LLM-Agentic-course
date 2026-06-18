"""Tool dispatch — execute_tool() dispatcher + ToolExecutor class."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm.types import ToolCall
from src.tools.schemas import (
    AccountActivityArgs,
    ClosePositionArgs,
    ExportReportArgs,
    GetNewsArgs,
    MarketPriceArgs,
    PlaceTradeArgs,
    PortfolioHistoryArgs,
    PortfolioInfoArgs,
    SecFilingsRAGArgs,
    ValidateClaimsArgs,
    ValidateClaimsLLMArgs,
)
from src.tools.sec_filings import run_sec_filings_rag
from src.tools.market_price import run_market_price_tool
from src.tools.validate_claims import run_validate_claims
from src.tools.export_report import run_export_investment_report
from src.tools.portfolio import (
    run_portfolio_info,
    run_portfolio_history,
    run_account_activity,
)
from src.tools.trading import run_place_trade, run_close_position
from src.tools.news import run_get_news

_LOGGER = logging.getLogger("src.tools.execute")

# Import legacy GraphState/ToolEvent until src/graph/ is migrated
from src.graph.state import GraphState, ToolEvent  # noqa: E402


@dataclass
class ToolExecutionOutcome:
    tool_call: ToolCall
    llm_args: dict[str, Any]
    event: ToolEvent
    message: dict[str, Any]
    result: dict[str, Any] | None


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def summarize_tool_args(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments[:120]
    summaries = {
        "sec_filings_rag_tool": f"query={args.get('query', '')!r}, tickers={args.get('tickers')}, years={args.get('years')}, doc_types={args.get('doc_types')}",
        "market_price_tool": f"tickers={args.get('tickers')}, {args.get('start_date')} -> {args.get('end_date')}",
        "export_investment_report_tool": f"title={args.get('title', '')!r}, format={args.get('format', 'md')}",
        "validate_claims_tool": f"claims_count={len(args.get('claims') or []) if isinstance(args.get('claims'), list) else 1}",
        "portfolio_info_tool": "portfolio_info",
        "place_trade_tool": f"ticker={args.get('ticker')}, side={args.get('side')}, qty={args.get('qty')}, type={args.get('order_type', 'market')}",
        "close_position_tool": f"close_all={args.get('all', False)} ticker={args.get('ticker')}"
        if args.get("ticker")
        else f"close_all={args.get('all', False)}",
        "get_news_tool": f"symbols={args.get('symbols')}, limit={args.get('limit', 10)}",
        "portfolio_history_tool": f"period={args.get('period', '1M')}, timeframe={args.get('timeframe', 'auto')}",
        "account_activity_tool": f"types={args.get('activity_types')}, page_size={args.get('page_size', 20)}",
    }
    return summaries.get(name, str(args)[:120])


def safe_result(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return {"text": str(obj)}


def execute_tool(
    name: str,
    args: BaseModel,
    *,
    agent: Any,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name == "sec_filings_rag_tool":
        return run_sec_filings_rag(args, agent=agent)
    if name == "market_price_tool":
        return run_market_price_tool(args, agent=agent)
    if name == "export_investment_report_tool":
        return run_export_investment_report(args)
    if name == "validate_claims_tool":
        if not isinstance(args, ValidateClaimsArgs):
            full = ValidateClaimsArgs(
                claims=args.claims,
                chunks=list((state or {}).get("final_chunks") or []),
                metadatas=list((state or {}).get("final_metadatas") or []),
            )
        else:
            full = args
        return run_validate_claims(full, agent=agent)
    if name == "portfolio_info_tool":
        return run_portfolio_info(args)
    if name == "place_trade_tool":
        return run_place_trade(args, state=state or {})
    if name == "close_position_tool":
        return run_close_position(args, state=state or {})
    if name == "get_news_tool":
        return run_get_news(args)
    if name == "portfolio_history_tool":
        return run_portfolio_history(args)
    if name == "account_activity_tool":
        return run_account_activity(args)
    return {"text": f"Unknown tool: {name}"}


class ToolExecutor:
    def __init__(self, agent: Any, state: GraphState):
        self.agent = agent
        self.state = state

    def parse_args(self, tool_call: ToolCall) -> dict[str, Any]:
        try:
            parsed = json.loads(tool_call.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def resolve_full_args(self, tool_name: str, llm_args: dict[str, Any]) -> BaseModel:
        mapping = {
            "validate_claims_tool": (
                ValidateClaimsLLMArgs,
                ValidateClaimsArgs,
                lambda a, s: ValidateClaimsArgs(
                    claims=a.claims,
                    chunks=list((s or {}).get("final_chunks") or []),
                    metadatas=list((s or {}).get("final_metadatas") or []),
                ),
            ),
            "sec_filings_rag_tool": (SecFilingsRAGArgs, None, lambda a, s: a),
            "market_price_tool": (MarketPriceArgs, None, lambda a, s: a),
            "portfolio_info_tool": (PortfolioInfoArgs, None, lambda a, s: a),
            "place_trade_tool": (PlaceTradeArgs, None, lambda a, s: a),
            "close_position_tool": (ClosePositionArgs, None, lambda a, s: a),
            "get_news_tool": (GetNewsArgs, None, lambda a, s: a),
            "portfolio_history_tool": (PortfolioHistoryArgs, None, lambda a, s: a),
            "account_activity_tool": (AccountActivityArgs, None, lambda a, s: a),
            "export_investment_report_tool": (ExportReportArgs, None, lambda a, s: a),
        }
        entry = mapping.get(tool_name)
        if not entry:
            raise ValueError(f"Unknown tool: {tool_name}")
        llm_cls, _, resolver = entry
        llm = llm_cls.model_validate(llm_args)
        return resolver(llm, self.state)

    async def dispatch(self, tool_name: str, full_args: BaseModel) -> dict[str, Any]:
        if tool_name == "sec_filings_rag_tool":
            return await run_sec_filings_rag(full_args, agent=self.agent)
        return await asyncio.to_thread(
            execute_tool,
            tool_name,
            full_args,
            agent=self.agent,
            state=self.state,
        )

    async def execute(self, tool_call: ToolCall) -> ToolExecutionOutcome:
        llm_args = self.parse_args(tool_call)
        event: ToolEvent = {
            "id": str(uuid.uuid4()),
            "tool": tool_call.name,
            "status": "running",
            "started_at": now_utc(),
            "args": llm_args,
            "args_summary": summarize_tool_args(tool_call.name, tool_call.arguments),
        }
        try:
            full_args = self.resolve_full_args(tool_call.name, llm_args)
        except ValidationError as e:
            msg = e.errors()[0]["msg"] if e.errors() else str(e)
            return self._failed(tool_call, llm_args, event, f"args_validation: {msg}")
        except Exception as e:
            return self._failed(tool_call, llm_args, event, f"args_validation: {e}")
        try:
            result = await self.dispatch(tool_call.name, full_args)
        except Exception as e:
            return self._failed(tool_call, llm_args, event, f"execution: {e}")
        event["status"] = "completed"
        event["result"] = safe_result(result)
        event["finished_at"] = now_utc()
        result_payload = result if isinstance(result, dict) else {"text": str(result)}
        tool_text = result_payload.get(
            "text", json.dumps(result_payload, ensure_ascii=False)
        )
        return ToolExecutionOutcome(
            tool_call=tool_call,
            llm_args=llm_args,
            event=event,
            message=self._tool_message(tool_call, tool_text),
            result=result_payload,
        )

    def _failed(
        self,
        tool_call: ToolCall,
        llm_args: dict[str, Any],
        event: ToolEvent,
        error: str,
    ) -> ToolExecutionOutcome:
        event["status"] = "failed"
        event["error"] = error
        event["finished_at"] = now_utc()
        tool_text = json.dumps({"error": error}, ensure_ascii=False)
        return ToolExecutionOutcome(
            tool_call=tool_call,
            llm_args=llm_args,
            event=event,
            message=self._tool_message(tool_call, tool_text),
            result=None,
        )

    @staticmethod
    def _tool_message(tool_call: ToolCall, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": content,
        }
