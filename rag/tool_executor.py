from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from rag.llm_provider import ToolCall
from rag.nodes.state import GraphState, ToolEvent
from rag.tool_schemas import (
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
from rag.tools import execute_tool


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
    if name == "sec_filings_rag_tool":
        return f"query={args.get('query', '')!r}, tickers={args.get('tickers')}, years={args.get('years')}, doc_types={args.get('doc_types')}"
    if name == "market_price_tool":
        return f"tickers={args.get('tickers')}, {args.get('start_date')} -> {args.get('end_date')}"
    if name == "export_investment_report_tool":
        return f"title={args.get('title', '')!r}, format={args.get('format', 'md')}"
    if name == "validate_claims_tool":
        claims = args.get("claims") or []
        return f"claims_count={len(claims) if isinstance(claims, list) else 1}"
    if name == "portfolio_info_tool":
        return "portfolio_info"
    if name == "place_trade_tool":
        return f"ticker={args.get('ticker')}, side={args.get('side')}, qty={args.get('qty')}, type={args.get('order_type', 'market')}"
    if name == "close_position_tool":
        ticker = args.get("ticker")
        return f"close_all={args.get('all', False)} ticker={ticker}" if ticker else f"close_all={args.get('all', False)}"
    if name == "get_news_tool":
        return f"symbols={args.get('symbols')}, limit={args.get('limit', 10)}"
    if name == "portfolio_history_tool":
        return f"period={args.get('period', '1M')}, timeframe={args.get('timeframe', 'auto')}"
    if name == "account_activity_tool":
        return f"types={args.get('activity_types')}, page_size={args.get('page_size', 20)}"
    return str(args)[:120]


def safe_result(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return {"text": str(obj)}


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
        if tool_name == "validate_claims_tool":
            llm = ValidateClaimsLLMArgs.model_validate(llm_args)
            return ValidateClaimsArgs(
                claims=llm.claims,
                chunks=list((self.state or {}).get("final_chunks") or []),
                metadatas=list((self.state or {}).get("final_metadatas") or []),
            )
        if tool_name == "sec_filings_rag_tool":
            return SecFilingsRAGArgs.model_validate(llm_args)
        if tool_name == "market_price_tool":
            return MarketPriceArgs.model_validate(llm_args)
        if tool_name == "portfolio_info_tool":
            return PortfolioInfoArgs.model_validate(llm_args)
        if tool_name == "place_trade_tool":
            return PlaceTradeArgs.model_validate(llm_args)
        if tool_name == "close_position_tool":
            return ClosePositionArgs.model_validate(llm_args)
        if tool_name == "get_news_tool":
            return GetNewsArgs.model_validate(llm_args)
        if tool_name == "portfolio_history_tool":
            return PortfolioHistoryArgs.model_validate(llm_args)
        if tool_name == "account_activity_tool":
            return AccountActivityArgs.model_validate(llm_args)
        if tool_name == "export_investment_report_tool":
            return ExportReportArgs.model_validate(llm_args)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def dispatch(self, tool_name: str, full_args: BaseModel) -> dict[str, Any]:
        if tool_name == "sec_filings_rag_tool":
            from rag.tools import run_sec_filings_rag

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
        tool_text = result_payload.get("text", json.dumps(result_payload, ensure_ascii=False))
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
