from __future__ import annotations

from typing import Any

from rag.llm_provider import ToolCall
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable
from rag.tool_executor import ToolExecutor


def _merge_tool_side_effects(
    tool_name: str,
    result: dict[str, Any] | None,
    *,
    final_chunks: list[str],
    final_metadatas: list[dict[str, Any]],
    price_context: str,
    report_artifacts: list[dict[str, Any]],
    stats: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], str, list[dict[str, Any]], dict[str, Any]]:
    if not result:
        return final_chunks, final_metadatas, price_context, report_artifacts, stats

    if tool_name == "sec_filings_rag_tool":
        final_chunks = result.get("final_chunks") or final_chunks
        final_metadatas = result.get("final_metadatas") or final_metadatas
        stats.update(result.get("stats") or {})
        stats["rag_tool_used"] = True

    if tool_name == "market_price_tool" and result.get("price_context"):
        price_context = result["price_context"]
        stats["price_tool_used"] = True

    if tool_name == "export_investment_report_tool" and result.get("path"):
        report_artifacts.append(
            {
                "path": result["path"],
                "filename": result.get("filename", ""),
                "title": result.get("title", ""),
                "format": result.get("format", "md"),
            }
        )
        stats["report_exported"] = True

    if tool_name == "validate_claims_tool":
        stats["validate_tool_used"] = True
        stats.update(result.get("stats") or {})

    if tool_name == "simulate_portfolio_tool" and result.get("positions"):
        stats["simulate_tool_used"] = True

    return final_chunks, final_metadatas, price_context, report_artifacts, stats


@traceable(name="tools_node")
async def tools_node(agent: Any, state: GraphState) -> GraphState:
    lc_messages = list(state.get("lc_messages") or [])
    pending: list[ToolCall] = state.get("pending_tool_calls") or []
    stats = dict(state.get("stats") or {})
    final_chunks = list(state.get("final_chunks") or [])
    final_metadatas = list(state.get("final_metadatas") or [])
    price_context = state.get("price_context", "")
    report_artifacts = list(state.get("report_artifacts") or [])
    tool_events = list(state.get("tool_events") or [])
    executor = ToolExecutor(agent=agent, state=state)

    for tool_call in pending:
        outcome = await executor.execute(tool_call)
        tool_events.append(outcome.event)
        lc_messages.append(outcome.message)
        (
            final_chunks,
            final_metadatas,
            price_context,
            report_artifacts,
            stats,
        ) = _merge_tool_side_effects(
            outcome.tool_call.name,
            outcome.result,
            final_chunks=final_chunks,
            final_metadatas=final_metadatas,
            price_context=price_context,
            report_artifacts=report_artifacts,
            stats=stats,
        )

    return {
        "lc_messages": lc_messages,
        "tool_calls_pending": False,
        "pending_tool_calls": [],
        "final_chunks": final_chunks,
        "final_metadatas": final_metadatas,
        "price_context": price_context,
        "report_artifacts": report_artifacts,
        "tool_events": tool_events,
        "stats": stats,
    }
