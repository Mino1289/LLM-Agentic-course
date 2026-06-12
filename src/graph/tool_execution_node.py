"""Noeud d'exécution des outils — appel séquentiel avec déduplication et fusion d'état."""
from __future__ import annotations

import json
from typing import Any

from src.llm.types import ToolCall
from src.graph.state import GraphState
from src.graph.tracing import traceable
from src.tools.execute import ToolExecutor, now_utc, summarize_tool_args


def _canonical_tool_args(raw_args: str | dict[str, Any]) -> str:
    if isinstance(raw_args, dict):
        parsed = raw_args
    else:
        try:
            parsed = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return str(raw_args or "").strip()
        if not isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_signature(tool_name: str, raw_args: str | dict[str, Any]) -> str:
    return f"{tool_name}:{_canonical_tool_args(raw_args)}"


def _completed_tool_signatures(tool_events: list[dict[str, Any]]) -> set[str]:
    sigs = set()
    for event in tool_events:
        if event.get("status") != "completed":
            continue
        tool = str(event.get("tool", ""))
        args = event.get("args")
        if tool and isinstance(args, dict):
            sigs.add(_tool_signature(tool, args))
    return sigs


def _skipped_duplicate_event(tool_call: ToolCall) -> dict[str, Any]:
    return {"tool": tool_call.name, "status": "skipped", "reason": "duplicate_tool_call", "args": _safe_tool_args(tool_call.arguments), "started_at": now_utc(), "finished_at": now_utc()}


def _safe_tool_args(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _duplicate_tool_message(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name,
        "content": '{"skipped": true, "reason": "duplicate_tool_call", "message": "This exact tool call was already executed successfully."}',
    }


def _rag_chunk_key(chunk, metadata) -> tuple:
    return (str(metadata.get("ticker", "")), str(metadata.get("year", "")), str(metadata.get("file_type", "")),
            str(metadata.get("section", "")), str(metadata.get("source", "")), chunk)


def _merge_rag_context(existing_chunks, existing_metadatas, new_chunks, new_metadatas):
    merged_chunks, merged_metadatas = [], []
    seen = set()
    pairs = [(c, existing_metadatas[i] if i < len(existing_metadatas) else {}) for i, c in enumerate(existing_chunks)]
    pairs.extend((c, new_metadatas[i] if i < len(new_metadatas) else {}) for i, c in enumerate(new_chunks))
    for chunk, metadata in pairs:
        key = _rag_chunk_key(chunk, metadata)
        if key in seen:
            continue
        seen.add(key)
        merged_chunks.append(chunk)
        merged_metadatas.append(metadata)
    return merged_chunks, merged_metadatas


def _merge_count_dicts(existing, incoming) -> dict[str, int]:
    merged = {}
    for source in (existing or {}, incoming or {}):
        for key, value in source.items():
            try:
                merged[str(key)] = merged.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue
    return merged


def _merge_unique_list(existing, incoming) -> list[str]:
    values = []
    for raw in (existing or []), (incoming or []):
        if not isinstance(raw, list):
            continue
        for item in raw:
            v = str(item)
            if v and v not in values:
                values.append(v)
    return sorted(values)


def _ticker_counts_from_metadatas(metadatas):
    counts = {}
    for metadata in metadatas:
        ticker = str(metadata.get("ticker", "UNKNOWN"))
        counts[ticker] = counts.get(ticker, 0) + 1
    return counts


def _merge_rag_stats(stats, tool_stats, final_chunks, final_metadatas):
    merged = dict(stats)
    count_keys = {"retrieval_candidate_ticker_counts", "rerank_final_ticker_counts"}
    list_keys = {"retrieval_scoped_tickers", "retrieval_scoped_doc_types"}
    sum_keys = {"decomposed_query_count", "retrieval_candidate_count"}
    for key, value in tool_stats.items():
        if key in count_keys or key in list_keys or key in sum_keys:
            continue
        merged[key] = value
    for key in count_keys:
        merged[key] = _merge_count_dicts(merged.get(key), tool_stats.get(key))
    for key in list_keys:
        merged[key] = _merge_unique_list(merged.get(key), tool_stats.get(key))
    for key in sum_keys:
        try:
            merged[key] = int(merged.get(key, 0) or 0) + int(tool_stats.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    merged["rerank_final_ticker_counts"] = _ticker_counts_from_metadatas(final_metadatas)
    merged["rerank_final_count"] = len(final_chunks)
    merged["chunks_used"] = len(final_chunks)
    return merged


def _merge_tool_side_effects(tool_name, result, *, final_chunks, final_metadatas, price_context, report_artifacts, stats):
    if not result:
        return final_chunks, final_metadatas, price_context, report_artifacts, stats
    if tool_name == "sec_filings_rag_tool":
        final_chunks, final_metadatas = _merge_rag_context(final_chunks, final_metadatas, result.get("final_chunks") or [], result.get("final_metadatas") or [])
        stats = _merge_rag_stats(stats, result.get("stats") or {}, final_chunks, final_metadatas)
        stats["rag_tool_used"] = True
    if tool_name == "market_price_tool" and result.get("price_context"):
        price_context = result["price_context"]
        stats["price_tool_used"] = True
    if tool_name == "export_investment_report_tool" and result.get("path"):
        report_artifacts.append({"path": result["path"], "filename": result.get("filename", ""), "title": result.get("title", ""), "format": result.get("format", "md")})
        stats["report_exported"] = True
    if tool_name == "validate_claims_tool":
        stats["validate_tool_used"] = True
        stats.update(result.get("stats") or {})
    if tool_name in ("portfolio_info_tool", "place_trade_tool", "close_position_tool"):
        stats["alpaca_tool_used"] = True
    return final_chunks, final_metadatas, price_context, report_artifacts, stats


@traceable(name="tools_node")
async def tools_node(agent: Any, state: GraphState) -> GraphState:
    lc_messages = list(state.get("lc_messages") or [])
    pending = state.get("pending_tool_calls") or []
    stats = dict(state.get("stats") or {})
    final_chunks = list(state.get("final_chunks") or [])
    final_metadatas = list(state.get("final_metadatas") or [])
    price_context = state.get("price_context", "")
    report_artifacts = list(state.get("report_artifacts") or [])
    tool_events = list(state.get("tool_events") or [])
    executor = ToolExecutor(agent=agent, state=state)
    completed_signatures = _completed_tool_signatures(tool_events)
    for tool_call in pending:
        signature = _tool_signature(tool_call.name, tool_call.arguments)
        if signature in completed_signatures:
            tool_events.append(_skipped_duplicate_event(tool_call))
            lc_messages.append(_duplicate_tool_message(tool_call))
            stats["duplicate_tool_calls_skipped"] = stats.get("duplicate_tool_calls_skipped", 0) + 1
            continue
        outcome = await executor.execute(tool_call)
        tool_events.append(outcome.event)
        lc_messages.append(outcome.message)
        if outcome.event.get("status") == "completed":
            completed_signatures.add(signature)
        final_chunks, final_metadatas, price_context, report_artifacts, stats = _merge_tool_side_effects(
            outcome.tool_call.name, outcome.result,
            final_chunks=final_chunks, final_metadatas=final_metadatas,
            price_context=price_context, report_artifacts=report_artifacts, stats=stats,
        )
    return {
        "lc_messages": lc_messages, "tool_calls_pending": False, "pending_tool_calls": [],
        "final_chunks": final_chunks, "final_metadatas": final_metadatas,
        "price_context": price_context, "report_artifacts": report_artifacts,
        "tool_events": tool_events, "stats": stats,
    }
