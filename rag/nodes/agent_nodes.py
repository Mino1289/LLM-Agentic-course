from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from rag.llm_provider import ToolCall
from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState, ToolEvent
from rag.nodes.tracing import traceable
from rag.tool_schemas import (
    ExportReportArgs,
    MarketPriceArgs,
    SecFilingsRAGArgs,
    SimulatePortfolioArgs,
    ValidateClaimsArgs,
    ValidateClaimsLLMArgs,
)
from rag.tools import execute_tool, get_tool_definitions

AGENT_SYSTEM_PROMPT = """You are a finance research assistant for NVDA, AMD, and MSFT.
You have tools to search SEC filings and earnings calls, fetch market prices, validate claims,
simulate fictional portfolios, and export investment reports.

Tool usage guidelines:
- sec_filings_rag_tool: fundamental risks, MD&A, SEC events, earnings call transcripts.
  Filter doc_types: 10-K, 10-Q, 8-K, EARNINGS_CALL.
- market_price_tool: performance, volatility, comparisons needing price history.
- validate_claims_tool: after RAG retrieval, verify key factual claims against excerpts (supported/partial/unsupported).
- simulate_portfolio_tool: fictional allocation/rebalance (weights sum to 100%, NVDA/AMD/MSFT only, no real trades).
- export_investment_report_tool: when the user asks to save/generate a report file.

For complex tasks (e.g. compare MSFT vs NVDA with 2024 SEC risks and 6-month performance then save):
1) Retrieve filings per company/year with sec_filings_rag_tool
2) Fetch prices with market_price_tool
3) validate_claims_tool on main factual statements from step 1
4) simulate_portfolio_tool if the user wants a fictional allocation
5) Synthesize in French
6) Call export_investment_report_tool with the full report body when saving is requested

Payload format note: some tool responses and the conversation memory block use
TOON (Token-Oriented Object Notation) instead of JSON. TOON tabular arrays look
like `name[N]{col1,col2,...}:\n  val1,val2\n  ...` where N is the row count and
each row is a comma-separated record matching the column order. Treat TOON
input the same as JSON — read fields by column name, not by position.

Respond in French unless the user writes in English. Do not invent facts not supported by tool outputs.
"""


def _format_chat_history(messages: list[dict[str, str]], keep_last: int = 8) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages[-keep_last:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_lc_messages(agent: Any, state: GraphState) -> list[dict[str, Any]]:
    lc_messages: list[dict[str, Any]] = list(state.get("lc_messages") or [])
    if lc_messages:
        return lc_messages

    messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    memory_summary = state.get("memory_summary", "")
    if memory_summary:
        messages.append({"role": "system", "content": f"Conversation memory:\n{memory_summary}"})

    universe = format_universe_hint(agent, max_items=12)
    messages.append(
        {
            "role": "system",
            "content": f"Tracked tickers: {universe}. Today (UTC): {datetime.now(UTC).date().isoformat()}.",
        }
    )

    history = _format_chat_history(state.get("messages", []), keep_last=8)
    if history:
        messages.append({"role": "system", "content": f"Recent chat:\n{history}"})

    query = state.get("normalized_query") or state.get("query", "")
    messages.append({"role": "user", "content": query})
    return messages


def _summarize_tool_args(name: str, arguments: str) -> str:
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
    if name == "simulate_portfolio_tool":
        return f"allocations={args.get('allocations')}, notional={args.get('notional_usd', 100000)}"
    return str(args)[:120]


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return {}


def _safe_result(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return {"text": str(obj)}


def _resolve_full_args(
    tool_name: str, llm_args: dict[str, Any], state: Any
) -> BaseModel:
    """Map a tool name + LLM-supplied args to a fully-resolved BaseModel.

    For ``validate_claims_tool``, chunks and metadatas are injected from
    ``state["final_chunks"]`` and ``state["final_metadatas"]``.
    """
    if tool_name == "validate_claims_tool":
        llm = ValidateClaimsLLMArgs.model_validate(llm_args)
        return ValidateClaimsArgs(
            claims=llm.claims,
            chunks=list((state or {}).get("final_chunks") or []),
            metadatas=list((state or {}).get("final_metadatas") or []),
        )
    if tool_name == "sec_filings_rag_tool":
        return SecFilingsRAGArgs.model_validate(llm_args)
    if tool_name == "market_price_tool":
        return MarketPriceArgs.model_validate(llm_args)
    if tool_name == "simulate_portfolio_tool":
        return SimulatePortfolioArgs.model_validate(llm_args)
    if tool_name == "export_investment_report_tool":
        return ExportReportArgs.model_validate(llm_args)
    raise ValueError(f"Unknown tool: {tool_name}")


def pre_agent_guard(state: GraphState) -> GraphState | None:
    """Short-circuit off-topic or empty queries without tool loop."""
    query = (state.get("normalized_query") or "").strip()
    if not query:
        return {
            "answer": "Peux-tu préciser ta question finance (entreprise, période, ou type de document) ?",
            "tool_calls_pending": False,
            "stats": {**(state.get("stats") or {}), "intent_route": "clarify"},
        }
    lower = query.lower()
    if any(token in lower for token in ["factorielle", "javascript", "python code", "écrire du code"]):
        return {
            "answer": "Je suis spécialisé en analyse financière (SEC, prix, rapports). Pose une question sur NVDA, AMD ou MSFT.",
            "tool_calls_pending": False,
            "stats": {**(state.get("stats") or {}), "intent_route": "reject_offtopic"},
        }
    return None


def route_after_agent(state: GraphState) -> str:
    if state.get("tool_calls_pending"):
        return "tools"
    return "finalize"


@traceable(name="agent_node")
async def agent_node(agent: Any, state: GraphState) -> GraphState:
    guard = pre_agent_guard(state)
    if guard:
        return guard

    iterations = state.get("agent_iterations", 0)
    if iterations >= agent.max_tool_iterations:
        return {
            "answer": state.get("answer")
            or "Limite d'appels d'outils atteinte. Reformule ta question ou réduis le périmètre.",
            "tool_calls_pending": False,
        }

    lc_messages = _build_lc_messages(agent, state)
    text_parts: list[str] = []
    tool_calls_dict: dict[str, dict[str, Any]] = {}

    async for stream_chunk in agent.rag.provider.ainvoke_with_tools_stream(
        lc_messages,
        tools=get_tool_definitions(),
        temperature=0.2,
        max_tokens=2500,
    ):
        if stream_chunk.delta:
            text_parts.append(stream_chunk.delta)
        if stream_chunk.tool_call_delta:
            for tc_delta in stream_chunk.tool_call_delta:
                # OpenAI streaming protocol: the FIRST chunk of a tool_call
                # carries `index`, `id`, `name`, and `arguments="""`; subsequent
                # continuation chunks of the SAME tool_call carry only
                # `arguments` (id, name, and sometimes index are None).
                # We MUST accumulate by `index` (the canonical position in the
                # tool_calls array), NOT by `id` — otherwise continuation chunks
                # get treated as separate tool_calls with synthesized ids
                # ("tc_0", "tc_1", ...) which break tool_call_id matching
                # in the next LLM call.
                tc_index = tc_delta.get("index")
                if tc_index is None:
                    # Fallback: if index is missing, assume this is a
                    # continuation of the LAST tool_call. This handles
                    # providers that omit `index` after the first chunk.
                    if tool_calls_dict:
                        last_key = next(reversed(tool_calls_dict))
                        tc_key = last_key
                    else:
                        tc_key = "tc_0"
                else:
                    tc_key = tc_index
                if tc_key not in tool_calls_dict:
                    tool_calls_dict[tc_key] = {
                        "id": tc_delta.get("id") or f"tc_{tc_key}",
                        "name": tc_delta.get("name") or "",
                        "arguments_parts": [],
                    }
                if tc_delta.get("id"):
                    tool_calls_dict[tc_key]["id"] = tc_delta["id"]
                if tc_delta.get("name"):
                    tool_calls_dict[tc_key]["name"] = tc_delta["name"]
                if tc_delta.get("arguments"):
                    tool_calls_dict[tc_key]["arguments_parts"].append(tc_delta["arguments"])

    final_text = "".join(text_parts)
    final_tool_calls: list[ToolCall] = [
        ToolCall(
            id=tc["id"],
            name=tc["name"],
            arguments="".join(tc["arguments_parts"]),
        )
        for tc in tool_calls_dict.values()
        if tc["name"]
    ]

    stats = dict(state.get("stats") or {})
    stats["agent_iterations"] = iterations + 1

    if final_tool_calls:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": final_text,
            "tool_calls": final_tool_calls,
        }
        lc_messages.append(assistant_msg)
        tool_events = list(state.get("tool_events") or [])
        for tc in final_tool_calls:
            tool_events.append(
                {
                    "tool": tc.name,
                    "args_summary": _summarize_tool_args(tc.name, tc.arguments),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        return {
            "lc_messages": lc_messages,
            "tool_calls_pending": True,
            "pending_tool_calls": final_tool_calls,
            "tool_events": tool_events,
            "agent_iterations": iterations + 1,
            "stats": stats,
        }

    answer = (final_text or "").strip() or "Je n'ai pas pu formuler de réponse."
    lc_messages.append({"role": "assistant", "content": answer})
    return {
        "lc_messages": lc_messages,
        "answer": answer,
        "tool_calls_pending": False,
        "tool_events": state.get("tool_events") or [],
        "agent_iterations": iterations + 1,
        "stats": stats,
    }


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

    for tc in pending:
        event_id = str(uuid.uuid4())
        try:
            llm_args = json.loads(tc.arguments or "{}")
        except json.JSONDecodeError:
            llm_args = {}

        event: ToolEvent = {
            "id": event_id,
            "tool": tc.name,
            "status": "running",
            "started_at": _now_utc(),
            "args": llm_args,
            "args_summary": _summarize_tool_args(tc.name, tc.arguments),
        }
        tool_events.append(event)

        # OpenAI API invariant: every assistant.tool_call MUST be followed by
        # a matching tool response. We accumulate a single tool_text per
        # pending tool_call and ALWAYS append the tool message below, even
        # on validation/execution failure (with an error payload).

        # Validate LLM args; resolve injected (chunks/metadatas) from state.
        tool_text: str
        try:
            full_args = _resolve_full_args(tc.name, llm_args, state)
        except ValidationError as e:
            event["status"] = "failed"
            msg = e.errors()[0]["msg"] if e.errors() else str(e)
            event["error"] = f"args_validation: {msg}"
            event["finished_at"] = _now_utc()
            tool_text = json.dumps({"error": f"args_validation: {msg}"}, ensure_ascii=False)
            lc_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": tool_text,
                }
            )
            continue

        # Execute the tool (blocking I/O → offload to thread).
        try:
            result = await asyncio.to_thread(
                execute_tool, tc.name, full_args, agent=agent, state=state,
            )
        except Exception as e:
            event["status"] = "failed"
            event["error"] = f"execution: {e}"
            event["finished_at"] = _now_utc()
            tool_text = json.dumps({"error": f"execution: {e}"}, ensure_ascii=False)
            lc_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": tool_text,
                }
            )
            continue

        event["status"] = "completed"
        event["result"] = _safe_result(result)
        event["finished_at"] = _now_utc()

        # Update state side-effects per tool.
        tool_text = result.get("text", json.dumps(result, ensure_ascii=False))

        if tc.name == "sec_filings_rag_tool":
            final_chunks = result.get("final_chunks") or final_chunks
            final_metadatas = result.get("final_metadatas") or final_metadatas
            stats.update(result.get("stats") or {})
            stats["rag_tool_used"] = True

        if tc.name == "market_price_tool" and result.get("price_context"):
            price_context = result["price_context"]
            stats["price_tool_used"] = True

        if tc.name == "export_investment_report_tool" and result.get("path"):
            report_artifacts.append(
                {
                    "path": result["path"],
                    "filename": result.get("filename", ""),
                    "title": result.get("title", ""),
                    "format": result.get("format", "md"),
                }
            )
            stats["report_exported"] = True

        if tc.name == "validate_claims_tool":
            stats["validate_tool_used"] = True
            stats.update(result.get("stats") or {})

        if tc.name == "simulate_portfolio_tool" and result.get("positions"):
            stats["simulate_tool_used"] = True

        lc_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": tool_text,
            }
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


async def finalize_from_agent_state(state: GraphState) -> GraphState:
    """Ensure answer and metadata are present for UI after guard or agent completion."""
    if state.get("answer"):
        return {}
    lc_messages = state.get("lc_messages") or []
    for msg in reversed(lc_messages):
        if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
            return {"answer": msg["content"]}
    return {"answer": "Aucune réponse générée."}
