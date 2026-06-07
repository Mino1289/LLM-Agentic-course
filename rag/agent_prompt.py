from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState


AGENT_SYSTEM_PROMPT = """You are a finance research assistant for the tracked company universe.
You have tools to search SEC filings and earnings calls, fetch market prices, validate claims,
simulate fictional portfolios, and export investment reports.

Tool usage guidelines:
- sec_filings_rag_tool: fundamental risks, MD&A, SEC/foreign issuer filings, earnings call transcripts.
  Filter doc_types: 10-K, 10-Q, 8-K, 20-F, 6-K, EARNINGS_CALL.
- market_price_tool: performance, volatility, comparisons needing price history.
- validate_claims_tool: after RAG retrieval, verify key factual claims against excerpts (supported/partial/unsupported).
- simulate_portfolio_tool: fictional allocation/rebalance across tracked tickers only, no real trades.
- export_investment_report_tool: when the user asks to save/generate a report file.

For complex tasks (e.g. compare two tracked companies with 2024 SEC risks and
6-month performance then save):
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


def build_lc_messages(agent: Any, state: GraphState) -> list[dict[str, Any]]:
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
