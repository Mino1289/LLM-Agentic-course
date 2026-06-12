"""Noeud agent — appel LLM avec outils, routage après réponse."""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from src.llm.types import ToolCall
from src.graph.state import GraphState
from src.graph.tracing import traceable

_LOGGER = logging.getLogger("src.graph.agent_node")

from src.prompts.agent_prompt import build_lc_messages


def route_after_agent(state: GraphState) -> str:
    if state.get("tool_calls_pending"):
        return "tools"
    return "finalize"


@traceable(name="agent_node")
async def agent_node(agent: Any, state: GraphState) -> GraphState:
    iterations = state.get("agent_iterations", 0)
    if iterations >= agent.max_tool_iterations:
        return {
            "answer": state.get("answer") or "Limite d'appels d'outils atteinte.",
            "tool_calls_pending": False,
        }
    from src.tools.definitions import get_tool_definitions
    lc_messages = build_lc_messages(agent, state)
    text_parts: list[str] = []
    tool_calls_dict: dict[str | int, dict[str, Any]] = {}
    llm_t0 = time.perf_counter()
    async for stream_chunk in agent.rag.provider.ainvoke_with_tools_stream(
        lc_messages, tools=get_tool_definitions(), temperature=0.2, max_tokens=2500,
    ):
        if stream_chunk.delta:
            text_parts.append(stream_chunk.delta)
        if stream_chunk.tool_call_delta:
            for tc_delta in stream_chunk.tool_call_delta:
                tc_index = tc_delta.get("index")
                if tc_index is None:
                    tc_key = next(reversed(tool_calls_dict)) if tool_calls_dict else "tc_0"
                else:
                    tc_key = tc_index
                if tc_key not in tool_calls_dict:
                    tool_calls_dict[tc_key] = {"id": tc_delta.get("id") or f"tc_{tc_key}", "name": tc_delta.get("name") or "", "arguments_parts": []}
                if tc_delta.get("id"):
                    tool_calls_dict[tc_key]["id"] = tc_delta["id"]
                if tc_delta.get("name"):
                    tool_calls_dict[tc_key]["name"] = tc_delta["name"]
                if tc_delta.get("arguments"):
                    tool_calls_dict[tc_key]["arguments_parts"].append(tc_delta["arguments"])
    final_text = "".join(text_parts)
    final_tool_calls = [ToolCall(id=tc["id"], name=tc["name"], arguments="".join(tc["arguments_parts"])) for tc in tool_calls_dict.values() if tc["name"]]
    stats = dict(state.get("stats") or {})
    stats["agent_iterations"] = iterations + 1
    if final_tool_calls:
        assistant_msg = {"role": "assistant", "content": final_text, "tool_calls": final_tool_calls}
        lc_messages.append(assistant_msg)
        tool_events = list(state.get("tool_events") or [])
        from src.tools.execute import summarize_tool_args
        for tc in final_tool_calls:
            tool_events.append({"tool": tc.name, "args_summary": summarize_tool_args(tc.name, tc.arguments), "timestamp": datetime.now(UTC).isoformat()})
        return {"lc_messages": lc_messages, "tool_calls_pending": True, "pending_tool_calls": final_tool_calls, "tool_events": tool_events, "agent_iterations": iterations + 1, "stats": stats}
    answer = (final_text or "").strip() or "Je n'ai pas pu formuler de réponse."
    lc_messages.append({"role": "assistant", "content": answer})
    return {"lc_messages": lc_messages, "answer": answer, "tool_calls_pending": False, "tool_events": state.get("tool_events") or [], "agent_iterations": iterations + 1, "stats": stats}


async def finalize_from_agent_state(state: GraphState) -> GraphState:
    if state.get("answer"):
        return {}
    lc_messages = state.get("lc_messages") or []
    for msg in reversed(lc_messages):
        if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
            return {"answer": msg["content"]}
    return {"answer": "Aucune réponse générée."}
