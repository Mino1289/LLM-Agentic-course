"""Noeuds mémoire — extraction et enrichissement du contexte mémoire depuis GraphState."""
from __future__ import annotations

from typing import Any

from src.graph.state import GraphState


def summarize_recent_context(state: GraphState, max_turns: int = 6) -> str:
    msgs = state.get("lc_messages") or []
    parts = []
    for msg in msgs[-max_turns * 2:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"[{role}]: {content.strip()[:300]}")
    if not parts:
        return ""
    return "Recent conversation summary:\n" + "\n".join(parts[-max_turns:])


def format_tool_event_context(tool_events: list[dict[str, Any]], max_items: int = 5) -> str:
    if not tool_events:
        return ""
    lines = []
    for event in tool_events[-max_items:]:
        tool = event.get("tool", "?")
        ts = event.get("timestamp", "")
        summary = event.get("args_summary", "")
        status = event.get("status", "completed")
        lines.append(f"- [{ts}] {tool}({summary}): {status}")
    return "Recent tool calls:\n" + "\n".join(lines)


def memory_context_node(state: GraphState) -> GraphState:
    conversation = summarize_recent_context(state)
    tool_context = format_tool_event_context(state.get("tool_events") or [])
    combined = "\n\n".join(p for p in [conversation, tool_context] if p)
    if combined:
        state["memory_context"] = combined
    return state
