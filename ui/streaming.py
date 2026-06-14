from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.llm.sinks import token_sink

WORD_SEPARATORS = (" ", "\n", ".", ",", ";", ":", "!", "?")

AGENT_ICONS = {
    "Intent Router": "🚦",
    "Portfolio Manager": "👔",
    "Fundamental Analyst": "📚",
    "Quantitative Analyst": "📈",
    "Simple Agent": "🤖",
    "Compliance Validator": "🛡️",
    "Executor Trader": "⚡",
    "Human Review": "👤",
}


def _get_icon(agent_name: str) -> str:
    for prefix, icon in AGENT_ICONS.items():
        if agent_name.startswith(prefix):
            return icon
    return "🔧"


def run_phase2_stream(
    agent: Any,
    query: str,
    conversation_id: str,
    messages: list[dict[str, str]],
    text_container: Any,
    status_container: Any,
    *,
    on_text_chunk: Optional[callable] = None,
) -> dict[str, Any]:
    final_state: dict[str, Any] = {}
    streamed_text: list[str] = []
    word_buffer: list[str] = []

    def _flush_buffer(force: bool = False) -> None:
        if not word_buffer:
            return
        chunk = "".join(word_buffer)
        if not force and not any(sep in chunk for sep in WORD_SEPARATORS):
            return
        word_buffer.clear()
        streamed_text.append(chunk)
        text_container.markdown("".join(streamed_text) + "▌")

    def _sink(delta: str) -> None:
        word_buffer.append(delta)
        _flush_buffer(force=False)

    async def consume() -> None:
        nonlocal final_state
        with token_sink(_sink):
            async for event in agent.astream(query, conversation_id, messages):
                kind = event.get("event")
                if kind == "on_chain_start":
                    name = event.get("name", "?")
                    status_container.update(label=f"⏳ {name} en cours...")
                elif kind == "on_tool_start":
                    tool = event.get("name", "?")
                    status_container.update(label=f"⏳ Outil `{tool}` en cours...")
                elif kind == "on_tool_end":
                    status_container.update(label="✅ Outil terminé")
                elif kind == "on_llm_token":
                    token = event.get("token", "")
                    if token:
                        word_buffer.append(token)
                        _flush_buffer(force=False)
                elif kind == "on_graph_end":
                    _flush_buffer(force=True)
                    final_state = event.get("state", {}) or {}
                    status_container.update(label="✅ Terminé")
                    if streamed_text:
                        text_container.markdown("".join(streamed_text))

    asyncio.run(consume())
    if on_text_chunk is not None:
        on_text_chunk("".join(streamed_text))
    return final_state


def run_phase3_stream(
    agent: Any,
    query: str,
    conversation_id: str,
    messages: list[dict[str, str]],
    console_container: Any,
    text_container: Any,
    status_container: Any,
    *,
    on_text_chunk: Optional[callable] = None,
    on_human_review: Optional[callable] = None,
) -> dict[str, Any]:
    final_state: dict[str, Any] = {}
    console_lines: list[str] = []

    async def consume() -> None:
        nonlocal final_state
        async for event in agent.astream(query, conversation_id, messages):
            kind = event.get("event")

            if kind == "on_spoke_event":
                agent_name = event.get("agent", "?")
                status = event.get("status", "running")
                message = event.get("message", "")
                detail = event.get("detail", "")
                icon = _get_icon(agent_name)

                if status == "running":
                    line = f"{icon} **{agent_name}** : {message}"
                    console_lines.append(line)
                    status_container.update(label=f"{icon} {agent_name} en cours...")
                elif status == "completed":
                    line = f"{icon} {agent_name} : ✅ {message}"
                    if detail:
                        line += f"\n\n> {detail}"
                    if console_lines:
                        console_lines[-1] = line
                    status_container.update(label=f"✅ {agent_name} terminé")
                elif status == "failed":
                    if console_lines:
                        console_lines[-1] = f"{icon} {agent_name} : ❌ {message}"
                    status_container.update(label=f"❌ {agent_name} échoué")

                console_container.markdown("\n\n".join(console_lines))

            elif kind == "on_tool_start":
                tool = event.get("name", "?")
                status_container.update(label=f"⏳ Outil `{tool}`...")

            elif kind == "on_tool_end":
                status_container.update(label="✅ Outil complété")

            elif kind == "on_graph_end":
                final_state = event.get("state", {}) or {}
                status_container.update(label="✅ Analyse terminée")

                if final_state.get("human_review_pending"):
                    if on_human_review:
                        on_human_review(final_state)

                answer = final_state.get("answer", "")
                if answer and text_container:
                    text_container.markdown(answer)

    asyncio.run(consume())
    if on_text_chunk is not None:
        on_text_chunk("".join(streamed_text))
    return final_state

# Backward-compatible alias for Phase 2 tests
run_stream = run_phase2_stream
