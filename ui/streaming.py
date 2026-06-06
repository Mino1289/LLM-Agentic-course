"""UI streaming helper for the Finance RAG LangGraph agent.

Consumes `agent.astream()` via `asyncio.run` and dispatches events to
the provided text/status containers. Returns the final GraphState
captured from the `on_graph_end` event (no double `arun()` call).

Token streaming: uses a contextvar-based token sink (rag.llm_provider.token_sink)
registered before the agent runs, so each text delta from the LLM triggers
a callback that updates the text container with a per-word buffer
(avoids Streamlit flicker on per-char re-renders).

This module is intentionally decoupled from Streamlit at the call site:
the caller passes anything that implements `.markdown(str)` (for the
text container) and `.update(label=...)` (for the status container).
In production these are `st.empty()` and `st.status(...)`; in tests
they are mocks.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from rag.llm_provider import token_sink

WORD_SEPARATORS = (" ", "\n", ".", ",", ";", ":", "!", "?")


def run_stream(
    agent: Any,
    query: str,
    conversation_id: str,
    messages: list[dict[str, str]],
    text_container: Any,
    status_container: Any,
    *,
    on_text_chunk: Optional[callable] = None,
) -> dict[str, Any]:
    """Synchronously drive `agent.astream()` and dispatch events to the
    containers. Returns the final GraphState.

    Uses `asyncio.run` internally. Raises RuntimeError if called from
    inside a running event loop (use `await agent.astream(...)` directly).

    Args:
        agent: A FinanceLangGraphAgent (anything with `astream(query, ...)`).
        query: The user question.
        conversation_id: Conversation ID for the run.
        messages: Prior messages for the conversation.
        text_container: Anything with `.markdown(str)` method. Receives
            progressive updates as tokens arrive.
        status_container: Anything with `.update(label=str)` method.
            Receives node/tool lifecycle updates.
        on_text_chunk: Optional callback invoked with the final streamed
            text when the stream ends (used by tests for assertions).

    Returns:
        The final GraphState (dict) — same as `agent.arun()` would
        return. Comes from the custom `on_graph_end` event emitted by
        `agent.astream()`.
    """
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
                    # Path used by tests that mock on_chat_model_stream.
                    # In production, the provider's token_sink fires first.
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
