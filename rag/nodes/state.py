from __future__ import annotations

from typing import Any, TypedDict

from rag.llm_provider import ToolCall


class GraphState(TypedDict, total=False):
    conversation_id: str
    query: str
    messages: list[dict[str, str]]
    lc_messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    report_artifacts: list[dict[str, Any]]
    tool_calls_pending: bool
    pending_tool_calls: list[ToolCall]
    agent_iterations: int
    normalized_query: str
    metadata_filter: dict[str, str]
    target_tickers: list[str]
    doc_type_priority: list[str]
    intent_route: str
    ambiguous_query: bool
    general_chat: bool
    off_topic_blocked: bool
    clarification_question: str
    decomposed_queries: list[str]
    price_context: str
    price_tool_used: bool
    price_tool_attempts: int
    price_tool_decision: str
    price_tickers: list[str]
    price_window_start: str
    price_window_end: str
    memory_summary: str
    memory_window: list[dict[str, str]]
    candidate_indices: list[int]
    final_chunks: list[str]
    final_metadatas: list[dict[str, Any]]
    draft_answer: str
    answer: str
    gc_applied: bool
    stats: dict[str, Any]
