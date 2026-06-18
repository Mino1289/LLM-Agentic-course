"""Orchestrateur du graphe LangGraph — noeuds, arêtes, compilation, streaming."""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from src.graph.state import GraphState
from src.graph.prepare_node import prepare_query_node
from src.graph.decompose_node import decompose_query_node
from src.graph.guard import guard_node, route_after_guard
from src.graph.retrieval_node import multi_retrieve_node
from src.graph.rerank_node import rerank_node
from src.graph.agent_node import (
    agent_node,
    route_after_agent,
    finalize_from_agent_state,
)
from src.graph.tool_execution_node import tools_node
from src.graph.memory_nodes import memory_context_node

_LOGGER = logging.getLogger("src.graph.flow")


class FinanceLangGraphAgent:
    def __init__(
        self,
        rag: Any,
        *,
        memory_store: Any = None,
        max_tool_iterations: int = 5,
        price_max_tickers: int = 10,
        price_default_days: int = 90,
        price_max_days: int = 730,
        price_max_points: int = 60,
        rerank_max_tokens: int = 8000,
        rerank_top_k: int = 10,
        decompose_query_count: int = 2,
        max_context_chunks: int = 8,
    ):
        self.rag = rag
        self.memory_store = memory_store
        self.max_tool_iterations = max_tool_iterations
        self.price_max_tickers = price_max_tickers
        self.price_default_days = price_default_days
        self.price_max_days = price_max_days
        self.price_max_points = price_max_points
        self.rerank_max_tokens = rerank_max_tokens
        self.rerank_top_k = rerank_top_k
        self.decompose_query_count = decompose_query_count
        self.max_context_chunks = max_context_chunks
        self.graph = self._build_graph()
        _LOGGER.info("FinanceLangGraphAgent compiled")

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GraphState)
        builder.add_node("prepare", partial(prepare_query_node, self))
        builder.add_node("guard", partial(guard_node, self))
        builder.add_node("decompose", partial(decompose_query_node, self))
        builder.add_node("retrieve", partial(multi_retrieve_node, self))
        builder.add_node("rerank", partial(rerank_node, self))
        builder.add_node("agent", partial(agent_node, self))
        builder.add_node("tools", partial(tools_node, self))
        builder.add_node("memory", memory_context_node)
        builder.add_node("finalize", finalize_from_agent_state)
        builder.set_entry_point("prepare")
        builder.add_edge("prepare", "guard")
        builder.add_conditional_edges(
            "guard",
            route_after_guard,
            {"agent": "agent", "finalize": "finalize", END: END},
        )
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "memory")
        builder.add_edge("memory", "agent")
        builder.add_edge("decompose", "retrieve")
        builder.add_conditional_edges(
            "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _initial_state(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> GraphState:
        return {
            "query": query,
            "messages": messages or [],
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "lc_messages": [],
            "tool_calls_pending": False,
            "pending_tool_calls": [],
            "answer": "",
            "final_chunks": [],
            "final_metadatas": [],
            "price_context": "",
            "price_series": [],
            "report_artifacts": [],
            "tool_events": [],
            "stats": {},
        }

    async def ainvoke(
        self, state: GraphState, config: dict[str, Any] | None = None
    ) -> GraphState:
        return await self.graph.ainvoke(state, config)

    def invoke(
        self, state: GraphState, config: dict[str, Any] | None = None
    ) -> GraphState:
        return self.graph.invoke(state, config)

    async def arun(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> GraphState:
        return await self.ainvoke(self._initial_state(query, conversation_id, messages))

    def run(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> GraphState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "Cannot call run() from inside a running event loop. Use await arun() instead."
            )
        return asyncio.run(self.arun(query, conversation_id, messages))

    async def astream(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict]:
        initial_state = self._initial_state(query, conversation_id, messages)
        last_state: GraphState | None = None
        async for event in self.graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                delta = ""
                if chunk is not None:
                    delta = getattr(chunk, "content", None) or ""
                    if not delta and isinstance(chunk, dict):
                        delta = chunk.get("content", "") or ""
                if delta:
                    yield {"event": "on_llm_token", "token": delta}
                continue
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    last_state = output
            yield event
        if last_state is not None:
            yield {"event": "on_graph_end", "state": last_state}
