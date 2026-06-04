from __future__ import annotations

import uuid
from functools import partial

from langgraph.graph import END, StateGraph

from rag.hybrid_rag import HybridRAG
from rag.nodes.agent_nodes import (
    agent_node,
    finalize_from_agent_state,
    route_after_agent,
    tools_node,
)
from rag.nodes.memory_nodes import gc_node, memory_read_node, memory_write_node
from rag.nodes.memory_store import MemoryStore
from rag.nodes.prepare_node import prepare_query_node
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


class FinanceLangGraphAgent:
    def __init__(
        self,
        rag: HybridRAG,
        memory_window_size: int = 6,
        summarize_every_n_turns: int = 6,
        max_context_chunks: int = 8,
        max_context_tokens: int = 3500,
        decompose_query_count: int = 4,
        price_max_days: int = 180,
        price_max_points: int = 40,
        price_max_tickers: int = 3,
        price_default_days: int = 90,
        price_max_attempts: int = 2,
        max_tool_iterations: int = 6,
    ):
        self.rag = rag
        self.memory_store = MemoryStore(window_size=memory_window_size)
        self.summarize_every_n_turns = max(4, summarize_every_n_turns)
        self.max_context_chunks = max(4, max_context_chunks)
        self.max_context_tokens = max(1200, max_context_tokens)
        self.decompose_query_count = max(3, decompose_query_count)
        self.price_max_days = max(30, price_max_days)
        self.price_max_points = max(10, price_max_points)
        self.price_max_tickers = max(1, price_max_tickers)
        self.price_default_days = max(15, min(price_default_days, self.price_max_days))
        self.price_max_attempts = max(1, price_max_attempts)
        self.max_tool_iterations = max(2, max_tool_iterations)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_query_node", partial(prepare_query_node, self))
        graph.add_node("memory_read_node", partial(memory_read_node, self))
        graph.add_node("agent_node", partial(agent_node, self))
        graph.add_node("tools_node", partial(tools_node, self))
        graph.add_node("finalize_node", finalize_from_agent_state)
        graph.add_node("memory_write_node", partial(memory_write_node, self))
        graph.add_node("gc_node", partial(gc_node, self))

        graph.set_entry_point("prepare_query_node")
        graph.add_edge("prepare_query_node", "memory_read_node")
        graph.add_edge("memory_read_node", "agent_node")
        graph.add_conditional_edges(
            "agent_node",
            route_after_agent,
            {
                "tools": "tools_node",
                "finalize": "finalize_node",
            },
        )
        graph.add_edge("tools_node", "agent_node")
        graph.add_edge("finalize_node", "memory_write_node")
        graph.add_edge("memory_write_node", "gc_node")
        graph.add_edge("gc_node", END)
        return graph.compile()

    @traceable(name="finance_langgraph_run")
    def run(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> GraphState:
        convo_id = conversation_id or str(uuid.uuid4())
        initial_state: GraphState = {
            "conversation_id": convo_id,
            "query": query,
            "messages": messages or [],
            "agent_iterations": 0,
            "tool_events": [],
            "report_artifacts": [],
        }
        return self.graph.invoke(initial_state)
