from __future__ import annotations

import uuid
from functools import partial

from langgraph.graph import END, StateGraph

from rag.hybrid_rag import HybridRAG
from rag.nodes.decompose_node import decompose_query_node
from rag.nodes.generation_node import answer_generate_node, synthesis_node
from rag.nodes.intent_node import clarify_node, intent_scope_node, route_after_intent_node
from rag.nodes.memory_nodes import gc_node, memory_read_node, memory_write_node
from rag.nodes.memory_store import MemoryStore
from rag.nodes.prepare_node import prepare_query_node
from rag.nodes.retrieval_node import multi_retrieve_node
from rag.nodes.rerank_node import rerank_node
from rag.nodes.state import GraphState
from rag.nodes.tool_nodes import (
    price_data_node,
    route_after_tool_orchestrator_node,
    tool_orchestrator_node,
)
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
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_query_node", partial(prepare_query_node, self))
        graph.add_node("intent_scope_node", partial(intent_scope_node, self))
        graph.add_node("clarify_node", partial(clarify_node, self))
        graph.add_node("memory_read_node", partial(memory_read_node, self))
        graph.add_node("tool_orchestrator_node", partial(tool_orchestrator_node, self))
        graph.add_node("price_data_node", partial(price_data_node, self))
        graph.add_node("decompose_query_node", partial(decompose_query_node, self))
        graph.add_node("multi_retrieve_node", partial(multi_retrieve_node, self))
        graph.add_node("rerank_node", partial(rerank_node, self))
        graph.add_node("answer_generate_node", partial(answer_generate_node, self))
        graph.add_node("synthesis_node", partial(synthesis_node, self))
        graph.add_node("memory_write_node", partial(memory_write_node, self))
        graph.add_node("gc_node", partial(gc_node, self))

        graph.set_entry_point("prepare_query_node")
        graph.add_edge("prepare_query_node", "intent_scope_node")
        graph.add_conditional_edges(
            "intent_scope_node",
            route_after_intent_node,
            {
                "clarify": "clarify_node",
                "continue": "memory_read_node",
            },
        )
        graph.add_edge("clarify_node", END)
        graph.add_edge("memory_read_node", "tool_orchestrator_node")
        graph.add_conditional_edges(
            "tool_orchestrator_node",
            route_after_tool_orchestrator_node,
            {
                "call_price_tool": "price_data_node",
                "continue": "decompose_query_node",
            },
        )
        graph.add_edge("price_data_node", "tool_orchestrator_node")
        graph.add_edge("decompose_query_node", "multi_retrieve_node")
        graph.add_edge("multi_retrieve_node", "rerank_node")
        graph.add_edge("rerank_node", "answer_generate_node")
        graph.add_edge("answer_generate_node", "synthesis_node")
        graph.add_edge("synthesis_node", "memory_write_node")
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
        }
        return self.graph.invoke(initial_state)
