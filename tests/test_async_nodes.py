"""Tests verifying that all graph nodes are async coroutines and that
they wrap blocking I/O calls via `asyncio.to_thread`.

These tests were RED at G1 (nodes are sync `def`, no to_thread wrapping).
They turn GREEN after G2 (refactor to `async def` + asyncio.to_thread).
"""

import asyncio
import inspect
import unittest
import unittest.mock


class AsyncNodeStructureTests(unittest.TestCase):
    def test_agent_node_is_coroutine(self):
        from rag.nodes.agent_nodes import agent_node
        self.assertTrue(
            inspect.iscoroutinefunction(agent_node),
            "agent_node must be `async def` to support astream_events",
        )

    def test_tools_node_is_coroutine(self):
        from rag.nodes.agent_nodes import tools_node
        self.assertTrue(
            inspect.iscoroutinefunction(tools_node),
            "tools_node must be `async def` to wrap blocking tool calls",
        )

    def test_finalize_from_agent_state_is_coroutine(self):
        from rag.nodes.agent_nodes import finalize_from_agent_state
        self.assertTrue(
            inspect.iscoroutinefunction(finalize_from_agent_state),
            "finalize_from_agent_state must be `async def`",
        )

    def test_memory_read_node_is_coroutine(self):
        from rag.nodes.memory_nodes import memory_read_node
        self.assertTrue(
            inspect.iscoroutinefunction(memory_read_node),
            "memory_read_node must be `async def`",
        )

    def test_memory_write_node_is_coroutine(self):
        from rag.nodes.memory_nodes import memory_write_node
        self.assertTrue(
            inspect.iscoroutinefunction(memory_write_node),
            "memory_write_node must be `async def`",
        )

    def test_gc_node_is_coroutine(self):
        from rag.nodes.memory_nodes import gc_node
        self.assertTrue(
            inspect.iscoroutinefunction(gc_node),
            "gc_node must be `async def`",
        )

    def test_prepare_query_node_is_coroutine(self):
        from rag.nodes.prepare_node import prepare_query_node
        self.assertTrue(
            inspect.iscoroutinefunction(prepare_query_node),
            "prepare_query_node must be `async def`",
        )

    def test_multi_retrieve_node_is_coroutine(self):
        from rag.nodes.retrieval_node import multi_retrieve_node
        self.assertTrue(
            inspect.iscoroutinefunction(multi_retrieve_node),
            "multi_retrieve_node must be `async def`",
        )

    def test_rerank_node_is_coroutine(self):
        from rag.nodes.rerank_node import rerank_node
        self.assertTrue(
            inspect.iscoroutinefunction(rerank_node),
            "rerank_node must be `async def`",
        )

    def test_decompose_query_node_is_coroutine(self):
        from rag.nodes.decompose_node import decompose_query_node
        self.assertTrue(
            inspect.iscoroutinefunction(decompose_query_node),
            "decompose_query_node must be `async def`",
        )


class AsyncNodeThreadWrappingTests(unittest.IsolatedAsyncioTestCase):
    """Verify that blocking I/O inside async nodes is wrapped via
    `asyncio.to_thread` so the event loop stays responsive."""

    async def _spy_to_thread(self, wrapped, exc=None):
        async def spy(func, *args, **kwargs):
            wrapped.append(func)
            if exc is not None:
                raise exc
            return None
        return spy

    async def test_agent_node_wraps_provider_call_in_to_thread(self):
        from rag.nodes import agent_nodes
        from rag.nodes.agent_nodes import agent_node
        wrapped: list = []
        spy = await self._spy_to_thread(wrapped, exc=asyncio.CancelledError())

        # Minimal stub agent with provider
        class _StubProvider:
            def invoke_with_tools(self, *a, **kw):
                return None
        _stub_provider = _StubProvider()
        class _StubAgent:
            max_tool_iterations = 10
            class rag:
                provider = _stub_provider
        state = {
            "conversation_id": "x",
            "query": "Hello",
            "normalized_query": "Hello",
            "messages": [],
            "tool_events": [],
            "agent_iterations": 0,
            "lc_messages": [],
        }
        with unittest.mock.patch.object(agent_nodes.asyncio, "to_thread", side_effect=spy):
            try:
                await agent_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # Verify that a function with name "invoke_with_tools" was wrapped
        names = [getattr(f, "__name__", "") for f in wrapped]
        self.assertIn(
            "invoke_with_tools", names,
            "agent_node must wrap provider.invoke_with_tools in asyncio.to_thread",
        )

    async def test_tools_node_wraps_execute_tool_in_to_thread(self):
        from rag.nodes import agent_nodes
        from rag.nodes.agent_nodes import tools_node
        from rag.tools import execute_tool
        from rag.llm_provider import ToolCall
        wrapped: list = []
        spy = await self._spy_to_thread(wrapped, exc=asyncio.CancelledError())

        state = {
            "conversation_id": "x",
            "query": "test",
            "normalized_query": "test",
            "messages": [],
            "tool_events": [],
            "agent_iterations": 0,
            "lc_messages": [],
            "llm_response": {},
            "pending_tool_calls": [
                ToolCall(id="1", name="sec_filings_rag_tool", arguments='{"query": "risk"}'),
            ],
            "retrieval_chunks": [],
            "retrieval_metadatas": [],
        }
        class _StubAgent:
            pass
        with unittest.mock.patch.object(agent_nodes.asyncio, "to_thread", side_effect=spy):
            try:
                await tools_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        self.assertIn(
            execute_tool, wrapped,
            "tools_node must wrap execute_tool in asyncio.to_thread",
        )

    async def test_memory_read_node_wraps_memory_store_in_to_thread(self):
        from rag.nodes import memory_nodes
        from rag.nodes.memory_nodes import memory_read_node
        wrapped: list = []
        spy = await self._spy_to_thread(wrapped, exc=asyncio.CancelledError())

        class _StubStore:
            def get_summary(self, *a, **kw):
                return ""
            def get_window(self, *a, **kw):
                return []
        _stub_store = _StubStore()
        class _StubAgent:
            memory_store = _stub_store
        state = {"conversation_id": "x", "query": "q", "messages": []}
        with unittest.mock.patch.object(memory_nodes.asyncio, "to_thread", side_effect=spy):
            try:
                await memory_read_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # At least one wrapped function should be a bound method of _stub_store
        bound_names = [
            getattr(f, "__name__", "")
            for f in wrapped
            if getattr(f, "__self__", None) is _stub_store
        ]
        self.assertTrue(
            any(n in {"get_summary", "get_window"} for n in bound_names),
            f"memory_read_node must wrap memory_store.get_summary/get_window; got {bound_names}",
        )

    async def test_multi_retrieve_node_wraps_rag_retrieve_in_to_thread(self):
        from rag.nodes import retrieval_node
        from rag.nodes.retrieval_node import multi_retrieve_node
        wrapped: list = []
        spy = await self._spy_to_thread(wrapped, exc=asyncio.CancelledError())

        class _StubRAG:
            def retrieve(self, *a, **kw):
                return type("R", (), {"chunk_indices": [], "documents": [], "metadatas": []})()
            def _deduplicate_indices(self, idxs):
                return list(idxs)
        _stub_rag = _StubRAG()
        class _StubAgent:
            rag = _stub_rag
        state = {
            "conversation_id": "x",
            "query": "test",
            "normalized_query": "test",
            "decomposed_queries": ["test"],
            "metadata_filter": {},
            "doc_type_priority": [],
            "target_tickers": [],
            "stats": {},
        }
        with unittest.mock.patch.object(retrieval_node.asyncio, "to_thread", side_effect=spy):
            try:
                await multi_retrieve_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # Verify rag.retrieve was wrapped (bound method with __self__ = _stub_rag)
        retrieve_calls = [
            f for f in wrapped
            if getattr(f, "__name__", "") == "retrieve"
            and getattr(f, "__self__", None) is _stub_rag
        ]
        self.assertGreaterEqual(
            len(retrieve_calls), 1,
            f"multi_retrieve_node must wrap rag.retrieve in asyncio.to_thread; wrapped={[getattr(f, '__name__', '') for f in wrapped]}",
        )


if __name__ == "__main__":
    unittest.main()
