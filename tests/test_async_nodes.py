"""Tests verifying that all graph nodes are async coroutines and that
they wrap blocking I/O calls via `asyncio.to_thread`.

These tests are RED at G1: all 8 assertions fail because nodes are still
sync `def`. They will turn GREEN after G2 (refactor to `async def`).
"""

import asyncio
import inspect
import unittest


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

    async def test_agent_node_wraps_provider_call_in_to_thread(self):
        from rag.nodes import agent_nodes
        # Track every function passed to asyncio.to_thread
        wrapped = []
        original_to_thread = asyncio.to_thread

        async def spy_to_thread(func, *args, **kwargs):
            wrapped.append(func)
            # We don't actually run the provider; return a coroutine-friendly
            # sentinel. The test asserts presence of the wrapper only.
            raise asyncio.CancelledError

        # Build a minimal agent stub
        from rag.nodes.agent_nodes import _build_lc_messages
        class _StubProvider:
            def invoke_with_tools(self, *a, **kw):
                return None
        class _StubAgent:
            class rag:
                provider = _StubProvider()
        agent = _StubAgent()
        # Build a minimal state
        state = {
            "conversation_id": "x",
            "query": "Hello",
            "messages": [],
            "tool_events": [],
            "agent_iterations": 0,
        }
        # Patch asyncio.to_thread used inside agent_nodes module
        with unittest.mock.patch.object(agent_nodes.asyncio, "to_thread", side_effect=spy_to_thread):
            try:
                await agent_nodes.agent_node(agent, state)
            except asyncio.CancelledError:
                pass
        # At least one wrap call should target the provider's invoke_with_tools
        bound = [getattr(f, "__self__", None) for f in wrapped]
        provider_methods = [b.invoke_with_tools for b in bound if b is not None and hasattr(b, "invoke_with_tools")]
        self.assertGreaterEqual(
            len(provider_methods), 1,
            "agent_node must wrap provider.invoke_with_tools in asyncio.to_thread",
        )

    async def test_tools_node_wraps_execute_tool_in_to_thread(self):
        from rag.nodes import agent_nodes
        from rag.tools import execute_tool
        wrapped = []

        async def spy_to_thread(func, *args, **kwargs):
            wrapped.append(func)
            raise asyncio.CancelledError

        # Build a minimal state with a pending tool call
        state = {
            "conversation_id": "x",
            "query": "test",
            "messages": [],
            "tool_events": [],
            "agent_iterations": 0,
            "llm_response": {},
            "pending_tool_calls": [
                {"id": "1", "name": "get_market_price", "arguments": "{}"}
            ],
            "retrieval_chunks": [],
            "retrieval_metadatas": [],
        }
        class _StubAgent:
            pass
        with unittest.mock.patch.object(agent_nodes.asyncio, "to_thread", side_effect=spy_to_thread):
            try:
                await agent_nodes.tools_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # Verify execute_tool was wrapped
        self.assertIn(
            execute_tool, wrapped,
            "tools_node must wrap execute_tool in asyncio.to_thread",
        )

    async def test_memory_read_node_wraps_memory_store_in_to_thread(self):
        from rag.nodes import memory_nodes
        wrapped = []

        async def spy_to_thread(func, *args, **kwargs):
            wrapped.append(func)
            raise asyncio.CancelledError

        class _StubStore:
            def get_summary(self, *a, **kw): return ""
            def get_window(self, *a, **kw): return []
        class _StubAgent:
            memory_store = _StubStore()
            class rag:
                class memory:
                    profile_name = "default"
                    @staticmethod
                    def last_interactions(*a, **kw): return []
        state = {"conversation_id": "x", "query": "q", "messages": []}
        with unittest.mock.patch.object(memory_nodes.asyncio, "to_thread", side_effect=spy_to_thread):
            try:
                await memory_nodes.memory_read_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # Verify memory_store.get_summary or .get_window was wrapped
        self.assertTrue(
            any(getattr(f, "__self__", None) is _StubStore() for f in wrapped),
            "memory_read_node must wrap a memory_store method in asyncio.to_thread",
        )

    async def test_multi_retrieve_node_wraps_rag_retrieve_in_to_thread(self):
        from rag.nodes import retrieval_node
        wrapped = []

        async def spy_to_thread(func, *args, **kwargs):
            wrapped.append(func)
            raise asyncio.CancelledError

        class _StubRAG:
            def retrieve(self, *a, **kw):
                return type("R", (), {"indices": [], "documents": [], "metadatas": []})()
        class _StubAgent:
            rag = _StubRAG()
        state = {
            "query": "test",
            "search_queries": ["test"],
            "retrieval_indices": [],
            "retrieval_chunks": [],
            "retrieval_metadatas": [],
            "retrieval_origin": [],
            "retrieval_tickers": [],
            "metadata_filter": {},
            "scoped_tickers": [],
            "all_tickers": [],
            "per_query_top_k": 8,
        }
        with unittest.mock.patch.object(retrieval_node.asyncio, "to_thread", side_effect=spy_to_thread):
            try:
                await retrieval_node.multi_retrieve_node(_StubAgent(), state)
            except asyncio.CancelledError:
                pass
        # Verify rag.retrieve was wrapped
        self.assertTrue(
            any(getattr(f, "__self__", None) is _StubRAG() for f in wrapped),
            "multi_retrieve_node must wrap rag.retrieve in asyncio.to_thread",
        )


if __name__ == "__main__":
    unittest.main()
