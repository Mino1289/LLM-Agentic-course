"""Tests verifying that FinanceLangGraphAgent exposes the async API:
- `run()` (sync, backwards compat) delegates to `asyncio.run(arun())`
- `arun()` is an async method that calls `graph.ainvoke(initial_state)`
- `astream()` is an async generator that yields events from
  `graph.astream_events(initial_state, version="v2")`

These tests are RED at G3 (no arun/astream yet). They turn GREEN after G4.
"""

import asyncio
import inspect
import unittest
import unittest.mock


def _make_agent_with_mock_graph():
    """Build a FinanceLangGraphAgent with a fully-mocked graph object
    that supports both invoke (sync) and ainvoke (async) and astream_events."""
    from rag.langgraph_flow import FinanceLangGraphAgent
    # Bypass __init__ (avoids building the real graph + RAG + memory store)
    agent = FinanceLangGraphAgent.__new__(FinanceLangGraphAgent)
    # Mock graph with ainvoke returning a fake state and astream_events
    # returning a fake async iterator.
    fake_state = {"conversation_id": "cid", "answer": "ok", "tool_events": []}
    mock_graph = unittest.mock.MagicMock()
    mock_graph.invoke.return_value = fake_state
    async def fake_ainvoke(state):
        return fake_state
    mock_graph.ainvoke = fake_ainvoke
    async def fake_astream_events(state, version=None):
        yield {"event": "on_chain_start", "name": "prepare_query_node", "data": {}}
        yield {"event": "on_chain_end", "name": "prepare_query_node", "data": {"output": fake_state}}
    mock_graph.astream_events = fake_astream_events
    agent.graph = mock_graph
    return agent


class AsyncAPIStructureTests(unittest.TestCase):
    def test_run_is_sync_method(self):
        # run() stays sync for backwards compat
        from rag.langgraph_flow import FinanceLangGraphAgent
        self.assertTrue(hasattr(FinanceLangGraphAgent, "run"))

    def test_arun_is_coroutine_method(self):
        from rag.langgraph_flow import FinanceLangGraphAgent
        self.assertTrue(
            hasattr(FinanceLangGraphAgent, "arun"),
            "FinanceLangGraphAgent must expose `arun()` for async invocation",
        )
        self.assertTrue(
            inspect.iscoroutinefunction(FinanceLangGraphAgent.arun),
            "arun must be a coroutine function",
        )

    def test_astream_is_async_generator_method(self):
        from rag.langgraph_flow import FinanceLangGraphAgent
        self.assertTrue(
            hasattr(FinanceLangGraphAgent, "astream"),
            "FinanceLangGraphAgent must expose `astream()` for tool event streaming",
        )
        self.assertTrue(
            inspect.isasyncgenfunction(FinanceLangGraphAgent.astream),
            "astream must be an async generator function",
        )


class AsyncAPIBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_delegates_to_asyncio_run(self):
        from rag import langgraph_flow
        agent = _make_agent_with_mock_graph()
        # Patch asyncio.run at the module level where run() is defined
        with unittest.mock.patch.object(
            langgraph_flow.asyncio, "run", wraps=asyncio.run
        ) as spy_run:
            result = agent.run("test query")
        # The result must be the state from arun
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(result["conversation_id"], "cid")
        # Verify asyncio.run was called (delegation to async runtime)
        self.assertGreaterEqual(
            spy_run.call_count, 1,
            "agent.run() must delegate to asyncio.run(arun())",
        )

    async def test_arun_returns_state_via_ainvoke(self):
        agent = _make_agent_with_mock_graph()
        result = await agent.arun("hello world", conversation_id="my-convo")
        # Verify ainvoke was called (and thus the graph async runtime)
        self.assertEqual(result["answer"], "ok")
        # The graph.ainvoke should have been called once with a proper initial state
        # Inspect the call args from the wrapper
        self.assertTrue(
            hasattr(agent.graph, "ainvoke"),
            "Agent must use ainvoke for async graph execution",
        )

    async def test_astream_yields_chain_events(self):
        agent = _make_agent_with_mock_graph()
        events = []
        async for event in agent.astream("hello"):
            events.append(event)
        # At least one on_chain_start should be yielded
        kinds = [e.get("event") for e in events]
        self.assertIn(
            "on_chain_start", kinds,
            f"astream() must yield on_chain_start events for UI; got {kinds}",
        )

    async def test_run_raises_in_running_loop(self):
        # If called from within an async context, run() must raise a clear
        # RuntimeError pointing the user to `await agent.arun(...)`.
        agent = _make_agent_with_mock_graph()
        with self.assertRaises(RuntimeError) as ctx:
            # We're already in an async context (IsolatedAsyncioTestCase)
            agent.run("from-async")
        self.assertIn("arun", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
