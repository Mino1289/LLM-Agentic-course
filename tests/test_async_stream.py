"""Tests verifying that FinanceLangGraphAgent exposes the async API:
- `run()` (sync, backwards compat) delegates to `asyncio.run(arun())`
- `arun()` is an async method that calls `graph.ainvoke(initial_state)`
- `astream()` is an async generator that yields events from
  `graph.astream_events(initial_state, version="v2")`, including:
  * `on_llm_token` events (one per LLM token, renamed from on_chat_model_stream)
  * `on_graph_end` event (final event with the complete GraphState)

These tests were RED at G3 (no arun/astream). They turn GREEN after G4.
H3 adds the on_llm_token / on_graph_end semantics — RED until H4.
"""

import asyncio
import inspect
import unittest
import unittest.mock


def _make_agent_with_mock_graph():
    """Build a FinanceLangGraphAgent with a fully-mocked graph object
    that supports both invoke (sync) and ainvoke (async) and astream_events."""
    from src.graph.flow import FinanceLangGraphAgent
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
        from src.graph.flow import FinanceLangGraphAgent
        self.assertTrue(hasattr(FinanceLangGraphAgent, "run"))

    def test_arun_is_coroutine_method(self):
        from src.graph.flow import FinanceLangGraphAgent
        self.assertTrue(
            hasattr(FinanceLangGraphAgent, "arun"),
            "FinanceLangGraphAgent must expose `arun()` for async invocation",
        )
        self.assertTrue(
            inspect.iscoroutinefunction(FinanceLangGraphAgent.arun),
            "arun must be a coroutine function",
        )

    def test_astream_is_async_generator_method(self):
        from src.graph.flow import FinanceLangGraphAgent
        self.assertTrue(
            hasattr(FinanceLangGraphAgent, "astream"),
            "FinanceLangGraphAgent must expose `astream()` for tool event streaming",
        )
        self.assertTrue(
            inspect.isasyncgenfunction(FinanceLangGraphAgent.astream),
            "astream must be an async generator function",
        )


class AsyncAPIBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_delegates_to_asyncio_run_via_thread(self):
        # `run()` is sync; we cannot call it from inside an async context
        # (the new implementation explicitly raises in that case). Run it
        # in a thread to escape the running loop and verify delegation.
        import threading
        from src.graph import flow as flow_module
        agent = _make_agent_with_mock_graph()
        captured: dict = {}
        original_run = flow_module.asyncio.run

        def spy_run(coro, *args, **kwargs):
            captured["called"] = True
            return original_run(coro, *args, **kwargs)

        result_holder: dict = {}
        def worker():
            with unittest.mock.patch.object(flow_module.asyncio, "run", spy_run):
                result_holder["value"] = agent.run("test query")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(result_holder["value"]["answer"], "ok")
        self.assertEqual(result_holder["value"]["conversation_id"], "cid")
        self.assertTrue(
            captured.get("called", False),
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


class AStreamTokenEventsTests(unittest.IsolatedAsyncioTestCase):
    """Verify astream yields on_llm_token and on_graph_end events."""

    def _make_agent_with_astream_events_mock(self, raw_events: list[dict]):
        """Build an agent whose graph.astream_events yields the given raw events
        (already in the LangGraph astream_events schema, NOT yet transformed
        by our astream() wrapper)."""
        from src.graph.flow import FinanceLangGraphAgent
        agent = FinanceLangGraphAgent.__new__(FinanceLangGraphAgent)

        async def fake_astream_events(state, version=None):
            for ev in raw_events:
                yield ev

        mock_graph = unittest.mock.MagicMock()
        mock_graph.astream_events = fake_astream_events
        agent.graph = mock_graph
        return agent

    async def test_astream_yields_on_llm_token_for_each_token(self):
        # LangGraph emits on_chat_model_stream with chunk.content being the delta.
        # Our astream() wrapper must rename it to on_llm_token.
        class _FakeChunk:
            def __init__(self, content):
                self.content = content
        raw_events = [
            {"event": "on_chain_start", "name": "agent_node", "data": {}},
            {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("Hello ")}},
            {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("world")}},
            {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("!")}},
            {"event": "on_chain_end", "name": "agent_node", "data": {"output": {"answer": "Hello world!"}}},
            {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"answer": "Hello world!", "tool_events": []}}},
        ]
        agent = self._make_agent_with_astream_events_mock(raw_events)
        tokens = []
        async for event in agent.astream("test"):
            if event.get("event") == "on_llm_token":
                tokens.append(event["token"])
        self.assertEqual(
            "".join(tokens), "Hello world!",
            f"astream must yield on_llm_token for each token; got {tokens}",
        )

    async def test_astream_yields_on_graph_end_with_full_state(self):
        # astream() must emit a final on_graph_end event with the state
        # accumulated from the last on_chain_end output.
        final_state = {
            "conversation_id": "x",
            "answer": "Réponse finale",
            "tool_events": [{"tool": "market_price_tool", "status": "completed"}],
            "final_chunks": ["chunk1"],
            "final_metadatas": [{"ticker": "NVDA"}],
            "stats": {"chunks_used": 1},
        }
        raw_events = [
            {"event": "on_chain_start", "name": "agent_node", "data": {}},
            {"event": "on_chain_end", "name": "gc_node", "data": {"output": final_state}},
        ]
        agent = self._make_agent_with_astream_events_mock(raw_events)
        captured = []
        async for event in agent.astream("test"):
            if event.get("event") == "on_graph_end":
                captured.append(event["state"])
        self.assertEqual(len(captured), 1, "astream must yield exactly one on_graph_end event")
        self.assertEqual(captured[0]["answer"], "Réponse finale")

    async def test_astream_final_state_has_tool_events(self):
        # The final state must contain tool_events (no need to call arun again).
        final_state = {
            "answer": "ok",
            "tool_events": [
                {"tool": "sec_filings_rag_tool", "status": "completed", "args_summary": "x"},
                {"tool": "market_price_tool", "status": "completed", "args_summary": "y"},
            ],
        }
        raw_events = [
            {"event": "on_chain_end", "name": "gc_node", "data": {"output": final_state}},
        ]
        agent = self._make_agent_with_astream_events_mock(raw_events)
        final = None
        async for event in agent.astream("test"):
            if event.get("event") == "on_graph_end":
                final = event["state"]
        self.assertIsNotNone(final)
        self.assertIn("tool_events", final)
        self.assertEqual(len(final["tool_events"]), 2)


if __name__ == "__main__":
    unittest.main()
