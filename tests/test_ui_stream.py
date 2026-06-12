"""Tests for the run_stream UI helper (PRD etape 6 §3.5).

The helper consumes agent.astream() via asyncio.run and dispatches events
to a text container (anything with .markdown(str)) and a status container
(anything with .update(label=str)). It returns the final GraphState
captured from the on_graph_end event.

These tests are RED at H5 (run_stream not implemented). They turn GREEN
after H6.
"""

import asyncio
import unittest
import unittest.mock


class _FakeTextContainer:
    def __init__(self):
        self.markdown_calls: list[str] = []

    def markdown(self, text: str, **kwargs):
        self.markdown_calls.append(text)


class _FakeStatusContainer:
    def __init__(self):
        self.update_calls: list[str] = []

    def update(self, *, label: str, **kwargs):
        self.update_calls.append(label)


def _build_agent_with_astream_events(raw_events: list[dict]):
    """Build a FinanceLangGraphAgent with a graph.astream_events that
    yields the given raw events."""
    from src.graph.flow import FinanceLangGraphAgent
    agent = FinanceLangGraphAgent.__new__(FinanceLangGraphAgent)

    async def fake_astream_events(state, version=None):
        for ev in raw_events:
            yield ev

    mock_graph = unittest.mock.MagicMock()
    mock_graph.astream_events = fake_astream_events
    agent.graph = mock_graph
    return agent


class RunStreamHelperTests(unittest.TestCase):
    def test_run_stream_invokes_asyncio_run(self):
        from ui.streaming import run_stream
        agent = _build_agent_with_astream_events([
            {"event": "on_chain_start", "name": "agent_node", "data": {}},
            {"event": "on_graph_end", "state": {"answer": "ok", "tool_events": []}},
        ])
        text = _FakeTextContainer()
        status = _FakeStatusContainer()
        # run_stream is sync (uses asyncio.run internally)
        final = run_stream(agent, "query", "cid", [], text, status)
        self.assertIn("answer", final)
        self.assertEqual(final["answer"], "ok")

    def test_run_stream_flushes_word_buffer_on_separator(self):
        from ui.streaming import run_stream
        # Stream yields "Hello" " " "world" "!" as 4 separate on_llm_token events
        agent = _build_agent_with_astream_events([
            {"event": "on_llm_token", "token": "Hello"},
            {"event": "on_llm_token", "token": " "},
            {"event": "on_llm_token", "token": "world"},
            {"event": "on_llm_token", "token": "."},
            {"event": "on_graph_end", "state": {"answer": "Hello world.", "tool_events": []}},
        ])
        text = _FakeTextContainer()
        status = _FakeStatusContainer()
        run_stream(agent, "query", "cid", [], text, status)
        # The text container must have been called at least twice:
        #  - first with "Hello " (separator triggered)
        #  - then with "Hello world." (separator or final flush)
        self.assertGreaterEqual(
            len(text.markdown_calls), 2,
            f"run_stream must flush word buffer on separator; got {text.markdown_calls}",
        )
        # Last markdown call should contain the full streamed text
        last = text.markdown_calls[-1]
        self.assertIn("Hello world", last)

    def test_run_stream_updates_status_on_chain_start(self):
        from ui.streaming import run_stream
        agent = _build_agent_with_astream_events([
            {"event": "on_chain_start", "name": "agent_node", "data": {}},
            {"event": "on_chain_start", "name": "tools_node", "data": {}},
            {"event": "on_graph_end", "state": {"answer": "x", "tool_events": []}},
        ])
        text = _FakeTextContainer()
        status = _FakeStatusContainer()
        run_stream(agent, "query", "cid", [], text, status)
        # Status container must have been updated with the node names
        self.assertGreaterEqual(
            len(status.update_calls), 2,
            f"status must be updated on each on_chain_start; got {status.update_calls}",
        )
        self.assertTrue(
            any("agent_node" in u for u in status.update_calls),
            f"status must mention agent_node; got {status.update_calls}",
        )

    def test_run_stream_returns_state_from_on_graph_end(self):
        from ui.streaming import run_stream
        final_state = {
            "answer": "Réponse finale",
            "tool_events": [
                {"tool": "market_price_tool", "status": "completed"},
            ],
            "final_chunks": ["x"],
            "stats": {"chunks_used": 1},
        }
        agent = _build_agent_with_astream_events([
            {"event": "on_graph_end", "state": final_state},
        ])
        text = _FakeTextContainer()
        status = _FakeStatusContainer()
        final = run_stream(agent, "query", "cid", [], text, status)
        # The returned state must be the one from the on_graph_end event
        self.assertEqual(final.get("answer"), "Réponse finale")
        self.assertEqual(len(final.get("tool_events")), 1)
        self.assertEqual(final.get("stats", {}).get("chunks_used"), 1)


if __name__ == "__main__":
    unittest.main()
