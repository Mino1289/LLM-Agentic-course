"""Lifecycle tool_events: running → completed / failed (PRD etape 4 §3.3)."""

import asyncio
import json
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch


def _build_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "lc_messages": [],
        "pending_tool_calls": [],
        "tool_events": [],
        "stats": {},
        "final_chunks": [],
        "final_metadatas": [],
        "price_context": "",
        "report_artifacts": [],
    }
    state.update(overrides)
    return state


async def _inline_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


class ToolEventLifecycleTests(unittest.TestCase):
    def test_running_then_completed(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.tool_execution_node import tools_node

        agent = MagicMock()
        tc = ToolCall(id="call_1", name="market_price_tool", arguments=json.dumps({
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.tool_executor.execute_tool") as mock_execute, \
             patch("rag.tool_executor.asyncio.to_thread", side_effect=_inline_to_thread):
            mock_execute.return_value = {"text": "ok", "price_context": "ctx"}
            result = asyncio.run(tools_node(agent, state))

        events = result["tool_events"]
        self.assertGreaterEqual(len(events), 1)
        last = events[-1]
        self.assertEqual(last["tool"], "market_price_tool")
        self.assertEqual(last["status"], "completed")
        self.assertIn("started_at", last)
        self.assertIn("finished_at", last)
        # timestamps are ISO UTC; finished_at should be >= started_at
        self.assertGreaterEqual(
            datetime.fromisoformat(last["finished_at"]),
            datetime.fromisoformat(last["started_at"]),
        )

    def test_running_then_failed_on_execution_exception(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.tool_execution_node import tools_node

        agent = MagicMock()
        tc = ToolCall(id="call_1", name="market_price_tool", arguments=json.dumps({
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.tool_executor.execute_tool") as mock_execute, \
             patch("rag.tool_executor.asyncio.to_thread", side_effect=_inline_to_thread):
            mock_execute.side_effect = RuntimeError("provider down")
            result = asyncio.run(tools_node(agent, state))

        events = result["tool_events"]
        last = events[-1]
        self.assertEqual(last["status"], "failed")
        self.assertIn("provider down", last["error"])
        self.assertIn("finished_at", last)

    def test_running_then_failed_on_validation_error(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.tool_execution_node import tools_node

        agent = MagicMock()
        # Missing required 'query' field → Pydantic validation should fail
        tc = ToolCall(id="call_1", name="sec_filings_rag_tool", arguments=json.dumps({}))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.tool_executor.execute_tool") as mock_execute, \
             patch("rag.tool_executor.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(tools_node(agent, state))

        events = result["tool_events"]
        last = events[-1]
        self.assertEqual(last["status"], "failed")
        self.assertIn("args_validation", last["error"])
        # execute_tool must NOT be called when validation fails
        mock_execute.assert_not_called()

    def test_injected_chunks_and_metadatas_passed_to_execute_tool(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.tool_execution_node import tools_node

        agent = MagicMock()
        tc = ToolCall(
            id="call_1",
            name="validate_claims_tool",
            arguments=json.dumps({"claims": ["MSFT risk"]}),
        )

        state = _build_state(
            pending_tool_calls=[tc],
            final_chunks=["Item 1A risk."],
            final_metadatas=[{"ticker": "MSFT", "year": "2024"}],
        )

        with patch("rag.tool_executor.execute_tool") as mock_execute, \
             patch("rag.tool_executor.asyncio.to_thread", side_effect=_inline_to_thread):
            mock_execute.return_value = {"text": "ok", "validations": []}
            asyncio.run(tools_node(agent, state))
            args = mock_execute.call_args[0][1]  # second positional arg

        # args is a ValidateClaimsArgs BaseModel
        self.assertEqual(args.claims, ["MSFT risk"])
        self.assertEqual(args.chunks, ["Item 1A risk."])
        self.assertEqual(args.metadatas, [{"ticker": "MSFT", "year": "2024"}])


if __name__ == "__main__":
    unittest.main()
