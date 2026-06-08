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
    def test_duplicate_tool_calls_are_skipped_after_first_success(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.tool_execution_node import tools_node

        agent = MagicMock()
        state = _build_state(
            pending_tool_calls=[
                ToolCall(
                    id="call_1",
                    name="simulate_portfolio_tool",
                    arguments=json.dumps(
                        {
                            "allocations": {"NVDA": 40, "AMD": 30, "MSFT": 30},
                            "notional_usd": 100000,
                        }
                    ),
                ),
                ToolCall(
                    id="call_2",
                    name="simulate_portfolio_tool",
                    arguments=json.dumps(
                        {
                            "notional_usd": 100000,
                            "allocations": {"NVDA": 40, "AMD": 30, "MSFT": 30},
                        }
                    ),
                ),
            ]
        )

        with patch("rag.tool_executor.execute_tool") as mock_execute, \
             patch("rag.tool_executor.asyncio.to_thread", side_effect=_inline_to_thread):
            mock_execute.return_value = {
                "text": "simulated",
                "positions": [{"ticker": "NVDA"}],
            }
            result = asyncio.run(tools_node(agent, state))

        mock_execute.assert_called_once()
        self.assertEqual([msg["tool_call_id"] for msg in result["lc_messages"]], ["call_1", "call_2"])
        self.assertEqual(result["tool_events"][-1]["status"], "skipped")
        self.assertEqual(result["tool_events"][-1]["reason"], "duplicate_tool_call")
        self.assertEqual(result["stats"]["duplicate_tool_calls_skipped"], 1)

    def test_rag_tool_results_are_accumulated_and_deduplicated(self) -> None:
        from rag.nodes.tool_execution_node import _merge_tool_side_effects

        stats: dict[str, Any] = {}
        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []
        price_context = ""
        artifacts: list[dict[str, Any]] = []

        chunks, metadatas, price_context, artifacts, stats = _merge_tool_side_effects(
            "sec_filings_rag_tool",
            {
                "final_chunks": ["nvda risk"],
                "final_metadatas": [
                    {
                        "ticker": "NVDA",
                        "year": "2024",
                        "file_type": "10-K",
                        "section": "Item_1A",
                        "source": "nvda-10-k_2024.htm",
                    }
                ],
                "stats": {
                    "retrieval_candidate_count": 1,
                    "retrieval_candidate_ticker_counts": {"NVDA": 1},
                },
            },
            final_chunks=chunks,
            final_metadatas=metadatas,
            price_context=price_context,
            report_artifacts=artifacts,
            stats=stats,
        )

        chunks, metadatas, price_context, artifacts, stats = _merge_tool_side_effects(
            "sec_filings_rag_tool",
            {
                "final_chunks": ["amd risk"],
                "final_metadatas": [
                    {
                        "ticker": "AMD",
                        "year": "2024",
                        "file_type": "10-K",
                        "section": "Item_1A",
                        "source": "amd-10-k_2024.htm",
                    }
                ],
                "stats": {
                    "retrieval_candidate_count": 1,
                    "retrieval_candidate_ticker_counts": {"AMD": 1},
                },
            },
            final_chunks=chunks,
            final_metadatas=metadatas,
            price_context=price_context,
            report_artifacts=artifacts,
            stats=stats,
        )

        chunks, metadatas, price_context, artifacts, stats = _merge_tool_side_effects(
            "sec_filings_rag_tool",
            {
                "final_chunks": ["nvda risk"],
                "final_metadatas": [
                    {
                        "ticker": "NVDA",
                        "year": "2024",
                        "file_type": "10-K",
                        "section": "Item_1A",
                        "source": "nvda-10-k_2024.htm",
                    }
                ],
                "stats": {
                    "retrieval_candidate_count": 1,
                    "retrieval_candidate_ticker_counts": {"NVDA": 1},
                },
            },
            final_chunks=chunks,
            final_metadatas=metadatas,
            price_context=price_context,
            report_artifacts=artifacts,
            stats=stats,
        )

        self.assertEqual([m["ticker"] for m in metadatas], ["NVDA", "AMD"])
        self.assertEqual(chunks, ["nvda risk", "amd risk"])
        self.assertEqual(stats["rerank_final_ticker_counts"], {"NVDA": 1, "AMD": 1})
        self.assertEqual(stats["chunks_used"], 2)

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
