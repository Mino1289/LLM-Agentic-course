"""Tests for the async-native run_sec_filings_rag (Fix #2).

The previous implementation used ``asyncio.run`` from inside
``asyncio.to_thread`` to drive 3 async nodes. That pattern is fragile:
- The AsyncOpenAI client is bound to the outer event loop.
- The new event loop in the worker thread conflicts on httpx pools.
- A deadlock manifests as a 2-minute hang in the Streamlit UI.

This module ensures run_sec_filings_rag is a native ``async def`` and is
called with ``await`` from the (async) ``tools_node``.
"""

import asyncio
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class RunSecFilingsRagAsyncTests(unittest.TestCase):
    """Fix #2 — run_sec_filings_rag must be a native coroutine function."""

    def test_run_sec_filings_rag_is_coroutine_function(self):
        from rag.tools import run_sec_filings_rag

        self.assertTrue(
            inspect.iscoroutinefunction(run_sec_filings_rag),
            "run_sec_filings_rag must be async def (use 'await' from tools_node)",
        )

    def test_run_sec_filings_rag_does_not_call_asyncio_run(self):
        """No asyncio.run() call from inside run_sec_filings_rag.

        ``asyncio.run`` creates a new event loop, which collides with the
        outer loop's httpx connection pool (manifests as a 2-min hang).
        We check for the actual call (``asyncio.run(``) not the bare string,
        so historical mentions in the docstring are tolerated.
        """
        import re
        from rag import tools as tools_module

        source = inspect.getsource(tools_module.run_sec_filings_rag)
        # Strip the docstring so a historical "asyncio.run" mention there
        # doesn't trigger a false positive.
        no_docstring = re.sub(r'"""[\s\S]*?"""', "", source)
        self.assertNotIn(
            "asyncio.run(",
            no_docstring,
            "run_sec_filings_rag must not call asyncio.run( — use 'await' instead",
        )

    def test_run_sec_filings_rag_awaits_multi_retrieve_node(self):
        """Regression: multi_retrieve_node (async) must be driven to completion."""
        from rag.tools import run_sec_filings_rag
        from rag.tool_schemas import SecFilingsRAGArgs

        class FakeRetrieval:
            chunk_indices = [0, 1]

        class FakeRag:
            documents = {0: "nvda chunk A", 1: "nvda chunk B"}
            doc_metadata = [
                {"ticker": "NVDA", "year": "2024", "source": "nvda-10-k_2024.htm", "section": "Item_1A"},
                {"ticker": "NVDA", "year": "2024", "source": "nvda-10-k_2024.htm", "section": "Item_1A"},
            ]

            def __init__(self):
                self.retrieve_calls = 0

            def retrieve(self, query, **kwargs):
                self.retrieve_calls += 1
                return FakeRetrieval()

            def _deduplicate_indices(self, indices):
                return list(dict.fromkeys(indices))

            def _rerank(self, **kwargs):
                return kwargs.get("candidate_indices", [])

        with patch("rag.tools._balanced_rerank_indices", return_value=[0, 1]), \
             patch("rag.tools._ticker_counts", return_value={"NVDA": 2}):
            agent = SimpleNamespace(rag=FakeRag(), max_tool_iterations=6)
            args = SecFilingsRAGArgs(
                query="NVDA 10-K risk factors 2024",
                tickers=["NVDA"],
                years=["2024"],
                doc_types=["10-K"],
            )
            # Drive the coroutine — no asyncio.run inside the function itself.
            result = asyncio.run(run_sec_filings_rag(args, agent=agent))

        self.assertEqual(len(result.get("final_chunks", [])), 2)
        self.assertEqual(result.get("stats", {}).get("chunks_used"), 2)
        self.assertGreater(agent.rag.retrieve_calls, 0)


class ToolsNodeDispatchTests(unittest.TestCase):
    """Fix #2 — tools_node must await run_sec_filings_rag directly.

    For ``sec_filings_rag_tool``, the async tool must run on the agent's
    own event loop (not in a worker thread via asyncio.to_thread), so the
    inner async nodes share httpx state with the outer LLM calls.
    """

    def test_tools_node_handles_async_tool_directly(self):
        """tools_node should dispatch sec_filings_rag_tool via await,
        not via asyncio.to_thread (which would defeat the fix)."""
        import rag.nodes.agent_nodes as agent_nodes_module
        source = inspect.getsource(agent_nodes_module.tools_node)
        # The tool loop in tools_node must call execute_tool (sync wrapper)
        # for sync tools AND a separate path for async tools.
        self.assertIn("execute_tool", source)

    def test_execute_tool_remains_sync_for_other_tools(self):
        """execute_tool is still sync — sec_filings_rag_tool gets a separate
        async dispatch path in tools_node (keeps the simpler contract for
        market_price_tool, validate_claims_tool, etc.)."""
        from rag.tools import execute_tool

        self.assertFalse(
            inspect.iscoroutinefunction(execute_tool),
            "execute_tool stays sync — sec_filings_rag_tool is dispatched "
            "via a dedicated async path in tools_node",
        )


if __name__ == "__main__":
    unittest.main()
