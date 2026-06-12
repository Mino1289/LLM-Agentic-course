"""Tests for the quick-win performance fixes QW1+QW2+QW3.

Context
-------
After Fix #1 (request timeout 30s) and Fix #2 (native async for
``run_sec_filings_rag``), multi-ticker queries still take 25-50s because of
4 sequential LLM calls:

    1. agent_node   (initial routing + tool selection)
    2. decompose    (query expansion → 4-6 sub-queries)
    3. retrieve     (multi_retrieve_node against ChromaDB)
    4. rerank       (LLM-based re-ranking)
    5. final        (final synthesis in agent_node)

Quick wins:

- **QW1**: Lower default ``decompose_query_count`` from 4 to 2 (cuts
  retrieval count by ~50 % for multi-ticker queries).
- **QW2**: Skip ``decompose_query`` whenever a single ticker is detected
  (saves one full LLM call = ~3-7s for year-agnostic ticker queries like
  ``"Quel est le revenu Azure FY2025 vs FY2024 ?"``).
- **QW3**: Add per-step timing logs in ``run_sec_filings_rag`` and
  ``agent_node`` so latency regressions are visible in production logs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class DecomposeQueryCountDefaultTests(unittest.TestCase):
    """QW1 — the default and floor of ``decompose_query_count`` drop from 4/3 to 2/1."""

    def test_default_decompose_query_count_is_2(self):
        """The constructor default must be 2 (was 4)."""
        from src.graph.flow import FinanceLangGraphAgent

        # Pass a stub RAG so we don't load ChromaDB.
        rag_stub = SimpleNamespace(provider=MagicMock(), documents=[], doc_metadata=[])
        with patch.object(FinanceLangGraphAgent, "_build_graph", return_value=MagicMock()):
            agent = FinanceLangGraphAgent(rag=rag_stub)
        self.assertEqual(agent.decompose_query_count, 2)

    def test_explicit_decompose_query_count_2_is_preserved(self):
        """Passing 2 must not be floored up to 3."""
        from src.graph.flow import FinanceLangGraphAgent

        rag_stub = SimpleNamespace(provider=MagicMock(), documents=[], doc_metadata=[])
        with patch.object(FinanceLangGraphAgent, "_build_graph", return_value=MagicMock()):
            agent = FinanceLangGraphAgent(rag=rag_stub, decompose_query_count=2)
        self.assertEqual(agent.decompose_query_count, 2)

    def test_explicit_decompose_query_count_1_is_preserved(self):
        """Passing 1 must not be floored up to 3 (floor is now 1)."""
        from src.graph.flow import FinanceLangGraphAgent

        rag_stub = SimpleNamespace(provider=MagicMock(), documents=[], doc_metadata=[])
        with patch.object(FinanceLangGraphAgent, "_build_graph", return_value=MagicMock()):
            agent = FinanceLangGraphAgent(rag=rag_stub, decompose_query_count=1)
        self.assertEqual(agent.decompose_query_count, 1)

    def test_studio_env_default_uses_2(self):
        """The Studio ``QUERY_DECOMPOSE_COUNT`` env var defaults to 2 (was 4).

        Patch ``load_dotenv`` (imported via ``rag.langsmith_env``) to keep
        the test hermetic — the real ``.env`` file at the repo root may
        have a different value that would leak in via
        ``ensure_langsmith_env()``.
        """
        env = {k: v for k, v in os.environ.items() if k != "QUERY_DECOMPOSE_COUNT"}
        with patch.dict(os.environ, env, clear=True), patch(
            "rag.langsmith_env.load_dotenv"
        ), patch("src.rag.core.HybridRAG"), patch(
            "rag.langgraph_studio.FinanceLangGraphAgent"
        ) as agent_cls:
            agent_cls.return_value = MagicMock(graph=MagicMock())
            from src.graph.flow import build_graph

            build_graph()
        self.assertEqual(
            agent_cls.call_args.kwargs["decompose_query_count"],
            2,
            "Studio default must be 2, not 4",
        )


class RunSecFilingsRagSkipDecomposeTests(unittest.TestCase):
    """QW2 — skip ``decompose_query`` whenever a single ticker is detected."""

    def _make_agent(self) -> SimpleNamespace:
        return SimpleNamespace(
            rag=SimpleNamespace(provider=MagicMock(), documents=[], doc_metadata=[]),
            decompose_query_count=2,
        )

    def _run(self, agent, args):
        from src.tools.sec_filings import run_sec_filings_rag

        return asyncio.run(run_sec_filings_rag(args, agent=agent))

    def test_single_ticker_with_two_years_skips_decompose(self):
        """The Azure FY2025 vs FY2024 use case: 1 ticker + 2 years → no decompose."""
        from src.tools.schemas import SecFilingsRAGArgs

        agent = self._make_agent()
        with patch("src.graph.decompose_node.decompose_query", new=AsyncMock()) as mock_decompose, patch(
            "src.tools.sec_filings.multi_retrieve_node", new=AsyncMock(return_value={"candidate_indices": []})
        ):
            result = self._run(
                agent,
                SecFilingsRAGArgs(
                    query="Quel est le revenu Azure (Microsoft) FY2025 vs FY2024 ?",
                    tickers=["MSFT"],
                    years=["2024", "2025"],
                ),
            )
        mock_decompose.assert_not_called()
        self.assertEqual(
            result["stats"].get("decomposed_count"),
            1,
            "Single-ticker queries must skip LLM decomposition",
        )

    def test_single_ticker_with_one_year_skips_decompose(self):
        """The previous behaviour (1 ticker + 1 year) must still skip decompose."""
        from src.tools.schemas import SecFilingsRAGArgs

        agent = self._make_agent()
        with patch("src.graph.decompose_node.decompose_query", new=AsyncMock()) as mock_decompose, patch(
            "src.tools.sec_filings.multi_retrieve_node", new=AsyncMock(return_value={"candidate_indices": []})
        ):
            self._run(
                agent,
                SecFilingsRAGArgs(
                    query="MSFT 10-K risk factors 2024",
                    tickers=["MSFT"],
                    years=["2024"],
                ),
            )
        mock_decompose.assert_not_called()

    def test_two_tickers_calls_decompose(self):
        """Multi-ticker queries still benefit from decomposition."""
        from src.tools.schemas import SecFilingsRAGArgs

        agent = self._make_agent()
        with patch("src.graph.decompose_node.decompose_query", new=AsyncMock(return_value=["q1", "q2"])) as mock_decompose, patch(
            "src.tools.sec_filings.multi_retrieve_node",
            new=AsyncMock(return_value={"candidate_indices": []}),
        ):
            self._run(
                agent,
                SecFilingsRAGArgs(
                    query="Compare NVDA et MSFT sur la croissance",
                    tickers=["NVDA", "MSFT"],
                ),
            )
        mock_decompose.assert_called_once()

    def test_no_ticker_with_years_calls_decompose(self):
        """Universe-wide year queries still need decomposition."""
        from src.tools.schemas import SecFilingsRAGArgs

        agent = self._make_agent()
        with patch("src.graph.decompose_node.decompose_query", new=AsyncMock(return_value=["q1", "q2"])) as mock_decompose, patch(
            "src.tools.sec_filings.multi_retrieve_node",
            new=AsyncMock(return_value={"candidate_indices": []}),
        ):
            self._run(
                agent,
                SecFilingsRAGArgs(
                    query="Quels sont les risques du secteur semi-conducteurs en 2024 ?",
                    years=["2024"],
                ),
            )
        mock_decompose.assert_called_once()


class RunSecFilingsRagTimingLogsTests(unittest.TestCase):
    """QW3 — per-step timing logs in ``run_sec_filings_rag``."""

    def test_logs_timing_for_each_stage(self):
        """Decompose, retrieve, and rerank stages each emit a duration log."""
        from src.tools.schemas import SecFilingsRAGArgs

        from src.tools.sec_filings import run_sec_filings_rag

        agent = SimpleNamespace(
            rag=SimpleNamespace(provider=MagicMock(), documents=[], doc_metadata=[]),
            decompose_query_count=2,
        )

        async def fake_decompose(a, q):
            return ["q1", "q2"]

        async def fake_retrieve(a, state):
            state["candidate_indices"] = [0, 1]
            state["stats"] = {"retrieve": "ok"}
            return state

        async def fake_rerank(a, state, candidates):
            return [0, 1]

        agent.rag.documents = ["doc0", "doc1"]
        agent.rag.doc_metadata = [{"ticker": "NVDA"}, {"ticker": "MSFT"}]

        with patch("src.graph.decompose_node.decompose_query", side_effect=fake_decompose), patch(
            "src.tools.sec_filings.multi_retrieve_node", side_effect=fake_retrieve
        ), patch("src.graph.rerank_node._balanced_rerank_indices", side_effect=fake_rerank), self.assertLogs(
            "rag.tools", level="INFO"
        ) as cm:
            asyncio.run(
                run_sec_filings_rag(
                    SecFilingsRAGArgs(
                        query="Compare NVDA et MSFT",
                        tickers=["NVDA", "MSFT"],
                    ),
                    agent=agent,
                )
            )

        log_blob = "\n".join(cm.output)
        self.assertIn("decompose", log_blob.lower())
        self.assertIn("retrieve", log_blob.lower())
        self.assertIn("rerank", log_blob.lower())


if __name__ == "__main__":
    unittest.main()
