import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


async def _inline_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


class GuardNodeTests(unittest.TestCase):
    def _agent(self, response: str = '{"route":"continue","reason":"ok"}'):
        provider = SimpleNamespace(generate=MagicMock(return_value=response))
        return SimpleNamespace(rag=SimpleNamespace(doc_metadata=[], provider=provider))

    def test_empty_query_clarifies(self):
        from src.graph.guard import guard_node, route_after_guard

        result = asyncio.run(guard_node(self._agent(), {"normalized_query": ""}))

        self.assertIn("préciser", result["answer"].lower())
        self.assertEqual(result["stats"]["guard_route"], "clarify")
        self.assertEqual(result["stats"]["guard_source"], "rule")
        self.assertEqual(route_after_guard(result), "finalize")

    def test_llm_coverage_question_answers_universe(self):
        from src.graph.guard import guard_node

        agent = self._agent('{"route":"coverage_info","reason":"user asks covered universe"}')
        with patch("src.graph.guard.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(
                guard_node(agent, {"normalized_query": "Quelles entreprises sont couvertes ?"})
            )

        self.assertIn("NVDA", result["answer"])
        self.assertIn("ASML", result["answer"])
        self.assertIn("MSFT", result["answer"])
        self.assertEqual(result["stats"]["guard_route"], "coverage_info")
        self.assertEqual(result["stats"]["guard_source"], "llm")
        agent.rag.provider.generate.assert_called_once()

    def test_llm_offtopic_question_is_rejected(self):
        from src.graph.guard import guard_node

        agent = self._agent('{"route":"reject_offtopic","reason":"coding request"}')
        with patch("src.graph.guard.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(
                guard_node(agent, {"normalized_query": "écrire du code python pour factorielle"})
            )

        self.assertIn("analyse financière", result["answer"])
        self.assertEqual(result["stats"]["guard_route"], "reject_offtopic")

    def test_llm_regular_finance_question_continues(self):
        from src.graph.guard import guard_node, route_after_guard

        agent = self._agent('{"route":"continue","reason":"finance comparison"}')
        with patch("src.graph.guard.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(
                guard_node(agent, {"normalized_query": "Compare NVDA et MSFT sur les risques 2024"})
            )

        self.assertNotIn("answer", result)
        self.assertEqual(result["stats"]["guard_route"], "continue")
        self.assertEqual(route_after_guard(result), "agent")

    def test_llm_parse_error_falls_back_to_continue(self):
        from src.graph.guard import guard_node, route_after_guard

        agent = self._agent("not json")
        with patch("src.graph.guard.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(
                guard_node(agent, {"normalized_query": "Peux-tu analyser NVDA en 2024 ?"})
            )

        self.assertNotIn("answer", result)
        self.assertEqual(result["stats"]["guard_route"], "continue")
        self.assertEqual(result["stats"]["guard_source"], "fallback")
        self.assertEqual(route_after_guard(result), "agent")


if __name__ == "__main__":
    unittest.main()
