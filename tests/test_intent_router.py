from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from src.llm.types import LLMStreamChunk
from src.orchestration.intent_router import _coerce_bool, intent_router_node
from src.orchestration.tool_domains import (
    detect_tool_domains,
    resolve_route_from_domains,
)


def _agent_streaming(text: str | None = None, *, raises: bool = False):
    """Agent factice dont le provider streame `text` (JSON du classifieur)."""

    async def stream(messages, tools=None, temperature=0.0, max_tokens=120):
        if raises:
            raise RuntimeError("LLM down")
        # split en deux pour vérifier la reconstruction sur plusieurs chunks
        yield LLMStreamChunk(delta=text[: len(text) // 2])
        yield LLMStreamChunk(delta=text[len(text) // 2 :])

    provider = SimpleNamespace(ainvoke_with_tools_stream=stream)
    return SimpleNamespace(rag=SimpleNamespace(provider=provider))


class DetectToolDomainsTests(unittest.TestCase):
    def test_news_only(self):
        domains = detect_tool_domains("Quelles sont les news récentes sur NVDA ?")
        self.assertEqual(domains, frozenset({"news"}))

    def test_price_only(self):
        domains = detect_tool_domains("Quel est le prix de MSFT ?")
        self.assertEqual(domains, frozenset({"quant"}))

    def test_sec_and_performance(self):
        query = "Compare les risques SEC 2024 et la perf 6 mois de NVDA et AMD"
        domains = detect_tool_domains(query)
        self.assertIn("fundamental", domains)
        self.assertIn("quant", domains)

    def test_trade_only(self):
        domains = detect_tool_domains("Achète 100 actions NVDA")
        self.assertIn("trade", domains)

    def test_analyze_and_buy(self):
        query = "Analyse NVDA et AMD et dis moi quelle action acheter"
        domains = detect_tool_domains(query)
        self.assertIn("trade", domains)


class ResolveRouteFromDomainsTests(unittest.TestCase):
    def test_multi_tool_is_complex(self):
        route, reason = resolve_route_from_domains(frozenset({"fundamental", "quant"}))
        self.assertEqual(route, "complex")
        self.assertEqual(reason, "multi_tool")

    def test_trade_is_complex(self):
        route, reason = resolve_route_from_domains(frozenset({"trade"}))
        self.assertEqual(route, "complex")
        self.assertEqual(reason, "action_keyword")

    def test_single_domain_is_simple(self):
        route, reason = resolve_route_from_domains(frozenset({"news"}))
        self.assertEqual(route, "simple")
        self.assertEqual(reason, "single_tool")

    def test_empty_needs_llm(self):
        self.assertIsNone(resolve_route_from_domains(frozenset()))


class IntentRouterIntegrationTests(unittest.TestCase):
    def test_end_to_end_routes(self):
        cases = [
            ("News NVDA", "simple", "single_tool"),
            ("Prix MSFT", "simple", "single_tool"),
            (
                "Compare risques SEC 2024 et perf 6 mois NVDA AMD",
                "complex",
                "multi_tool",
            ),
            ("Achète 100 NVDA", "complex", "action_keyword"),
            (
                "Analyse NVDA et AMD et quelle action acheter",
                "complex",
                "action_keyword",
            ),
        ]
        for query, expected_route, expected_reason in cases:
            with self.subTest(query=query):
                domains = detect_tool_domains(query)
                resolved = resolve_route_from_domains(domains)
                self.assertIsNotNone(resolved, msg=f"No route for: {query}")
                route, reason = resolved  # type: ignore[misc]
                self.assertEqual(route, expected_route)
                self.assertEqual(reason, expected_reason)


class CoerceBoolTests(unittest.TestCase):
    def test_native_and_textual(self):
        self.assertIs(_coerce_bool(True), True)
        self.assertIs(_coerce_bool(False), False)
        self.assertIs(_coerce_bool("true"), True)
        self.assertIs(_coerce_bool("yes"), True)
        self.assertIs(_coerce_bool("non"), False)
        self.assertIs(_coerce_bool(1), True)

    def test_missing_or_garbage_is_none(self):
        self.assertIsNone(_coerce_bool(None))
        self.assertIsNone(_coerce_bool("maybe"))


class IntentRouterLLMTradeTests(unittest.TestCase):
    """Cas ambigus (aucun mot-clé) : le classifieur LLM décide is_trade."""

    def _run(self, query: str, agent):
        state = {"query": query, "normalized_query": query, "stats": {}}
        return asyncio.run(intent_router_node(agent, state))

    def test_ambiguous_query_reaches_llm(self):
        # "mets 500$ sur MSFT" n'est plus capté par mots-clés (haute précision).
        self.assertEqual(detect_tool_domains("mets 500$ sur MSFT"), frozenset())

    def test_llm_marks_informal_order_as_trade(self):
        agent = _agent_streaming('{"route":"complex","is_trade":true,"reason":"order"}')
        result = self._run("mets 500$ sur MSFT", agent)
        self.assertTrue(result["trade_requested"])
        self.assertEqual(result["intent_route"], "complex")

    def test_llm_marks_advice_as_not_trade(self):
        agent = _agent_streaming('{"route":"complex","is_trade":false,"reason":"advice"}')
        result = self._run("que faire de bien avec NVDA selon toi", agent)
        self.assertFalse(result["trade_requested"])

    def test_is_trade_true_forces_complex_route(self):
        agent = _agent_streaming('{"route":"simple","is_trade":true,"reason":"x"}')
        result = self._run("mets 500$ sur MSFT", agent)
        self.assertEqual(result["intent_route"], "complex")

    def test_missing_is_trade_falls_back_to_keywords(self):
        # is_trade absent + aucun mot-clé trade => trade_requested False.
        agent = _agent_streaming('{"route":"simple","reason":"x"}')
        result = self._run("explique-moi NVDA en deux phrases", agent)
        self.assertFalse(result["trade_requested"])
        self.assertEqual(result["intent_route"], "simple")

    def test_llm_failure_falls_back_to_simple(self):
        agent = _agent_streaming(raises=True)
        result = self._run("explique-moi NVDA en deux phrases", agent)
        self.assertEqual(result["intent_route"], "simple")
        self.assertFalse(result["trade_requested"])


if __name__ == "__main__":
    unittest.main()
