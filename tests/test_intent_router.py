from __future__ import annotations

import unittest

from src.orchestration.tool_domains import (
    detect_tool_domains,
    resolve_route_from_domains,
)


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


if __name__ == "__main__":
    unittest.main()
