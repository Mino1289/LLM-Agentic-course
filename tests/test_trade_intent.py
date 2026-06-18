import unittest

from src.orchestration.pm_decision import parse_dollar_amount
from src.orchestration.tool_domains import (
    detect_tool_domains,
    resolve_route_from_domains,
)
from src.orchestration.trade_intent import (
    has_trade_proposal,
    is_trade_requested,
    route_after_compliance,
    route_after_pm_synthesis,
)


class TradeIntentTests(unittest.TestCase):
    def test_compare_query_is_multi_tool_not_trade(self):
        query = "Compare risques SEC et perf 6 mois NVDA/AMD"
        domains = detect_tool_domains(query)
        route = resolve_route_from_domains(domains)

        self.assertIn("fundamental", domains)
        self.assertIn("quant", domains)
        self.assertNotIn("trade", domains)
        self.assertEqual(route, ("complex", "multi_tool"))

    def test_compare_query_does_not_request_trade(self):
        state = {
            "query": "Compare risques SEC et perf 6 mois NVDA/AMD",
            "trade_requested": False,
            "stats": {
                "tool_domains": ["fundamental", "quant"],
                "intent_reason": "multi_tool",
            },
        }
        self.assertFalse(is_trade_requested(state))

    def test_analysis_synthesis_skips_compliance_and_human_review(self):
        state = {
            "trade_requested": False,
            "pm_decision": {
                "action": "none",
                "response": "Comparaison NVDA vs AMD sur les risques SEC et la perf 6 mois.",
            },
            "answer": "Comparaison NVDA vs AMD sur les risques SEC et la perf 6 mois.",
        }
        self.assertFalse(has_trade_proposal(state))
        self.assertEqual(route_after_pm_synthesis(state), "__end__")

    def test_trade_query_requests_trade(self):
        state = {
            "query": "Achète 10 actions NVDA",
            "trade_requested": True,
            "stats": {"tool_domains": ["trade"], "intent_reason": "action_keyword"},
        }
        self.assertTrue(is_trade_requested(state))

    def test_valid_trade_proposal_routes_to_compliance(self):
        state = {
            "trade_requested": True,
            "pm_decision": {
                "ticker": "NVDA",
                "side": "buy",
                "qty": "10",
                "order_type": "market",
                "response": "Achat de 10 NVDA.",
            },
        }
        self.assertTrue(has_trade_proposal(state))
        self.assertEqual(route_after_pm_synthesis(state), "compliance_validator")

    def test_compliance_pass_with_trade_goes_to_human_review(self):
        state = {
            "pm_decision": {"ticker": "NVDA", "side": "buy", "qty": "10"},
            "compliance_verdict": "PASS",
        }
        self.assertEqual(route_after_compliance(state), "human_review")

    def test_compliance_pass_without_trade_ends(self):
        state = {
            "pm_decision": {"action": "none", "response": "Analyse seulement."},
            "compliance_verdict": "PASS",
        }
        self.assertEqual(route_after_compliance(state), "__end__")

    def test_informal_trade_phrasing_routes_complex(self):
        query = "mets 500$ sur MSFT"
        domains = detect_tool_domains(query)
        route = resolve_route_from_domains(domains)
        self.assertIn("trade", domains)
        self.assertEqual(route, ("complex", "action_keyword"))

    def test_parse_dollar_amount_from_query(self):
        self.assertEqual(parse_dollar_amount("investis 500$ dans NVDA"), 500.0)
        self.assertEqual(parse_dollar_amount("buy $1,250 of MSFT"), 1250.0)

    def test_place_trade_requires_human_approval(self):
        from src.tools.schemas import PlaceTradeArgs
        from src.tools.trading import run_place_trade

        result = run_place_trade(PlaceTradeArgs(ticker="MSFT", side="buy", qty=1))
        self.assertEqual(result.get("error"), "human_approval_required")

    def test_place_trade_allowed_when_human_approved(self):
        from unittest.mock import patch

        from src.tools.schemas import PlaceTradeArgs
        from src.tools.trading import run_place_trade

        with patch("src.alpaca.client.get_alpaca_client", return_value=None):
            result = run_place_trade(
                PlaceTradeArgs(ticker="MSFT", side="buy", qty=1),
                state={"human_approved": True},
            )
        self.assertEqual(result.get("error"), "alpaca_not_configured")


if __name__ == "__main__":
    unittest.main()
