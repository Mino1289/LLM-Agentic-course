import json
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from api.services.artifact_mapper import (
    price_series_to_artifacts,
    state_to_artifacts,
    state_to_response,
)
from src.graph.tool_nodes import fetch_price_data


class PriceSeriesTests(unittest.TestCase):
    def test_fetch_price_data_returns_structured_series(self):
        agent = MagicMock()
        agent.price_max_points = 40
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        close = pd.Series(
            [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            index=dates,
        )
        frame = pd.DataFrame({"Close": close})

        with patch("src.graph.tool_nodes.yf.download", return_value=frame):
            payload = fetch_price_data(agent, ["MSFT"], "2026-01-01", "2026-01-10")

        self.assertIn("MSFT", payload["text"])
        self.assertEqual(len(payload["series"]), 1)
        series = payload["series"][0]
        self.assertEqual(series["ticker"], "MSFT")
        self.assertGreaterEqual(len(series["points"]), 1)
        self.assertIn("perf_pct", series["stats"])

    def test_state_to_artifacts_maps_price_charts(self):
        state = {
            "price_series": [
                {
                    "ticker": "MSFT",
                    "start_date": "2026-03-18",
                    "end_date": "2026-06-18",
                    "points": [{"date": "2026-03-18", "close": 390.5}],
                    "stats": {
                        "perf_pct": -3.08,
                        "vol_ann_pct": 32.57,
                        "max_drawdown_pct": -17.72,
                        "close_min": 356.0,
                        "close_max": 460.52,
                        "close_last": 378.91,
                        "high_date": "2026-06-01",
                        "low_date": "2026-05-10",
                    },
                }
            ],
            "stats": {},
            "tool_events": [],
        }
        artifacts = state_to_artifacts(state, locale="fr")
        self.assertEqual(len(artifacts.price_charts), 1)
        chart = artifacts.price_charts[0]
        self.assertEqual(chart.ticker, "MSFT")
        self.assertEqual(chart.points[0].close, 390.5)
        self.assertEqual(chart.stats.perf_pct, -3.08)

    def test_price_series_to_artifacts_empty(self):
        self.assertEqual(price_series_to_artifacts([]), [])

    def test_price_series_to_artifacts_accepts_capitalized_keys(self):
        charts = price_series_to_artifacts(
            [
                {
                    "ticker": "MSFT",
                    "start_date": "2026-03-18",
                    "end_date": "2026-06-18",
                    "points": [{"Date": "2026-03-18", "Close": 390.5}],
                    "stats": {"perf_pct": -3.08},
                }
            ]
        )
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0].points[0].close, 390.5)

    def test_chat_response_serializes_multi_point_chart(self):
        agent = MagicMock()
        agent.price_max_points = 60
        dates = pd.date_range("2026-03-18", periods=65, freq="B")
        close = pd.Series([380 + (i % 20) - 10 for i in range(65)], index=dates)
        frame = pd.DataFrame({"Close": close})

        with patch("src.graph.tool_nodes.yf.download", return_value=frame):
            payload = fetch_price_data(agent, ["MSFT"], "2026-03-18", "2026-06-18")

        state = {
            "price_series": payload["series"],
            "stats": {},
            "tool_events": [],
            "answer": "MSFT price summary",
        }
        response = state_to_response(state, "conv-1", "run-1", locale="fr")
        payload_json = json.loads(response.model_dump_json(by_alias=True))
        charts = payload_json["artifacts"]["priceCharts"]
        self.assertEqual(len(charts), 1)
        self.assertGreaterEqual(len(charts[0]["points"]), 2)
        self.assertIsNotNone(charts[0]["stats"]["perfPct"])


if __name__ == "__main__":
    unittest.main()
