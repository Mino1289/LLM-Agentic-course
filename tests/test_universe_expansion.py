"""Tests for the expanded RAG universe (20 tickers) and SEC pacing/filter logic."""

import os
import time
import unittest
from unittest.mock import patch


class TrackedTickersUniverseTests(unittest.TestCase):
    """PRD/etape2 — expand the RAG universe from 3 to 20 tickers."""

    def test_tracked_tickers_has_exactly_20_entries(self):
        from rag.config import TRACKED_TICKERS

        self.assertEqual(len(TRACKED_TICKERS), 20)

    def test_tracked_tickers_contains_all_requested_symbols(self):
        from rag.config import TRACKED_TICKERS

        expected = {
            "NVDA", "ASML", "TSM", "AMD", "AVGO", "ARM", "MSFT", "AAPL",
            "INTC", "QCOM", "MC.PA", "RMS.PA", "KER.PA", "AIR.PA", "TTE.PA",
            "BRK-B", "JPM", "CAT", "NKE", "XOM",
        }
        self.assertEqual(set(TRACKED_TICKERS), expected)

    def test_supported_companies_covers_all_20_tickers(self):
        from rag.download_share_prices import supported_companies
        from rag.config import TRACKED_TICKERS

        mapping_values = set(supported_companies.values())
        self.assertEqual(mapping_values, set(TRACKED_TICKERS))

    def test_supported_companies_slugs_are_filesystem_safe(self):
        """Slugs are used to name CSV files; no special chars allowed."""
        from rag.download_share_prices import supported_companies

        for slug in supported_companies:
            self.assertNotIn(".", slug, f"slug '{slug}' contains a dot")
            self.assertNotIn("/", slug, f"slug '{slug}' contains a slash")
            self.assertNotIn("\\", slug, f"slug '{slug}' contains a backslash")


class SecPacingTests(unittest.TestCase):
    """SEC inter-ticker sleep must be env-driven with a safety floor."""

    def _reload_module(self):
        """Reload download_SEC_reports to re-read env-derived constants."""
        import importlib
        import rag.download_SEC_reports as mod
        importlib.reload(mod)
        return mod

    def test_default_sleep_is_150ms(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEC_INTER_TICKER_SLEEP", None)
            mod = self._reload_module()
            self.assertEqual(mod._INTER_TICKER_SLEEP, 0.15)

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"SEC_INTER_TICKER_SLEEP": "0.3"}):
            mod = self._reload_module()
            self.assertEqual(mod._INTER_TICKER_SLEEP, 0.3)

    def test_env_var_below_floor_is_clamped_to_safety_minimum(self):
        with patch.dict(os.environ, {"SEC_INTER_TICKER_SLEEP": "0.01"}):
            mod = self._reload_module()
            self.assertEqual(mod._INTER_TICKER_SLEEP, 0.05)

    def test_invalid_env_var_falls_back_to_default(self):
        with patch.dict(os.environ, {"SEC_INTER_TICKER_SLEEP": "not-a-number"}):
            mod = self._reload_module()
            self.assertEqual(mod._INTER_TICKER_SLEEP, 0.15)

    def test_rate_limit_sleep_uses_configured_value(self):
        with patch.dict(os.environ, {"SEC_INTER_TICKER_SLEEP": "0.2"}):
            mod = self._reload_module()
            with patch("rag.download_SEC_reports.time.sleep") as mocked_sleep:
                mod.rate_limit_sleep()
                mocked_sleep.assert_called_once_with(0.2)


class SecTickerFilterTests(unittest.TestCase):
    """.PA tickers (Euronext) must be excluded from SEC downloads."""

    def test_pa_tickers_excluded_from_sec_tickers(self):
        from rag.download_SEC_reports import SEC_TICKERS, SKIPPED_TICKERS

        pa_tickers = {t for t in SEC_TICKERS if t.endswith(".PA")}
        self.assertEqual(pa_tickers, set(), "no .PA ticker should be in SEC_TICKERS")

    def test_pa_tickers_listed_in_skipped(self):
        from rag.download_SEC_reports import SEC_TICKERS, SKIPPED_TICKERS, TRACKED_TICKERS

        all_pa = {t for t in TRACKED_TICKERS if t.endswith(".PA")}
        self.assertEqual(set(SKIPPED_TICKERS), all_pa)
        # every skipped ticker must NOT be in SEC_TICKERS
        self.assertEqual(set(SKIPPED_TICKERS) & set(SEC_TICKERS), set())

    def test_sec_tickers_count_is_15(self):
        from rag.download_SEC_reports import SEC_TICKERS

        self.assertEqual(len(SEC_TICKERS), 15)

    def test_pa_ticker_count_is_5(self):
        from rag.download_SEC_reports import SKIPPED_TICKERS

        self.assertEqual(len(SKIPPED_TICKERS), 5)


if __name__ == "__main__":
    unittest.main()
