"""Tests for the debug RAG universe and SEC pacing/filter logic."""

import os
import time
import unittest
from unittest.mock import patch


class TrackedTickersUniverseTests(unittest.TestCase):
    """Etape 2 debug mode — keep a compact tracked universe."""

    def test_tracked_tickers_has_debug_entries(self):
        from rag.config import TRACKED_TICKERS

        self.assertEqual(TRACKED_TICKERS, ("NVDA", "ASML", "AMD", "ARM", "MSFT"))

    def test_tracked_tickers_contains_all_requested_symbols(self):
        from rag.config import TRACKED_TICKERS

        expected = {"NVDA", "ASML", "AMD", "ARM", "MSFT"}
        self.assertEqual(set(TRACKED_TICKERS), expected)

    def test_price_download_scope_covers_tracked_tickers(self):
        from rag.download_share_prices import entreprises
        from rag.config import TRACKED_TICKERS

        mapping_values = set(entreprises.values())
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

    def test_sec_tickers_count_is_5(self):
        from rag.download_SEC_reports import SEC_TICKERS

        self.assertEqual(len(SEC_TICKERS), 5)

    def test_pa_ticker_count_is_0(self):
        from rag.download_SEC_reports import SKIPPED_TICKERS

        self.assertEqual(len(SKIPPED_TICKERS), 0)


class SecFormCoverageTests(unittest.TestCase):
    """Foreign private issuers file 20-F (annual) and 6-K (interim) instead of 10-K/10-Q."""

    def test_get_filings_accepts_us_forms(self):
        from rag.download_SEC_reports import get_filings

        response = _FakeResponse(
            {
                "filings": {
                    "recent": {
                        "form": ["10-K", "10-Q", "8-K", "8-K", "S-8"],
                        "filingDate": ["2024-02-01"] * 5,
                        "items": ["", "", "2.02,9.01", "1.01", ""],
                        "accessionNumber": ["0001-01"] * 5,
                        "primaryDocument": ["doc.htm"] * 5,
                    }
                }
            }
        )
        with patch("rag.download_SEC_reports.requests.get", return_value=response), \
             patch("rag.download_SEC_reports.rate_limit_sleep"):
            filings = get_filings("NVDA", "0001045810", 2024, 2026)
        self.assertGreater(len(filings), 0)
        forms = {f["form"] for f in filings}
        # NVDA files 10-K, 10-Q, 8-K only
        self.assertTrue(forms.issubset({"10-K", "10-Q", "8-K"}))

    def test_get_filings_includes_foreign_issuer_forms(self):
        from rag.download_SEC_reports import get_filings

        # ASML is a Dutch foreign private issuer (CIK 0000937966)
        response = _FakeResponse(
            {
                "filings": {
                    "recent": {
                        "form": ["20-F", "6-K", "S-8"],
                        "filingDate": ["2024-02-01"] * 3,
                        "items": ["", "", ""],
                        "accessionNumber": ["0002-02"] * 3,
                        "primaryDocument": ["doc.htm"] * 3,
                    }
                }
            }
        )
        with patch("rag.download_SEC_reports.requests.get", return_value=response), \
             patch("rag.download_SEC_reports.rate_limit_sleep"):
            filings = get_filings("ASML", "0000937966", 2024, 2026)
        self.assertGreater(len(filings), 0, "ASML should have foreign-issuer filings (20-F/6-K)")
        forms = {f["form"] for f in filings}
        # Must include at least one 20-F (annual) and 6-K (interim)
        self.assertIn("20-F", forms)
        self.assertIn("6-K", forms)
        # Must NOT include 10-K/10-Q (those are US issuers only)
        self.assertNotIn("10-K", forms)
        self.assertNotIn("10-Q", forms)


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
