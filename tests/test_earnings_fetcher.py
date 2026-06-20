"""Tests du fetcher d'earnings calls et du contrat de nommage des transcripts.

Le pipeline existant (preprocess + indexation) ramasse les transcripts d'après
leur seul nom de fichier. Ces tests verrouillent ce contrat en réutilisant les
mêmes helpers que le pipeline, pour que le fetcher reste cohérent avec lui.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.fetchers import download_earnings_calls as dec
from src.preprocess.classify import (
    extract_ticker_from_filename,
    extract_year_from_filename,
    is_earnings_call_filename,
)
from src.rag.metadata.extract import extract_file_type_from_source


class FilenameContractTests(unittest.TestCase):
    """Le nom produit doit être décodable par le pipeline (ticker/year/type)."""

    def _name(self, ticker: str, year: int, quarter: int) -> str:
        return f"{ticker.lower()}-earnings_call_{year}q{quarter}.txt"

    def test_contract_holds_for_all_universe_tickers(self):
        cases = [("NVDA", 2025, 1), ("TSM", 2025, 4), ("MU", 2024, 3), ("AVGO", 2025, 2)]
        for ticker, year, quarter in cases:
            name = self._name(ticker, year, quarter)
            with self.subTest(ticker=ticker):
                self.assertEqual(extract_ticker_from_filename(name), ticker)
                self.assertEqual(extract_year_from_filename(name), str(year))
                self.assertEqual(extract_file_type_from_source(name), "EARNINGS_CALL")
                self.assertTrue(is_earnings_call_filename(name))


class ParsePayloadTests(unittest.TestCase):
    """Réponse Alpha Vantage : {symbol, quarter, transcript:[{speaker,title,content}]}."""

    def _turns(self, n: int = 6) -> list[dict]:
        return [
            {
                "speaker": f"Speaker {i}",
                "title": "CFO",
                "content": "Revenue grew strongly this quarter across all segments. " * 2,
                "sentiment": "0.6",
            }
            for i in range(n)
        ]

    def test_valid_transcript_joined(self):
        payload = {"symbol": "NVDA", "quarter": "2025Q1", "transcript": self._turns()}
        text = dec.parse_transcript_payload(payload)
        self.assertIsNotNone(text)
        self.assertIn("Speaker 0 (CFO):", text)
        self.assertGreaterEqual(len(text), 200)

    def test_short_transcript_rejected(self):
        payload = {"transcript": [{"speaker": "X", "content": "hi"}]}
        self.assertIsNone(dec.parse_transcript_payload(payload))

    def test_empty_or_missing_transcript_rejected(self):
        self.assertIsNone(dec.parse_transcript_payload({"transcript": []}))
        self.assertIsNone(dec.parse_transcript_payload({"symbol": "NVDA"}))
        self.assertIsNone(dec.parse_transcript_payload([]))

    def test_rate_limit_or_error_message_rejected(self):
        self.assertIsNone(
            dec.parse_transcript_payload({"Information": "rate limit reached"})
        )
        self.assertIsNone(
            dec.parse_transcript_payload({"Error Message": "invalid api key"})
        )


class SaveTranscriptTests(unittest.TestCase):
    def test_writes_well_named_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            with (
                patch.object(dec, "PROJECT_ROOT", root),
                patch.object(dec, "DATA_DIR", data),
            ):
                path = dec.save_transcript("TSM", 2025, 1, "x" * 500)
            self.assertEqual(path, "data/tsm-earnings_call_2025q1.txt")
            self.assertTrue((data / "tsm-earnings_call_2025q1.txt").exists())

    def test_does_not_overwrite_existing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            existing = data / "tsm-earnings_call_2025q1.txt"
            existing.write_text("original", encoding="utf-8")
            with (
                patch.object(dec, "PROJECT_ROOT", root),
                patch.object(dec, "DATA_DIR", data),
            ):
                dec.save_transcript("TSM", 2025, 1, "new content" * 50)
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")


class MainTests(unittest.TestCase):
    def test_no_api_key_skips_cleanly(self):
        with (
            patch.object(sys, "argv", ["prog"]),
            patch.object(dec, "_api_key", return_value=""),
        ):
            self.assertEqual(dec.main(), 0)

    def test_main_writes_transcripts_and_metadata(self):
        def fake_fetch(ticker, year, quarter, api_key):
            # un seul transcript renvoyé, pour NVDA Q1 2025
            if ticker == "NVDA" and quarter == 1:
                return "earnings call content " * 50
            return None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            meta = data / "earnings_calls_metadata.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["prog", "--min-year", "2025", "--max-year", "2025"],
                ),
                patch.object(dec, "_api_key", return_value="fake-key"),
                patch.object(dec, "PROJECT_ROOT", root),
                patch.object(dec, "DATA_DIR", data),
                patch.object(dec, "EARNINGS_CALLS_METADATA", meta),
                patch.object(dec, "fetch_transcript", side_effect=fake_fetch),
            ):
                rc = dec.main()
            self.assertEqual(rc, 0)
            self.assertTrue((data / "nvda-earnings_call_2025q1.txt").exists())
            written = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(len(written["NVDA"]), 1)
            self.assertEqual(written["NVDA"][0]["quarter"], 1)


if __name__ == "__main__":
    unittest.main()
