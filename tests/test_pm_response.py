"""Tests for PM answer extraction + RESPONSE-section streaming."""

from __future__ import annotations

import unittest

from src.orchestration.pm_decision import _RESPONSE_RE, extract_user_response


class ExtractUserResponseTests(unittest.TestCase):
    def test_extracts_after_marker(self):
        text = (
            "SYNTHESIS:\n- Fundamental: bla\n- Quant: bla\n"
            "RESPONSE: Voici la comparaison NVDA vs AMD."
        )
        self.assertEqual(
            extract_user_response(text), "Voici la comparaison NVDA vs AMD."
        )

    def test_extracts_after_parenthetical_marker(self):
        text = "DECISION:\n- Ticker: NVDA\nRESPONSE (French, human-readable): Achat conseillé."
        self.assertEqual(extract_user_response(text), "Achat conseillé.")

    def test_accented_french_marker(self):
        self.assertEqual(extract_user_response("RÉPONSE : bonjour"), "bonjour")

    def test_fallback_when_no_marker(self):
        text = "Pas de marqueur, juste du texte."
        self.assertEqual(extract_user_response(text), text)

    def test_uses_last_marker_occurrence(self):
        text = "RESPONSE: premier\nRESPONSE: dernier"
        self.assertEqual(extract_user_response(text), "dernier")


class StreamingMarkerTests(unittest.TestCase):
    """Reproduit la logique de streaming incrémental de pm_synthesis_node."""

    def _stream(self, deltas: list[str]) -> str:
        buf = ""
        emitted = 0
        started = False
        out: list[str] = []
        for delta in deltas:
            buf += delta
            if not started:
                marker = _RESPONSE_RE.search(buf)
                if marker:
                    started = True
                    emitted = marker.end()
            if started and len(buf) > emitted:
                out.append(buf[emitted:])
                emitted = len(buf)
        return "".join(out)

    def test_marker_split_across_tokens(self):
        # "RESPONSE" coupé entre deux deltas ("RESP" | "ONSE")
        streamed = self._stream(["SYNTH", "ESIS: x\nRESP", "ONSE: Bon", "jour."])
        self.assertEqual(streamed, "Bonjour.")

    def test_nothing_streamed_before_marker(self):
        streamed = self._stream(["PLAN:\n- ", "task a\n", "DECISION:\n"])
        self.assertEqual(streamed, "")


if __name__ == "__main__":
    unittest.main()
