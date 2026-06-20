"""Tests pour l'extraction des raisons de conformité.

Régression : COMPLIANCE_PROMPT impose "PASS"/"FAIL" comme premier mot. Quand le
verdict était seul sur sa ligne, _extract_reasons le captait (mot-clé "fail") et
affichait "Raison principale : FAIL" au lieu de l'explication réelle.
"""

from __future__ import annotations

import unittest

from src.orchestration.compliance_node import (
    _extract_reasons,
    _is_bare_verdict,
    _strip_verdict_prefix,
)


class BareVerdictTests(unittest.TestCase):
    def test_bare_verdict_detection(self):
        for line in ("FAIL", "fail", "**FAIL**", "FAIL.", "PASS"):
            self.assertTrue(_is_bare_verdict(line), line)

    def test_real_reason_is_not_bare_verdict(self):
        self.assertFalse(_is_bare_verdict("FAIL. Insufficient buying power."))
        self.assertFalse(_is_bare_verdict("Reduce quantity to 5 shares."))

    def test_strip_inline_verdict_prefix(self):
        self.assertEqual(
            _strip_verdict_prefix("FAIL. Insufficient buying power by $200."),
            "Insufficient buying power by $200.",
        )
        self.assertEqual(
            _strip_verdict_prefix("PASS: all checks satisfied"),
            "all checks satisfied",
        )


class ExtractReasonsTests(unittest.TestCase):
    def test_verdict_on_own_line_does_not_become_reason(self):
        result = "FAIL\n\nThe order exceeds the 25% concentration limit. Reduce quantity."
        reasons = _extract_reasons(result)
        self.assertNotIn("FAIL", reasons)
        self.assertTrue(reasons[0].lower().startswith("the order exceeds"))

    def test_inline_verdict_reason_is_cleaned(self):
        result = "FAIL. Insufficient buying power by $500."
        reasons = _extract_reasons(result)
        self.assertEqual(reasons[0], "Insufficient buying power by $500.")

    def test_no_detail_yields_placeholder_not_verdict(self):
        reasons = _extract_reasons("FAIL")
        self.assertEqual(len(reasons), 1)
        self.assertNotEqual(reasons[0].strip().lower(), "fail")

    def test_falls_back_to_useful_lines_without_keywords(self):
        result = "FAIL\n\nThe proposed allocation is too aggressive for this account."
        reasons = _extract_reasons(result)
        self.assertNotIn("FAIL", reasons)
        self.assertIn("aggressive", reasons[0].lower())


if __name__ == "__main__":
    unittest.main()
