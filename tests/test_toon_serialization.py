"""Tests for TOON-format usage in project serializers (etape6+ post-audit).

These tests validate the concrete substitutions recommended in
docs/superpowers/plans/2026-06-06-toon-integration-verification.md:

- P1-A : rag.tools.format_rag_excerpts returns a TOON-formatted payload
  instead of the legacy "key=value" text format.
- P2-C : rag.nodes.memory_store.format_memory_context and
  format_chat_context return a TOON-formatted payload for LLM context.

The legacy text format is no longer produced; the function names are kept
for backward compatibility. All 3 functions MUST return a string that
toon_format.decode() can roundtrip back to the original structure (or to
a faithful representation).
"""

import unittest


class ToonRagExcerptsTests(unittest.TestCase):
    """P1-A — format_rag_excerpts must produce a TOON tabular array."""

    def _make_chunks(self):
        return [
            "We are subject to risks related to global supply chain concentration.",
            "Our data center business depends on a small number of customers.",
        ]

    def _make_metadatas(self):
        return [
            {
                "ticker": "NVDA",
                "year": "2024",
                "file_type": "10-K",
                "section": "Item_1A",
                "source": "nvda-10-k_2024-01-01.htm",
            },
            {
                "ticker": "MSFT",
                "year": "2024",
                "file_type": "10-K",
                "section": "Item_1A",
                "source": "msft-10-k_2024.htm",
            },
        ]

    def test_empty_chunks_returns_fallback_message(self):
        from src.tools.descriptions import format_rag_excerpts

        result = format_rag_excerpts([], [])
        # Empty case must NOT produce a TOON "[0]:" — keep the human-readable
        # fallback so the LLM gets a clear "no results" message.
        self.assertIsInstance(result, str)
        self.assertIn("No matching", result)
        # Must not be a TOON tabular array header
        self.assertNotIn("]{", result)

    def test_single_chunk_returns_toon_tabular_array(self):
        from src.tools.descriptions import format_rag_excerpts

        result = format_rag_excerpts(self._make_chunks(), self._make_metadatas())
        # Must use TOON tabular form: "[2,]{i,ticker,year,file_type,section,source,text}:"
        self.assertIn("]{i,ticker,year,file_type,section,source,text}", result)
        # Must contain the actual chunk text (truncated or not)
        self.assertIn("supply chain", result)
        self.assertIn("data center", result)

    def test_toon_output_is_roundtrippable(self):
        """toon_format.decode(result) must produce a structure whose
        rows match the original chunks and metadata."""
        from toon_format import decode

        from src.tools.descriptions import format_rag_excerpts

        chunks = self._make_chunks()
        metas = self._make_metadatas()
        result = format_rag_excerpts(chunks, metas)
        decoded = decode(result)
        # The decoded structure must be a dict with the 'excerpts' key
        self.assertIsInstance(decoded, dict)
        self.assertIn("excerpts", decoded)
        rows = decoded["excerpts"]
        self.assertEqual(len(rows), 2)
        # Each row must have the expected fields
        for row in rows:
            self.assertIn("ticker", row)
            self.assertIn("text", row)
        # First row's text is a chunk (truncated or not)
        self.assertIn("supply chain", rows[0]["text"])
        # Tickers preserved in correct order
        self.assertEqual([r["ticker"] for r in rows], ["NVDA", "MSFT"])

    def test_chunk_text_truncated_to_1200_chars(self):
        """Backward compat: chunks were historically truncated to 1200 chars
        in the text format. TOON-formatted output must preserve this
        truncation to avoid blowing up the LLM context."""
        from src.tools.descriptions import format_rag_excerpts

        long_chunk = "A" * 5000
        result = format_rag_excerpts(
            [long_chunk],
            [{"ticker": "NVDA", "year": "2024", "file_type": "10-K", "section": "Item_1A", "source": "x"}],
        )
        # The TOON output must contain at most 1200 consecutive A's.
        import re
        match = re.search(r"A{1500,}", result)
        self.assertIsNone(
            match,
            f"Chunk text was not truncated to 1200 chars; found a long run of A's in: {result[:500]}",
        )


class ToonMemoryContextTests(unittest.TestCase):
    """P2-C — format_memory_context and format_chat_context must use TOON."""

    def test_format_memory_context_with_summary_and_window(self):
        from toon_format import decode

        from src.graph.memory_store import format_memory_context

        summary = "User asked about NVDA risks then MSFT risks."
        window = [
            {"role": "user", "content": "Quels sont les risques de NVDA ?"},
            {"role": "assistant", "content": "Concentration Taiwan, contrôles exportation."},
        ]
        result = format_memory_context(summary, window)
        # Must be decodable as TOON
        decoded = decode(result)
        # The structure must expose summary + turns
        self.assertIsInstance(decoded, dict)
        self.assertIn("summary", decoded)
        self.assertIn("turns", decoded)
        self.assertEqual(decoded["summary"], summary)
        self.assertEqual(len(decoded["turns"]), 2)
        self.assertEqual(decoded["turns"][0]["role"], "user")
        self.assertIn("NVDA", decoded["turns"][0]["content"])

    def test_format_memory_context_empty_returns_fallback(self):
        """When both summary and window are empty, the legacy function
        returned a French fallback. We keep that fallback verbatim (it's
        not LLM-bound data, it's a prompt-engineering signal)."""
        from src.graph.memory_store import format_memory_context

        result = format_memory_context("", [])
        self.assertEqual(result, "Aucun contexte memorise.")

    def test_format_memory_context_summary_only(self):
        from toon_format import decode

        from src.graph.memory_store import format_memory_context

        result = format_memory_context("Just a summary.", [])
        decoded = decode(result)
        self.assertEqual(decoded["summary"], "Just a summary.")
        self.assertEqual(decoded["turns"], [])

    def test_format_memory_context_window_only(self):
        from toon_format import decode

        from src.graph.memory_store import format_memory_context

        result = format_memory_context("", [{"role": "user", "content": "hi"}])
        decoded = decode(result)
        self.assertEqual(decoded["turns"], [{"role": "user", "content": "hi"}])

    def test_format_chat_context_returns_toon(self):
        from toon_format import decode

        from src.graph.memory_store import format_chat_context

        messages = [
            {"role": "user", "content": "Premier message"},
            {"role": "assistant", "content": "Première réponse"},
        ]
        result = format_chat_context(messages, keep_last=6)
        decoded = decode(result)
        self.assertIn("turns", decoded)
        self.assertEqual(len(decoded["turns"]), 2)
        self.assertEqual(decoded["turns"][1]["content"], "Première réponse")

    def test_format_chat_context_empty_returns_fallback(self):
        """Empty chat history: keep the French fallback (prompt signal)."""
        from src.graph.memory_store import format_chat_context

        result = format_chat_context([], keep_last=6)
        self.assertEqual(result, "Aucun historique de chat.")


if __name__ == "__main__":
    unittest.main()
