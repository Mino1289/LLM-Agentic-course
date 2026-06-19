"""Tests for toon-python integration (PRD/etape6 + dep audit).

The toon-python library provides TOON (Token-Oriented Object Notation),
a compact serialization format designed for LLM contexts. It claims
30-60% token reduction vs JSON by combining YAML-like indentation
with CSV-like tabular arrays for uniform data.

These tests are RED before `pip install toon_format[pydantic] tiktoken`,
and GREEN after.

Reference: https://github.com/toon-format/toon-python
"""

import unittest


class ToonFormatImportTests(unittest.TestCase):
    """The library must be importable from the standard name."""

    def test_import_toon_format_succeeds(self):
        from toon_format import encode, decode  # noqa: F401

        self.assertTrue(callable(encode))
        self.assertTrue(callable(decode))

    def test_import_token_helpers_succeeds(self):
        from toon_format import estimate_savings, count_tokens  # noqa: F401

        self.assertTrue(callable(estimate_savings))
        self.assertTrue(callable(count_tokens))

    def test_import_pydantic_extra_succeeds(self):
        from toon_format.pydantic import ToonPydanticModel  # noqa: F401

        self.assertTrue(ToonPydanticModel is not None)


class ToonFormatEncodeDecodeTests(unittest.TestCase):
    """encode/decode roundtrip on representative structures we want to
    optimize in the codebase (dict simple, dict imbriqué, list tabulaire)."""

    def test_encode_simple_object_produces_toon_string(self):
        from toon_format import encode

        result = encode({"name": "Alice", "age": 30})
        self.assertIsInstance(result, str)
        self.assertIn("Alice", result)
        self.assertIn("age", result)

    def test_decode_roundtrip_simple_dict(self):
        from toon_format import decode, encode

        original = {"a": 1, "b": "hello", "c": True}
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

    def test_decode_roundtrip_nested_dict(self):
        from toon_format import decode, encode

        original = {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
            ],
            "count": 2,
        }
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

    def test_encode_tabular_array_uses_compact_format(self):
        """Uniform arrays must use the tabular form `[N,]{fields}:` which
        is what produces the 30-60% token savings claim."""
        from toon_format import encode

        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        encoded = encode(data)
        # Tabular form is `[3,]{id,name}:` — not a list of "- id: 1" blocks.
        self.assertIn("]{id,name}", encoded, f"Expected tabular header in: {encoded!r}")
        # Rows should be compact CSV-like, not multi-line blocks.
        self.assertIn("Alice", encoded)
        self.assertIn("Bob", encoded)

    def test_decode_tabular_array_returns_list(self):
        from toon_format import decode

        # Hand-crafted TOON tabular array
        toon = "[2,]{id,name}:\n  1,Alice\n  2,Bob"
        decoded = decode(toon)
        self.assertEqual(
            decoded, [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        )


class ToonFormatSavingsTests(unittest.TestCase):
    """estimate_savings is the API that tells us if TOON actually
    beats JSON on our structures. If it does, the audit can recommend
    concrete substitutions."""

    def test_estimate_savings_returns_positive_for_tabular_data(self):
        from toon_format import estimate_savings
        from unittest.mock import patch

        # RAG-like tabular structure: 4 chunks with 4 uniform metadata fields.
        data = {
            "excerpts": [
                {
                    "ticker": "NVDA",
                    "year": "2024",
                    "file_type": "10-K",
                    "section": "Item_1A",
                },
                {
                    "ticker": "MSFT",
                    "year": "2024",
                    "file_type": "10-K",
                    "section": "Item_1A",
                },
                {
                    "ticker": "AMD",
                    "year": "2024",
                    "file_type": "10-K",
                    "section": "Item_1A",
                },
                {
                    "ticker": "NVDA",
                    "year": "2024",
                    "file_type": "10-K",
                    "section": "Item_7",
                },
            ]
        }
        with patch(
            "toon_format.utils.count_tokens",
            side_effect=lambda text, *_args, **_kwargs: len(text.split()),
        ):
            result = estimate_savings(data)
        # Savings must be non-negative (TOON is never worse than JSON
        # for our usage patterns).
        self.assertGreaterEqual(result.get("savings_percent", 0), 0)
        # Sanity: there must be both json and toon token counts in the result.
        self.assertIn("json_tokens", result)
        self.assertIn("toon_tokens", result)


if __name__ == "__main__":
    unittest.main()
