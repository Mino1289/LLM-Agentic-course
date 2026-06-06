"""Tests for the query-embedding cache.

Context
-------
GitHub Models enforces a 150 requests/day limit on ``text-embedding-3-small``.
The full indexing batch consumes ~1,500 chunks and exhausts the quota in a
single run. Subsequent *queries* then fail with ``429 RateLimitReached``
even though the index is already populated — because every user query
embeds a fresh text via the same OpenAI-compatible client.

This test suite verifies a persistent JSON cache that:
- stores query vectors keyed by SHA-256 of the query text,
- survives process restarts (file at ``data/embedding_query_cache.json``),
- prunes entries older than ``ttl_seconds``,
- evicts the oldest entries when ``max_entries`` is exceeded,
- exposes hit/miss counters via ``stats()`` for observability,
- is wired into ``LLMProvider.embed()`` so the second call to a repeated
  query never hits the network.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


class EmbeddingCacheUnitTests(unittest.TestCase):
    """Unit tests for ``EmbeddingCache`` itself (no LLM provider)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="embedding_cache_test_")
        self.cache_file = Path(self.tmpdir) / "cache.json"

    def _make(self, ttl: int = 86400, max_entries: int = 100):
        from rag.embedding_cache import EmbeddingCache

        return EmbeddingCache(self.cache_file, ttl_seconds=ttl, max_entries=max_entries)

    def test_get_returns_none_for_missing_text(self):
        cache = self._make()
        self.assertIsNone(cache.get("Azure FY2025 vs FY2024"))

    def test_put_then_get_roundtrip(self):
        cache = self._make()
        vec = [0.1, 0.2, 0.3, 0.4]
        cache.put("hello", vec)
        self.assertEqual(cache.get("hello"), vec)

    def test_different_texts_yield_different_keys(self):
        """The key is SHA-256(text) — collisions for distinct inputs are infeasible."""
        cache = self._make()
        cache.put("Azure FY2025 vs FY2024", [0.1])
        cache.put("Azure FY2024 vs FY2023", [0.2])
        self.assertEqual(cache.get("Azure FY2025 vs FY2024"), [0.1])
        self.assertEqual(cache.get("Azure FY2024 vs FY2023"), [0.2])

    def test_key_is_sha256_of_utf8(self):
        from rag.embedding_cache import EmbeddingCache

        expected = hashlib.sha256("café".encode("utf-8")).hexdigest()
        self.assertEqual(EmbeddingCache._key("café"), expected)

    def test_persistence_across_instances(self):
        """A second EmbeddingCache reading the same file must see prior writes."""
        cache_a = self._make()
        cache_a.put("persistent", [0.5, 0.6])
        cache_b = self._make()
        self.assertEqual(cache_b.get("persistent"), [0.5, 0.6])

    def test_ttl_expiration(self):
        """Entries older than ttl_seconds are pruned on load AND on get()."""
        cache_a = self._make(ttl=10)
        cache_a.put("stale", [0.1])
        # Backdate the entry beyond the TTL.
        raw = json.loads(self.cache_file.read_text())
        for entry in raw.values():
            entry["ts"] = time.time() - 100
        self.cache_file.write_text(json.dumps(raw))
        cache_b = self._make(ttl=10)
        self.assertIsNone(cache_b.get("stale"))
        # A second instance must also see no entry.
        cache_c = self._make(ttl=10)
        self.assertIsNone(cache_c.get("stale"))

    def test_max_entries_evicts_oldest(self):
        cache = self._make(max_entries=2)
        cache.put("a", [1.0])
        time.sleep(0.01)
        cache.put("b", [2.0])
        time.sleep(0.01)
        cache.put("c", [3.0])
        # "a" was the oldest; it must be gone, "b" and "c" remain.
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), [2.0])
        self.assertEqual(cache.get("c"), [3.0])

    def test_stats_reports_size_hits_misses(self):
        cache = self._make()
        cache.put("x", [0.1])
        cache.get("x")  # hit
        cache.get("x")  # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertTrue(str(self.cache_file) in str(stats["file"]))

    def test_corrupt_cache_starts_empty(self):
        self.cache_file.write_text("{not valid json")
        cache = self._make()
        self.assertEqual(cache.stats()["size"], 0)
        # Should still be usable.
        cache.put("fresh", [0.1])
        self.assertEqual(cache.get("fresh"), [0.1])

    def test_clear_removes_entries_and_file(self):
        cache = self._make()
        cache.put("x", [0.1])
        self.assertTrue(self.cache_file.exists())
        cache.clear()
        self.assertEqual(cache.stats()["size"], 0)
        self.assertFalse(self.cache_file.exists())

    def test_get_many_returns_parallel_results(self):
        cache = self._make()
        cache.put("foo", [1.0, 2.0])
        cache.put("bar", [3.0, 4.0])
        results = cache.get_many(["foo", "bar", "missing"])
        self.assertEqual(results, [[1.0, 2.0], [3.0, 4.0], None])

    def test_put_many_writes_atomic_file(self):
        """The cache file is rewritten via a temp + rename, never partial."""
        cache = self._make()
        cache.put_many([("x", [0.1]), ("y", [0.2])])
        # File exists, parses, has both entries.
        raw = json.loads(self.cache_file.read_text())
        self.assertEqual(len(raw), 2)
        # No leftover .tmp file.
        self.assertFalse(self.cache_file.with_suffix(self.cache_file.suffix + ".tmp").exists())


class EmbeddingCacheProviderIntegrationTests(unittest.TestCase):
    """Wire the cache into ``LLMProvider.embed()`` and verify call savings."""

    def _make_provider(self, cache_file: Path) -> Any:
        from rag.llm_provider import LLMConfig, LLMProvider

        cfg = LLMConfig(
            provider="github_models",
            chat_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
            api_key="test-key",
        )
        provider = LLMProvider.__new__(LLMProvider)
        provider.config = cfg
        # No real client — we'll patch embed in tests below.
        provider._embedding_cache = self._make_cache(cache_file)
        return provider

    def _make_cache(self, cache_file: Path):
        from rag.embedding_cache import EmbeddingCache

        return EmbeddingCache(cache_file)

    def test_embed_uses_cache_on_second_call(self):
        """The second call to embed() for the same query must NOT hit the API."""
        tmpdir = tempfile.mkdtemp()
        cache_file = Path(tmpdir) / "cache.json"
        provider = self._make_provider(cache_file)
        provider.embedding_client = MagicMock()
        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        provider.embedding_client.embeddings.create.return_value = fake_response

        # First call: API hit.
        r1 = provider.embed(["Azure FY2025 vs FY2024"])
        self.assertEqual(r1, [[0.1, 0.2, 0.3]])
        self.assertEqual(provider.embedding_client.embeddings.create.call_count, 1)

        # Second call (same query): cached, NO API call.
        r2 = provider.embed(["Azure FY2025 vs FY2024"])
        self.assertEqual(r2, [[0.1, 0.2, 0.3]])
        self.assertEqual(provider.embedding_client.embeddings.create.call_count, 1)

    def test_embed_dedups_duplicate_inputs(self):
        """A batch with the same query 3 times → 1 API call (not 3)."""
        tmpdir = tempfile.mkdtemp()
        cache_file = Path(tmpdir) / "cache.json"
        provider = self._make_provider(cache_file)
        provider.embedding_client = MagicMock()
        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[0.5])]
        provider.embedding_client.embeddings.create.return_value = fake_response

        result = provider.embed(["NVDA risks", "NVDA risks", "NVDA risks"])
        self.assertEqual(result, [[0.5], [0.5], [0.5]])
        # Only 1 embedding for the 3 duplicates.
        self.assertEqual(provider.embedding_client.embeddings.create.call_count, 1)

    def test_embed_falls_back_to_api_on_miss(self):
        """First call to a new query hits the API; subsequent calls don't."""
        tmpdir = tempfile.mkdtemp()
        cache_file = Path(tmpdir) / "cache.json"
        provider = self._make_provider(cache_file)
        provider.embedding_client = MagicMock()
        provider.embedding_client.embeddings.create.side_effect = [
            MagicMock(data=[MagicMock(embedding=[0.1])]),
            MagicMock(data=[MagicMock(embedding=[0.2])]),
        ]

        provider.embed(["query A"])
        provider.embed(["query A"])  # cached
        provider.embed(["query B"])  # miss → 1 new API call
        self.assertEqual(provider.embedding_client.embeddings.create.call_count, 2)

    def test_embed_skips_empty_inputs(self):
        tmpdir = tempfile.mkdtemp()
        cache_file = Path(tmpdir) / "cache.json"
        provider = self._make_provider(cache_file)
        provider.embedding_client = MagicMock()
        self.assertEqual(provider.embed([]), [])
        self.assertEqual(provider.embed(["", "   ", "\n"]), [])
        provider.embedding_client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
