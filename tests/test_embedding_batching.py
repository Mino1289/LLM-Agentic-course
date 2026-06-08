"""Anti-regression tests for the embedding batched-pipeline contract.

PRD etape 3 — ÉTAPE 3 guarantee: 1 API call = up to ``embedding_batch_size``
chunks (default 32). Indexing 1,500 chunks at batch_size=32 must consume
~47 API calls, NOT 1,500 individual calls.

This test suite pins the contract so any future refactor that accidentally
reverts to per-chunk embedding (e.g. dropping ``iter_batches`` and calling
``provider.embed([chunk])`` in a loop) is caught immediately.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from rag.hybrid_rag import HybridRAG, iter_batches


def _build_fake_rag(num_chunks: int, embedding_batch_size: int = 32):
    """Build a HybridRAG pre-loaded with ``num_chunks`` fake documents.

    The provider, collection, and ``_build_corpus`` are all mocked so no
    network, no filesystem, and no real ChromaDB is touched.

    ``rag.provider.embed`` is wired with ``side_effect=fake_embed`` so
    MagicMock's built-in ``call_args_list`` / ``call_count`` accurately
    reflect what the indexing loop sent. We do NOT manually append to
    ``call_args_list`` (that breaks the auto-managed counter).
    """
    rag = object.__new__(HybridRAG)
    rag.chunk_strategy = "semantic"
    rag.documents = [f"chunk text #{i:04d}" for i in range(num_chunks)]
    rag.doc_metadata = [
        {"ticker": "NVDA", "year": "2024", "source": f"nvda-{i:04d}.htm"}
        for i in range(num_chunks)
    ]
    rag.chunk_ids = [f"nvda_{i:04d}" for i in range(num_chunks)]

    def fake_embed(texts):
        if not isinstance(texts, list):
            raise AssertionError(
                f"provider.embed() must receive a list of texts, got {type(texts).__name__}"
            )
        if texts and not isinstance(texts[0], str):
            raise AssertionError(
                f"provider.embed() must receive a list[str], got list[{type(texts[0]).__name__}]"
            )
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    rag.provider = MagicMock()
    rag.provider.embed.side_effect = fake_embed

    # Collection mock: get() returns empty (no existing chunks), upsert() is a no-op.
    rag.collection = MagicMock()
    rag.collection.name = "fake_collection"
    rag.collection.get.return_value = {"ids": []}
    rag.collection.upsert.return_value = None
    rag.collection.count.return_value = num_chunks

    # _build_corpus returns what we set on rag.documents / metadata / ids.
    rag._build_corpus = lambda max_files=None: (
        rag.documents,
        rag.doc_metadata,
        rag.chunk_ids,
        len(rag.documents),
    )
    return rag


class IterBatchesContractTests(unittest.TestCase):
    """``iter_batches`` must chunk N items into ceil(N / batch_size) groups."""

    def test_iter_batches_yields_exact_groups_for_100_items(self):
        items = list(range(100))
        batches = list(iter_batches(items, 32))
        self.assertEqual(len(batches), math.ceil(100 / 32))  # 4 batches
        self.assertEqual([len(b) for b in batches], [32, 32, 32, 4])

    def test_iter_batches_handles_partial_last_batch(self):
        items = list(range(33))
        batches = list(iter_batches(items, 32))
        self.assertEqual([len(b) for b in batches], [32, 1])

    def test_iter_batches_handles_exact_multiple(self):
        items = list(range(64))
        batches = list(iter_batches(items, 32))
        self.assertEqual([len(b) for b in batches], [32, 32])

    def test_iter_batches_clamps_batch_size_to_minimum_1(self):
        items = list(range(5))
        batches = list(iter_batches(items, 0))  # 0 clamped to 1
        self.assertEqual([len(b) for b in batches], [1, 1, 1, 1, 1])


class IndexingBatchContractTests(unittest.TestCase):
    """The indexing pipeline must call ``provider.embed()`` with a list[str]."""

    def _index(self, num_chunks: int, embedding_batch_size: int):
        rag = _build_fake_rag(num_chunks, embedding_batch_size)
        tmpdir = tempfile.mkdtemp(prefix="rag_indexing_test_")
        state_path = Path(tmpdir) / "quota_state.json"
        with open(state_path, "w") as f:
            json.dump({"date": "2026-06-06", "quota_used": 0}, f)

        plan = rag.load_and_index_data(
            daily_quota_used=0,
            daily_quota_limit=0,  # unlimited
            max_new_embeddings=num_chunks,
            embedding_batch_size=embedding_batch_size,
            rpm_limit=10_000,  # disable inter-batch sleep in tests
            quota_state_path=state_path,
        )
        return rag, plan

    def test_100_chunks_at_batch_size_32_makes_4_api_calls(self):
        """The cardinal anti-regression: 100 chunks must produce 4 API calls."""
        rag, plan = self._index(num_chunks=100, embedding_batch_size=32)
        self.assertEqual(rag.provider.embed.call_count, 4)
        self.assertEqual(plan.missing_chunks, 100)
        self.assertEqual(plan.embeddable_now, 100)

    def test_each_embed_call_receives_a_list_not_a_string(self):
        """Each call must be a batched list[str] — never a single string."""
        rag, _ = self._index(num_chunks=100, embedding_batch_size=32)
        for call in rag.provider.embed.call_args_list:
            args = call.args
            self.assertEqual(len(args), 1, "embed() must take exactly one positional arg")
            self.assertIsInstance(args[0], list, "embed() arg must be a list")
            self.assertGreater(len(args[0]), 0, "embed() batch must be non-empty")
            for text in args[0]:
                self.assertIsInstance(text, str)

    def test_no_batch_exceeds_embedding_batch_size(self):
        rag, _ = self._index(num_chunks=100, embedding_batch_size=32)
        for call in rag.provider.embed.call_args_list:
            batch = call.args[0]
            self.assertLessEqual(len(batch), 32)

    def test_total_chunks_across_batches_equals_missing_chunks(self):
        rag, _ = self._index(num_chunks=100, embedding_batch_size=32)
        total = sum(len(call.args[0]) for call in rag.provider.embed.call_args_list)
        self.assertEqual(total, 100)

    def test_batch_size_64_makes_2_api_calls(self):
        rag, _ = self._index(num_chunks=100, embedding_batch_size=64)
        self.assertEqual(rag.provider.embed.call_count, 2)
        sizes = [len(c.args[0]) for c in rag.provider.embed.call_args_list]
        self.assertEqual(sizes, [64, 36])

    def test_batch_size_16_makes_7_api_calls(self):
        rag, _ = self._index(num_chunks=100, embedding_batch_size=16)
        self.assertEqual(rag.provider.embed.call_count, 7)
        sizes = [len(c.args[0]) for c in rag.provider.embed.call_args_list]
        self.assertEqual(sizes, [16, 16, 16, 16, 16, 16, 4])

    def test_collection_upsert_called_once_per_batch(self):
        """Each embed batch is followed by exactly one ChromaDB upsert."""
        rag, _ = self._index(num_chunks=100, embedding_batch_size=32)
        self.assertEqual(rag.collection.upsert.call_count, 4)

    def test_upsert_receives_matching_documents_and_embeddings(self):
        """Sanity: upsert's documents and embeddings must align."""
        rag, _ = self._index(num_chunks=50, embedding_batch_size=20)
        for call in rag.collection.upsert.call_args_list:
            docs = call.kwargs.get("documents", call.args[0] if call.args else None)
            embs = call.kwargs.get("embeddings")
            self.assertIsNotNone(docs)
            self.assertIsNotNone(embs)
            self.assertEqual(len(docs), len(embs))


if __name__ == "__main__":
    unittest.main()
