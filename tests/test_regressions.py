from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chromadb

os.environ["LANGSMITH_TRACING"] = "false"

from rag.hybrid_rag import HybridRAG, build_chunk_id, chunk_text_semantic
from rag.nodes.decompose_node import parse_query_list
from rag.nodes.generation_node import format_retrieved_excerpts, synthesis_node
from rag.nodes.memory_nodes import context_prune_node
from rag.nodes.prepare_node import extract_metadata_filter, prepare_query_node
from rag.nodes.prompt_context import get_known_tickers
from rag.nodes.rerank_node import rerank_node
from rag.nodes.retrieval_node import multi_retrieve_node
from rag.nodes.scope_node import query_scope_node
from rag.nodes.tool_nodes import tool_orchestrator_node
from rag.preprocess import SECTION_SPECS, _extract_between, is_in_year_range


class ExtractionTests(unittest.TestCase):
    def test_extract_between_skips_table_of_contents_occurrence(self):
        spec = SECTION_SPECS["Item_7"]
        toc = "ITEM 7. MANAGEMENT'S DISCUSSION\n" + ("toc filler " * 80)
        body = "\nITEM 7. MANAGEMENT'S DISCUSSION\n" + ("REAL_BODY " * 80)
        text = toc + "\nITEM 2. PROPERTIES\n" + ("between " * 100) + body + "\nITEM 7A."

        result = _extract_between(text, spec["starts"], spec["ends"])

        self.assertIn("REAL_BODY", result)
        self.assertNotIn("ITEM 2. PROPERTIES", result)

    def test_year_range_applies_to_every_filing_type(self):
        self.assertTrue(is_in_year_range("amd-10-k_2024-01-01.htm", 2024, 2026))
        self.assertFalse(is_in_year_range("amd-10-q_2023-01-01.htm", 2024, 2026))
        self.assertFalse(is_in_year_range("amd-8-k_2027-01-01.htm", 2024, 2026))
        self.assertFalse(is_in_year_range("transcript-without-date.txt", 2024, 2026))


class ChunkingTests(unittest.TestCase):
    def test_long_sentences_are_split_without_data_loss(self):
        text = ("A" * 150) + ". " + ("B" * 150) + "."

        chunks = chunk_text_semantic(text, max_size=100)

        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(sum(chunk.count("A") for chunk in chunks), 150)
        self.assertEqual(sum(chunk.count("B") for chunk in chunks), 150)

    def test_chunk_id_changes_when_content_changes(self):
        old_id = build_chunk_id("doc", "Item_7", "semantic", 0, "old text")
        new_id = build_chunk_id("doc", "Item_7", "semantic", 0, "new text")

        self.assertNotEqual(old_id, new_id)


class RetrievalTests(unittest.TestCase):
    def test_vector_search_applies_metadata_filter_inside_chroma(self):
        rag = object.__new__(HybridRAG)
        rag.documents = ["amd text", "nvda text"]
        rag.doc_metadata = [
            {"ticker": "AMD", "year": "2024"},
            {"ticker": "NVDA", "year": "2024"},
        ]
        rag.chunk_ids = ["amd", "nvda"]
        rag.provider = SimpleNamespace(embed=lambda _texts: [[1.0, 0.0]])
        rag.collection = chromadb.EphemeralClient().get_or_create_collection("metadata-filter-test")
        rag.collection.add(
            ids=rag.chunk_ids,
            documents=rag.documents,
            metadatas=rag.doc_metadata,
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
        )

        result = rag.retrieve(
            "query",
            metadata_filter={"ticker": "AMD", "year": "2024"},
            use_reranking=False,
        )

        self.assertEqual(result.chunk_indices, [0])

    def test_scoped_retrieval_does_not_fallback_to_another_company(self):
        class Retrieval:
            def __init__(self, indices):
                self.chunk_indices = indices

        class FakeRag:
            doc_metadata = [{"ticker": "NVDA", "source": "nvda-10-k_2024.htm", "section": "Item_7"}]

            def retrieve(self, _query, **kwargs):
                return Retrieval([] if kwargs.get("metadata_filter") else [0])

            def _deduplicate_indices(self, indices):
                return list(dict.fromkeys(indices))

        state = {
            "normalized_query": "AMD 2024 risques",
            "decomposed_queries": ["AMD 2024 risques"],
            "metadata_filter": {"ticker": "AMD", "year": "2024"},
            "target_tickers": ["AMD"],
            "doc_type_priority": ["10-K"],
            "stats": {},
        }

        result = multi_retrieve_node(SimpleNamespace(rag=FakeRag()), state)

        self.assertEqual(result["candidate_indices"], [])

    def test_multi_retrieve_uses_explicit_query_tickers_even_if_scope_lost_one(self):
        class Retrieval:
            def __init__(self, indices):
                self.chunk_indices = indices

        class FakeRag:
            doc_metadata = [
                {"ticker": "MSFT", "source": "msft-10-q_2026.htm", "section": "Item_7"},
                {"ticker": "NVDA", "source": "nvda-10-q_2026.htm", "section": "Item_7"},
            ]

            def __init__(self):
                self.calls = []

            def retrieve(self, query, **kwargs):
                ticker = (kwargs.get("metadata_filter") or {}).get("ticker")
                self.calls.append((query, ticker))
                if ticker == "MSFT":
                    return Retrieval([0])
                if ticker == "NVDA":
                    return Retrieval([1])
                return Retrieval([])

            def _deduplicate_indices(self, indices):
                return list(dict.fromkeys(indices))

        rag = FakeRag()
        state = {
            "normalized_query": "Compare MSFT et NVDA sur la croissance recente du chiffre d'affaires.",
            "decomposed_queries": ["Compare MSFT et NVDA sur la croissance recente du chiffre d'affaires."],
            "metadata_filter": {},
            "target_tickers": ["MSFT"],
            "doc_type_priority": ["10-Q"],
            "stats": {},
        }

        result = multi_retrieve_node(SimpleNamespace(rag=rag), state)

        self.assertEqual(result["candidate_indices"], [0, 1])
        nvda_queries = [query for query, ticker in rag.calls if ticker == "NVDA"]
        self.assertTrue(nvda_queries)
        self.assertNotIn("MSFT", nvda_queries[0])


class DecomposeTests(unittest.TestCase):
    def test_parse_query_list_handles_json_code_fence(self):
        raw = '```json\n["MSFT revenue growth", "NVDA revenue growth"]\n```'

        self.assertEqual(
            parse_query_list(raw),
            ["MSFT revenue growth", "NVDA revenue growth"],
        )


class ScopeTests(unittest.TestCase):
    def test_scope_preserves_explicit_tickers_when_llm_returns_subset(self):
        class Provider:
            def generate(self, *_args, **_kwargs):
                return '{"target_tickers":["MSFT"],"doc_type_priority":["10-Q"],"reason":"subset"}'

        agent = SimpleNamespace(
            rag=SimpleNamespace(
                provider=Provider(),
                doc_metadata=[{"ticker": "MSFT"}, {"ticker": "NVDA"}],
            )
        )
        state = {
            "normalized_query": "Compare MSFT et NVDA sur la croissance recente du chiffre d'affaires.",
            "metadata_filter": {},
            "target_tickers": ["MSFT", "NVDA"],
            "stats": {},
        }

        result = query_scope_node(agent, state)

        self.assertEqual(result["target_tickers"], ["MSFT", "NVDA"])


class PrepareTests(unittest.TestCase):
    def test_metadata_filter_ignores_uppercase_non_ticker_tokens(self):
        result = extract_metadata_filter("Que dit la SEC sur les risques AI et GPU en 2024 ?")

        self.assertEqual(result, {"year": "2024"})

    def test_metadata_filter_keeps_tracked_ticker(self):
        result = extract_metadata_filter("AMD: risques AI et GPU en 2024")

        self.assertEqual(result, {"ticker": "AMD", "year": "2024"})

    def test_comparison_query_keeps_all_tickers_without_single_ticker_filter(self):
        state = prepare_query_node(
            SimpleNamespace(),
            {"query": "Compare MSFT et NVDA sur la croissance recente du chiffre d'affaires."},
        )

        self.assertEqual(state["target_tickers"], ["MSFT", "NVDA"])
        self.assertNotIn("ticker", state["metadata_filter"])


class RerankTests(unittest.TestCase):
    def test_rerank_balances_chunks_across_target_tickers(self):
        class Rag:
            documents = ["msft 1", "msft 2", "msft 3", "nvda 1", "nvda 2", "nvda 3"]
            doc_metadata = [
                {"ticker": "MSFT"},
                {"ticker": "MSFT"},
                {"ticker": "MSFT"},
                {"ticker": "NVDA"},
                {"ticker": "NVDA"},
                {"ticker": "NVDA"},
            ]

            def _rerank(self, _query, indices, top_k):
                return indices[:top_k]

        agent = SimpleNamespace(rag=Rag(), max_context_chunks=4)
        state = {
            "normalized_query": "Compare MSFT et NVDA revenue growth",
            "target_tickers": ["MSFT", "NVDA"],
            "candidate_indices": [0, 1, 2, 3, 4, 5],
        }

        result = rerank_node(agent, state)

        self.assertEqual(
            [meta["ticker"] for meta in result["final_metadatas"]],
            ["MSFT", "MSFT", "NVDA", "NVDA"],
        )
        self.assertEqual(result["stats"]["rerank_final_ticker_counts"], {"MSFT": 2, "NVDA": 2})


class GenerationTests(unittest.TestCase):
    def test_retrieved_excerpts_include_metadata_labels(self):
        result = format_retrieved_excerpts(
            ["microsoft excerpt", "nvidia excerpt"],
            [
                {"ticker": "MSFT", "source": "msft-10-q_2026.htm", "section": "Item_7", "year": "2026"},
                {"ticker": "NVDA", "source": "nvda-10-q_2026.htm", "section": "Item_7", "year": "2026"},
            ],
        )

        self.assertIn("Ticker: MSFT", result)
        self.assertIn("Ticker: NVDA", result)
        self.assertIn("Source: nvda-10-q_2026.htm", result)

    def test_single_chunk_synthesis_preserves_grounded_draft(self):
        state = {
            "final_chunks": ["source chunk"],
            "draft_answer": "AMD mentionne un risque de demande cyclique.",
        }

        result = synthesis_node(SimpleNamespace(), state)

        self.assertIn("AMD mentionne un risque de demande cyclique.", result["answer"])
        self.assertIn("un seul extrait", result["answer"])

    def test_synthesis_prompt_preserves_target_tickers(self):
        class Provider:
            def __init__(self):
                self.prompt = ""

            def generate(self, prompt, **_kwargs):
                self.prompt = prompt
                return "synthese"

        provider = Provider()
        agent = SimpleNamespace(rag=SimpleNamespace(provider=provider))
        state = {
            "normalized_query": "Compare MSFT et NVDA",
            "target_tickers": ["MSFT", "NVDA"],
            "final_chunks": ["msft chunk", "nvda chunk"],
            "draft_answer": "MSFT ... NVDA ...",
        }

        synthesis_node(agent, state)

        self.assertIn("Target tickers: MSFT, NVDA", provider.prompt)
        self.assertIn("do not collapse the answer to only one company", provider.prompt)


class ContextPruneTests(unittest.TestCase):
    def test_context_pruning_keeps_metadata_aligned(self):
        class MemoryStore:
            def is_duplicate_chunk(self, _conversation_id, chunk):
                return chunk == "drop"

        class Rag:
            def count_context_tokens(self, chunks):
                return len(chunks) * 1000

        agent = SimpleNamespace(memory_store=MemoryStore(), rag=Rag(), max_context_tokens=1200)
        state = {
            "conversation_id": "conversation",
            "final_chunks": ["keep", "drop", "tail"],
            "final_metadatas": [
                {"source": "keep-meta"},
                {"source": "drop-meta"},
                {"source": "tail-meta"},
            ],
            "stats": {},
        }

        result = context_prune_node(agent, state)

        self.assertEqual(result["final_chunks"], ["keep"])
        self.assertEqual(result["final_metadatas"], [{"source": "keep-meta"}])


class ConfigurationTests(unittest.TestCase):
    def test_universe_is_limited_to_debug_tickers(self):
        rag = SimpleNamespace(doc_metadata=[{"ticker": "AMD"}, {"ticker": "INTC"}])

        self.assertEqual(get_known_tickers(SimpleNamespace(rag=rag), max_items=20), ["AMD", "NVDA", "MSFT"])

    def test_price_tool_can_be_disabled(self):
        agent = SimpleNamespace()
        state = {
            "normalized_query": "prix AMD",
            "price_tool_attempts": 0,
            "stats": {},
        }

        with patch.dict("os.environ", {"PRICE_TOOL_ENABLED": "false"}):
            result = tool_orchestrator_node(agent, state)

        self.assertEqual(result["price_tool_decision"], "continue")
        self.assertEqual(result["stats"]["price_tool_decision"], "disabled_by_config")


class IndexSynchronizationTests(unittest.TestCase):
    def test_stale_index_entries_are_deleted(self):
        class Collection:
            def __init__(self):
                self.deleted = []

            def get(self):
                return {"ids": ["keep", "stale"]}

            def delete(self, ids):
                self.deleted.extend(ids)

        rag = object.__new__(HybridRAG)
        rag.collection = Collection()

        removed = rag.remove_stale_index_entries(["keep"])

        self.assertEqual(removed, 1)
        self.assertEqual(rag.collection.deleted, ["stale"])


if __name__ == "__main__":
    unittest.main()
