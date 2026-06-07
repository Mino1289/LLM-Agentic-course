from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import chromadb

os.environ["LANGSMITH_TRACING"] = "false"

from rag.hybrid_rag import HybridRAG, build_chunk_id, chunk_text_semantic
from rag.nodes.decompose_node import parse_query_list
from rag.nodes._v1_legacy.generation_node import format_retrieved_excerpts, synthesis_node
from rag.nodes.memory_nodes import context_prune_node
from rag.nodes.prepare_node import extract_metadata_filter, prepare_query_node
from rag.nodes.prompt_context import get_known_tickers
from rag.nodes.rerank_node import rerank_node
from rag.nodes.retrieval_node import multi_retrieve_node
from rag.nodes._v1_legacy.scope_node import query_scope_node
from rag.llm_provider import (
    LLMConfig,
    LLMProvider,
    LLMToolResponse,
    ToolCall,
    _normalize_github_model_id,
    _parse_openai_tool_calls,
)
from rag.nodes.agent_nodes import agent_node
from rag.nodes.tool_execution_node import tools_node
from rag.tool_schemas import ExportReportArgs, SimulatePortfolioArgs
from rag.tools import (
    _normalize_doc_types,
    run_export_investment_report,
    run_simulate_portfolio,
    run_validate_claims,
)
from rag.preprocess import SECTION_SPECS, _extract_between, is_in_year_range


async def _inline_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


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

        with patch("rag.nodes.retrieval_node.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(multi_retrieve_node(SimpleNamespace(rag=FakeRag()), state))

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

        with patch("rag.nodes.retrieval_node.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(multi_retrieve_node(SimpleNamespace(rag=rag), state))

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
        state = asyncio.run(prepare_query_node(
            SimpleNamespace(),
            {"query": "Compare MSFT et NVDA sur la croissance recente du chiffre d'affaires."},
        ))

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

        with patch("rag.nodes.rerank_node.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(rerank_node(agent, state))

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

        with patch("rag.nodes.memory_nodes.asyncio.to_thread", side_effect=_inline_to_thread):
            result = asyncio.run(context_prune_node(agent, state))

        self.assertEqual(result["final_chunks"], ["keep"])
        self.assertEqual(result["final_metadatas"], [{"source": "keep-meta"}])


class ConfigurationTests(unittest.TestCase):
    def test_universe_is_limited_to_debug_tickers(self):
        from rag.config import TRACKED_TICKERS

        rag = SimpleNamespace(doc_metadata=[{"ticker": "AMD"}, {"ticker": "ASML"}, {"ticker": "FAKE"}])

        result = get_known_tickers(SimpleNamespace(rag=rag), max_items=20)
        # Unknown tickers must be filtered, the rest comes from TRACKED_TICKERS order.
        self.assertNotIn("FAKE", result)
        self.assertIn("AMD", result)
        self.assertIn("ASML", result)
        self.assertEqual(len(result), len(TRACKED_TICKERS))

    def test_universe_respects_max_items(self):
        rag = SimpleNamespace(doc_metadata=[])

        result = get_known_tickers(SimpleNamespace(rag=rag), max_items=3)
        self.assertEqual(len(result), 3)

    def test_normalize_doc_types_includes_earnings_call(self):
        normalized = _normalize_doc_types(["earnings call", "10-K", "EARNINGS_CALL"])
        self.assertEqual(normalized, ["EARNINGS_CALL", "10-K"])

    def test_normalize_doc_types_includes_foreign_issuer_forms(self):
        normalized = _normalize_doc_types(["20-F", "6-K", "10-K"])
        self.assertEqual(normalized, ["20-F", "6-K", "10-K"])

    def test_file_type_infers_sec_form_from_source_name(self):
        from rag.hybrid_rag import extract_file_type_from_source

        self.assertEqual(
            extract_file_type_from_source("asml-20-f_2025-03-05.htm__foreign_annual_report.txt"),
            "20-F",
        )
        self.assertEqual(
            extract_file_type_from_source("arm-6-k_2025-01-01.htm__foreign_interim_report.txt"),
            "6-K",
        )

    def test_export_report_writes_markdown_file(self):
        with patch("rag.tools.REPORTS_DIR", Path(os.getenv("TMPDIR", "/tmp")) / "finance_rag_test_reports"):
            from rag import tools as tools_module

            tools_module.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            result = run_export_investment_report(ExportReportArgs(title="Test Report", content="## Section\nContent", format="md"))
            path = Path(result["path"])
            self.assertTrue(path.is_file())
            self.assertIn("# Test Report", path.read_text(encoding="utf-8"))
            path.unlink(missing_ok=True)

    def test_normalize_github_model_id_strips_openai_prefix(self):
        self.assertEqual(_normalize_github_model_id("openai/gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(_normalize_github_model_id("gpt-4.1-mini"), "gpt-4.1-mini")

    def test_gemini_config_uses_separate_embedding_provider(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-gemini",
                "OPENAI_API_KEY": "test-openai",
                "EMBEDDING_PROVIDER": "openai",
            },
            clear=False,
        ):
            from rag.llm_provider import build_llm_config_from_env

            config = build_llm_config_from_env()
            self.assertEqual(config.provider, "gemini")
            self.assertEqual(config.embedding_provider, "openai")
            self.assertEqual(config.embedding_api_key, "test-openai")

    def test_tool_definitions_count(self):
        from rag.tools import get_tool_definitions

        self.assertEqual(len(get_tool_definitions()), 5)

    def test_parse_openai_tool_calls(self):
        raw = [
            {
                "id": "call_1",
                "function": {"name": "market_price_tool", "arguments": '{"tickers":["MSFT"]}'},
            }
        ]
        parsed = _parse_openai_tool_calls(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].name, "market_price_tool")
        self.assertIn("MSFT", parsed[0].arguments)


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


class AgentToolsTests(unittest.TestCase):
    def test_validate_claims_supported_and_unsupported(self):
        from rag.tool_schemas import ValidateClaimsArgs
        from rag.tools import run_validate_claims

        chunks = [
            "Item 1A risk factors include supply chain concentration and regulatory scrutiny.",
            "Revenue growth accelerated in data center segment year over year.",
        ]
        metadatas = [
            {"ticker": "MSFT", "year": "2024", "file_type": "10-K"},
            {"ticker": "NVDA", "year": "2024", "file_type": "10-K"},
        ]
        canned = (
            '{"results": ['
            '{"claim": "MSFT supply chain risk", "status": "supported",'
            ' "best_source_index": 1, "reasoning": "Item 1A confirms supply chain risk."},'
            '{"claim": "Mars reactor", "status": "unsupported",'
            ' "best_source_index": null, "reasoning": "No relevant excerpt."}'
            ']}'
        )
        agent = SimpleNamespace(rag=SimpleNamespace(provider=MagicMock()))
        agent.rag.provider.generate.return_value = canned

        result = run_validate_claims(
            args=ValidateClaimsArgs(
                claims=[
                    "MSFT supply chain risk",
                    "The company operates a nuclear fusion reactor on Mars",
                ],
                chunks=chunks,
                metadatas=metadatas,
            ),
            agent=agent,
        )
        statuses = {v["status"] for v in result["validations"]}
        self.assertIn("supported", statuses)
        self.assertIn("unsupported", statuses)
        self.assertEqual(result["stats"]["validate_nli_used"], True)

    def test_validate_claims_requires_rag_chunks(self):
        from rag.tool_schemas import ValidateClaimsArgs
        from rag.tools import run_validate_claims

        agent = SimpleNamespace(rag=SimpleNamespace(provider=MagicMock()))
        result = run_validate_claims(
            args=ValidateClaimsArgs(claims=["test claim"], chunks=[], metadatas=[]),
            agent=agent,
        )
        self.assertIn("sec_filings_rag_tool", result["text"])
        agent.rag.provider.generate.assert_not_called()
        self.assertEqual(result["stats"]["validate_nli_used"], False)

    def test_simulate_portfolio_rejects_invalid_weights(self):
        bad_sum = run_simulate_portfolio(SimulatePortfolioArgs(allocations={"MSFT": 40, "NVDA": 40}))
        self.assertEqual(bad_sum.get("error"), "invalid_weights")

        bad_ticker = run_simulate_portfolio(SimulatePortfolioArgs(allocations={"ZZZZ": 100}))
        self.assertEqual(bad_ticker.get("error"), "invalid_tickers")

    def test_simulate_portfolio_valid_allocation(self):
        result = run_simulate_portfolio(SimulatePortfolioArgs(allocations={"MSFT": 50, "NVDA": 50}, notional_usd=10_000))
        self.assertEqual(len(result["positions"]), 2)
        self.assertAlmostEqual(sum(p["notional_usd"] for p in result["positions"]), 10_000, places=0)

    def test_agent_tool_loop_mocked(self):
        agent = SimpleNamespace(max_tool_iterations=6, rag=SimpleNamespace(provider=MagicMock()))

        first_response = LLMToolResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_rag",
                    name="sec_filings_rag_tool",
                    arguments='{"query":"MSFT risk factors 2024","tickers":["MSFT"]}',
                )
            ],
        )
        second_response = LLMToolResponse(content="Synthèse mock basée sur les outils.", tool_calls=[])

        agent.rag.provider.invoke_with_tools.side_effect = [first_response, second_response]

        state = {
            "normalized_query": "Risques MSFT 2024",
            "query": "Risques MSFT 2024",
            "messages": [],
            "agent_iterations": 0,
            "tool_events": [],
            "stats": {},
        }

        with patch("rag.tools.run_sec_filings_rag") as mock_rag:
            async def fake_rag(args, *, agent):
                return {
                    "text": "[1] ticker=MSFT excerpt",
                    "final_chunks": ["risk factors supply chain"],
                    "final_metadatas": [{"ticker": "MSFT", "year": "2024", "file_type": "10-K"}],
                    "stats": {"chunks_used": 1},
                }
            mock_rag.side_effect = fake_rag
            from rag.llm_provider import LLMStreamChunk

            # First ainvoke_with_tools_stream call → tool_call (sec_filings_rag_tool)
            # Second call → final answer text
            async def fake_astream_factory():
                responses = [
                    [
                        LLMStreamChunk(
                            tool_call_delta=[{
                                "id": "tc_1",
                                "name": "sec_filings_rag_tool",
                                "arguments": '{"query": "risk"}',
                            }],
                            finish_reason="tool_calls",
                        ),
                    ],
                    [
                        LLMStreamChunk(delta="Synthèse mock basée sur les outils."),
                        LLMStreamChunk(delta="", finish_reason="stop"),
                    ],
                ]
                idx = {"i": 0}
                async def stream(messages, tools=None, temperature=0.1, max_tokens=2000):
                    out = responses[idx["i"]]
                    idx["i"] += 1
                    for c in out:
                        yield c
                return stream

            stream_fn = asyncio.run(fake_astream_factory())
            agent.rag.provider.ainvoke_with_tools_stream = stream_fn

            after_agent = asyncio.run(agent_node(agent, state))
            self.assertTrue(after_agent.get("tool_calls_pending"))

            merged = {**state, **after_agent}
            after_tools = asyncio.run(tools_node(agent, merged))
            self.assertFalse(after_tools.get("tool_calls_pending"))
            self.assertTrue(after_tools.get("stats", {}).get("rag_tool_used"))

            final = asyncio.run(agent_node(agent, {**merged, **after_tools}))
            self.assertEqual(final.get("answer"), "Synthèse mock basée sur les outils.")
            self.assertGreaterEqual(len(final.get("tool_events", [])), 1)
            self.assertGreaterEqual(final.get("stats", {}).get("agent_iterations", 0), 2)

    def test_agent_streaming_tool_call_continuation_chunks_have_id_none(self):
        """Real OpenAI streaming protocol: only the FIRST chunk of a tool_call
        carries `id` and `name`; subsequent chunks for the SAME tool_call
        have `id=None` and `name=None` (only `arguments` is set).

        The accumulation in agent_node must NOT synthesize a new tool_call per
        chunk — that would break tool_call_id matching in the next LLM call
        (assistant.tool_calls ids no longer match the tool messages' tool_call_id).
        """
        from rag.llm_provider import LLMStreamChunk

        agent = SimpleNamespace(max_tool_iterations=6, rag=SimpleNamespace(provider=MagicMock()))
        REAL_ID = "call_r3YATjEm9WWID8AGbBzQddld"

        # Chunks 1 has id+name+empty args; chunks 2-3 have id=None, name=None,
        # only arguments delta. This is exactly what the OpenAI Python client
        # emits for `chat.completions.create(stream=True)`.
        responses = [
            [
                LLMStreamChunk(
                    tool_call_delta=[{
                        "id": REAL_ID,
                        "name": "sec_filings_rag_tool",
                        "arguments": '{"query":',
                    }],
                    finish_reason=None,
                ),
                LLMStreamChunk(
                    tool_call_delta=[{
                        "id": None,
                        "name": None,
                        "arguments": '"risks NVDA 2024"',
                    }],
                    finish_reason=None,
                ),
                LLMStreamChunk(
                    tool_call_delta=[{
                        "id": None,
                        "name": None,
                        "arguments": ',"tickers":["NVDA"]}',
                    }],
                    finish_reason="tool_calls",
                ),
            ],
            [
                LLMStreamChunk(delta="Réponse finale."),
                LLMStreamChunk(delta="", finish_reason="stop"),
            ],
        ]
        idx = {"i": 0}

        async def stream(messages, tools=None, temperature=0.1, max_tokens=2000):
            out = responses[idx["i"]]
            idx["i"] += 1
            for c in out:
                yield c

        agent.rag.provider.ainvoke_with_tools_stream = stream

        state = {
            "normalized_query": "Risques NVDA 2024",
            "query": "Risques NVDA 2024",
            "messages": [],
            "agent_iterations": 0,
            "tool_events": [],
            "stats": {},
        }

        with patch("rag.tools.run_sec_filings_rag") as mock_rag:
            async def fake_rag(args, *, agent):
                return {
                    "text": "[1] NVDA risk excerpt",
                    "final_chunks": ["risk"],
                    "final_metadatas": [{"ticker": "NVDA", "year": "2024", "file_type": "10-K"}],
                    "stats": {"chunks_used": 1},
                }

            mock_rag.side_effect = fake_rag

            after_agent = asyncio.run(agent_node(agent, state))
            # Only ONE tool_call must be produced from 3 chunks.
            pending = after_agent.get("pending_tool_calls") or []
            self.assertEqual(
                len(pending), 1,
                f"Expected 1 tool_call (3 continuation chunks of same call), got {len(pending)}: "
                f"{[(t.id, t.name, t.arguments) for t in pending]}",
            )
            # The id must be the REAL_ID from the first chunk — not a synthesized "tc_N".
            self.assertEqual(pending[0].id, REAL_ID)
            self.assertEqual(pending[0].name, "sec_filings_rag_tool")
            # Arguments must be the full JSON, not just the first chunk.
            self.assertEqual(pending[0].arguments, '{"query":"risks NVDA 2024","tickers":["NVDA"]}')

            merged = {**state, **after_agent}
            after_tools = asyncio.run(tools_node(agent, merged))

            # The tool message must use the SAME id as the assistant tool_call,
            # otherwise the next LLM call will fail with "missing tool_call_id response".
            lc_msgs = after_tools.get("lc_messages") or []
            tool_msgs = [m for m in lc_msgs if m.get("role") == "tool"]
            self.assertEqual(len(tool_msgs), 1)
            self.assertEqual(
                tool_msgs[0].get("tool_call_id"), REAL_ID,
                "tool_call_id in tool message must match the assistant tool_call id",
            )

            # Now simulate the second LLM call to confirm no API error
            # would arise. We don't actually call the API — we just verify
            # the message list is internally consistent.
            final = asyncio.run(agent_node(agent, {**merged, **after_tools}))
            self.assertEqual(final.get("answer"), "Réponse finale.")

    def test_agent_parallel_tool_calls_with_one_execution_failure(self):
        """When the LLM calls 2 tools in parallel and one fails (validation
        or execution), tools_node MUST still append a tool message for the
        failed one. Otherwise the next LLM call fails with:
          400 invalid_request_error: assistant.tool_calls ids without matching
          tool responses.
        This is an OpenAI API invariant, not a convention.
        """
        from rag.llm_provider import LLMStreamChunk

        agent = SimpleNamespace(max_tool_iterations=6, rag=SimpleNamespace(provider=MagicMock()))
        ID_OK = "call_ok_real"
        ID_BAD = "call_bad_real"

        responses = [
            [
                # Two parallel tool_calls with different index
                LLMStreamChunk(
                    tool_call_delta=[{
                        "index": 0, "id": ID_OK, "name": "sec_filings_rag_tool",
                        "arguments": '{"query":"risks NVDA 2024","tickers":["NVDA"]}',
                    }],
                    finish_reason=None,
                ),
                LLMStreamChunk(
                    tool_call_delta=[{
                        "index": 1, "id": ID_BAD, "name": "sec_filings_rag_tool",
                        "arguments": '{"query":"risks AMD 2024","tickers":["AMD"]}',
                    }],
                    finish_reason="tool_calls",
                ),
            ],
            [
                LLMStreamChunk(delta="Synthèse après outils partiels."),
                LLMStreamChunk(delta="", finish_reason="stop"),
            ],
        ]
        idx = {"i": 0}

        async def stream(messages, tools=None, temperature=0.1, max_tokens=2000):
            out = responses[idx["i"]]
            idx["i"] += 1
            for c in out:
                yield c

        agent.rag.provider.ainvoke_with_tools_stream = stream

        state = {
            "normalized_query": "Compare NVDA et AMD 2024",
            "query": "Compare NVDA et AMD 2024",
            "messages": [],
            "agent_iterations": 0,
            "tool_events": [],
            "stats": {},
        }

        with patch("rag.tools.run_sec_filings_rag") as mock_rag:
            async def rag_side_effect(args, *, agent):
                # First call (NVDA) succeeds; second call (AMD) raises
                # to simulate retrieval error.
                if args.tickers == ["NVDA"]:
                    return {
                        "text": "[1] NVDA excerpt",
                        "final_chunks": ["risk"],
                        "final_metadatas": [{"ticker": "NVDA", "year": "2024", "file_type": "10-K"}],
                        "stats": {"chunks_used": 1},
                    }
                raise RuntimeError("simulated retrieval failure for AMD")

            mock_rag.side_effect = rag_side_effect

            after_agent = asyncio.run(agent_node(agent, state))
            pending = after_agent.get("pending_tool_calls") or []
            # Both parallel calls must be present.
            self.assertEqual(len(pending), 2, f"Expected 2 parallel tool_calls, got {len(pending)}")
            ids = {t.id for t in pending}
            self.assertEqual(ids, {ID_OK, ID_BAD})

            merged = {**state, **after_agent}
            after_tools = asyncio.run(tools_node(agent, merged))

            # CRITICAL: a tool message MUST exist for BOTH ids, even though
            # one tool raised during execution. OpenAI rejects the next
            # request if any tool_call lacks a matching tool response.
            lc_msgs = after_tools.get("lc_messages") or []
            tool_msgs = [m for m in lc_msgs if m.get("role") == "tool"]
            tool_msg_ids = {m.get("tool_call_id") for m in tool_msgs}
            self.assertEqual(
                tool_msg_ids, {ID_OK, ID_BAD},
                f"tools_node must append a tool message for EVERY tool_call, "
                f"even on failure. Got tool_msg_ids={tool_msg_ids}, "
                f"expected={ {ID_OK, ID_BAD} }",
            )

            # The failed tool message should mention the error.
            bad_msg = next(m for m in tool_msgs if m.get("tool_call_id") == ID_BAD)
            self.assertIn("error", bad_msg.get("content", "").lower())

            # The successful tool message should have the actual result.
            ok_msg = next(m for m in tool_msgs if m.get("tool_call_id") == ID_OK)
            self.assertEqual(ok_msg.get("content"), "[1] NVDA excerpt")

            # tool_events should record both (one completed, one failed).
            events = after_tools.get("tool_events") or []
            # Full ToolEvents (with status) come from tools_node; agent_node
            # also appends a lighter "args summary" record. Filter to full ones.
            full_events = [e for e in events if "status" in e]
            rag_full = [e for e in full_events if e.get("tool") == "sec_filings_rag_tool"]
            self.assertEqual(len(rag_full), 2)
            statuses = [e.get("status") for e in rag_full]
            self.assertIn("completed", statuses)
            self.assertIn("failed", statuses)

    def test_run_sec_filings_rag_awaits_multi_retrieve_node(self):
        """run_sec_filings_rag is async (called from tools_node via await).
        It calls multi_retrieve_node (async) and decompose_query (async) —
        both must be driven to completion on the agent's event loop,
        otherwise the coroutine is never awaited and rag_state.update raises
        TypeError. The bug surfaces in the UI as "0 chunks" + the LLM
        apologizing for a 'technical problem accessing documents' (it
        actually received a 'coroutine is not iterable' error message).
        """
        import asyncio
        from rag.tools import run_sec_filings_rag
        from rag.tool_schemas import SecFilingsRAGArgs

        # Fake agent whose rag.retrieve returns 2 NVDA chunks.
        class FakeRetrieval:
            chunk_indices = [0, 1]

        class FakeRag:
            documents = {0: "nvda risk chunk A", 1: "nvda risk chunk B"}
            doc_metadata = [
                {"ticker": "NVDA", "year": "2024", "source": "nvda-10-k_2024.htm", "section": "Item_1A"},
                {"ticker": "NVDA", "year": "2024", "source": "nvda-10-k_2024.htm", "section": "Item_1A"},
            ]

            def __init__(self):
                self.retrieve_calls = 0

            def retrieve(self, query, **kwargs):
                self.retrieve_calls += 1
                return FakeRetrieval()

            def _deduplicate_indices(self, indices):
                return list(dict.fromkeys(indices))

            def _rerank(self, **kwargs):
                return kwargs.get("candidate_indices", [])

        # Avoid the rerank path complexity by stubbing _balanced_rerank_indices
        # to just return the candidates as-is.
        with patch("rag.tools._balanced_rerank_indices", return_value=[0, 1]), \
             patch("rag.tools._ticker_counts", return_value={"NVDA": 2}), \
             patch("rag.nodes.retrieval_node.asyncio.to_thread", side_effect=_inline_to_thread):
            agent = SimpleNamespace(
                rag=FakeRag(),
                max_tool_iterations=6,
            )
            args = SecFilingsRAGArgs(
                query="NVDA 10-K risk factors 2024",
                tickers=["NVDA"],
                years=["2024"],
                doc_types=["10-K"],
            )
            result = asyncio.run(run_sec_filings_rag(args, agent=agent))

        # Must return chunks (the bug returned 0 because the coroutine
        # was never awaited, causing rag_state.update to fail).
        self.assertEqual(
            len(result.get("final_chunks", [])), 2,
            f"run_sec_filings_rag must drive multi_retrieve_node to "
            f"completion via await. Got: {result}",
        )
        self.assertEqual(
            result.get("stats", {}).get("chunks_used"), 2,
            f"chunks_used stat must reflect retrieved chunks. Got: {result.get('stats')}",
        )
        self.assertGreater(agent.rag.retrieve_calls, 0, "rag.retrieve must be called")


class ValidateClaimsNLITests(unittest.TestCase):
    """PRD §2.2 + §4.4 — validate_claims_tool uses an LLM NLI judge, not
    token-overlap. NLI prompt must be invoked, JSON results parsed, fallback
    on parse/provider error, and NLI stats surfaced."""

    CHUNKS = [
        "Item 1A risk factors include supply chain concentration and regulatory scrutiny.",
        "Revenue growth accelerated in data center segment year over year.",
    ]
    METADATAS = [
        {"ticker": "MSFT", "year": "2024", "file_type": "10-K"},
        {"ticker": "NVDA", "year": "2024", "file_type": "10-K"},
    ]

    def _make_agent(self, canned_response):
        provider = MagicMock()
        provider.generate.return_value = canned_response
        rag = SimpleNamespace(provider=provider)
        return SimpleNamespace(rag=rag)

    def test_validate_uses_nli_path(self):
        from rag.tool_schemas import ValidateClaimsArgs
        from rag.tools import run_validate_claims

        agent = self._make_agent(
            '{"results": ['
            '{"claim": "MSFT supply chain risk", "status": "supported",'
            ' "best_source_index": 1, "reasoning": "Item 1A mentions supply chain risk."},'
            '{"claim": "Mars reactor", "status": "unsupported",'
            ' "best_source_index": null, "reasoning": "No relevant excerpt."}'
            ']}'
        )

        result = run_validate_claims(
            args=ValidateClaimsArgs(
                claims=["MSFT supply chain risk", "Mars reactor"],
                chunks=self.CHUNKS,
                metadatas=self.METADATAS,
            ),
            agent=agent,
        )

        agent.rag.provider.generate.assert_called_once()
        call_args = agent.rag.provider.generate.call_args
        call_kwargs = call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        prompt = call_kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        self.assertIn("NLI", system_prompt)
        self.assertEqual(call_kwargs.get("temperature"), 0.0)
        self.assertIn("supported", prompt)
        self.assertEqual(result["stats"]["validate_nli_used"], True)
        self.assertEqual(result["stats"]["validate_nli_claims"], 2)
        statuses = {v["status"] for v in result["validations"]}
        self.assertIn("supported", statuses)
        self.assertIn("unsupported", statuses)
        for v in result["validations"]:
            self.assertTrue(v.get("nli_used"))
            self.assertIn("reasoning", v)

    def test_validate_fallback_on_invalid_json(self):
        from rag.tool_schemas import ValidateClaimsArgs
        from rag.tools import run_validate_claims

        agent = self._make_agent("not valid json at all")
        result = run_validate_claims(
            args=ValidateClaimsArgs(
                claims=["claim A", "claim B"],
                chunks=self.CHUNKS,
                metadatas=self.METADATAS,
            ),
            agent=agent,
        )

        self.assertEqual(len(result["validations"]), 2)
        for v in result["validations"]:
            self.assertEqual(v["status"], "unsupported")
            self.assertIn("nli_parse_error", v["reasoning"])
        self.assertEqual(result["stats"]["validate_nli_used"], True)

    def test_validate_nli_skipped_when_chunks_empty(self):
        from rag.tool_schemas import ValidateClaimsArgs
        from rag.tools import run_validate_claims

        agent = self._make_agent("")
        result = run_validate_claims(
            args=ValidateClaimsArgs(claims=["any claim"], chunks=[], metadatas=[]),
            agent=agent,
        )
        agent.rag.provider.generate.assert_not_called()
        self.assertIn("sec_filings_rag_tool", result["text"])
        self.assertEqual(result["stats"]["validate_nli_used"], False)


class StateAuditTests(unittest.TestCase):
    """PRD §4.3 — Audit du State : aucune clé fantôme V1 ne doit être déclarée
    ni lue/écrite dans le flow actif."""

    GHOST_KEYS = {
        "intent_route",
        "ambiguous_query",
        "general_chat",
        "decomposed_queries",
        "price_tool_decision",
        "price_tool_used",
        "price_tool_attempts",
    }

    def test_graphstate_typeddict_has_no_ghosts(self):
        from rag.nodes.state import GraphState

        declared = set(GraphState.__annotations__.keys())
        overlap = self.GHOST_KEYS & declared
        self.assertFalse(
            overlap,
            f"Clés fantômes V1 toujours déclarées dans GraphState: {sorted(overlap)}",
        )


if __name__ == "__main__":
    unittest.main()
