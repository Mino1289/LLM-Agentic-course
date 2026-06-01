from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
import yfinance as yf

from rag.hybrid_rag import HybridRAG

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(func):
            return func
        return _decorator


class GraphState(TypedDict, total=False):
    conversation_id: str
    query: str
    messages: list[dict[str, str]]
    normalized_query: str
    metadata_filter: dict[str, str]
    ambiguous_query: bool
    clarification_question: str
    decomposed_queries: list[str]
    price_context: str
    price_tool_used: bool
    price_tool_attempts: int
    price_tool_decision: str
    price_tickers: list[str]
    price_window_start: str
    price_window_end: str
    memory_summary: str
    memory_window: list[dict[str, str]]
    candidate_indices: list[int]
    final_chunks: list[str]
    final_metadatas: list[dict[str, Any]]
    draft_answer: str
    answer: str
    gc_applied: bool
    stats: dict[str, Any]


@dataclass
class ConversationMemory:
    summary: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)
    last_chunk_fingerprints: set[str] = field(default_factory=set)


class MemoryStore:
    def __init__(self, window_size: int = 6):
        self.window_size = max(2, window_size)
        self._store: dict[str, ConversationMemory] = {}

    def get_or_create(self, conversation_id: str) -> ConversationMemory:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationMemory()
        return self._store[conversation_id]

    def get_window(self, conversation_id: str) -> list[dict[str, str]]:
        memory = self.get_or_create(conversation_id)
        return memory.turns[-self.window_size :]

    def append_turn(self, conversation_id: str, role: str, content: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.turns.append({"role": role, "content": content.strip()})

    def update_summary(self, conversation_id: str, new_summary: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.summary = new_summary.strip()

    def get_summary(self, conversation_id: str) -> str:
        return self.get_or_create(conversation_id).summary

    def trim_turns(self, conversation_id: str, keep_last: int) -> None:
        memory = self.get_or_create(conversation_id)
        memory.turns = memory.turns[-max(1, keep_last) :]

    def is_duplicate_chunk(self, conversation_id: str, chunk: str) -> bool:
        memory = self.get_or_create(conversation_id)
        fingerprint = chunk[:220].strip()
        return fingerprint in memory.last_chunk_fingerprints

    def remember_chunk(self, conversation_id: str, chunk: str) -> None:
        memory = self.get_or_create(conversation_id)
        memory.last_chunk_fingerprints.add(chunk[:220].strip())
        if len(memory.last_chunk_fingerprints) > 60:
            memory.last_chunk_fingerprints = set(list(memory.last_chunk_fingerprints)[-40:])


class FinanceLangGraphAgent:
    def __init__(
        self,
        rag: HybridRAG,
        memory_window_size: int = 6,
        summarize_every_n_turns: int = 6,
        max_context_chunks: int = 8,
        max_context_tokens: int = 3500,
        decompose_query_count: int = 4,
        price_max_days: int = 180,
        price_max_points: int = 40,
        price_max_tickers: int = 3,
        price_default_days: int = 90,
        price_max_attempts: int = 2,
    ):
        self.rag = rag
        self.memory_store = MemoryStore(window_size=memory_window_size)
        self.summarize_every_n_turns = max(4, summarize_every_n_turns)
        self.max_context_chunks = max(4, max_context_chunks)
        self.max_context_tokens = max(1200, max_context_tokens)
        self.decompose_query_count = max(3, decompose_query_count)
        self.price_max_days = max(30, price_max_days)
        self.price_max_points = max(10, price_max_points)
        self.price_max_tickers = max(1, price_max_tickers)
        self.price_default_days = max(15, min(price_default_days, self.price_max_days))
        self.price_max_attempts = max(1, price_max_attempts)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_query_node", self.prepare_query_node)
        graph.add_node("intent_scope_node", self.intent_scope_node)
        graph.add_node("clarify_node", self.clarify_node)
        graph.add_node("memory_read_node", self.memory_read_node)
        graph.add_node("tool_orchestrator_node", self.tool_orchestrator_node)
        graph.add_node("price_data_node", self.price_data_node)
        graph.add_node("decompose_query_node", self.decompose_query_node)
        graph.add_node("multi_retrieve_node", self.multi_retrieve_node)
        graph.add_node("rerank_node", self.rerank_node)
        graph.add_node("answer_generate_node", self.answer_generate_node)
        graph.add_node("synthesis_node", self.synthesis_node)
        graph.add_node("memory_write_node", self.memory_write_node)
        graph.add_node("gc_node", self.gc_node)

        graph.set_entry_point("prepare_query_node")
        graph.add_edge("prepare_query_node", "intent_scope_node")
        graph.add_conditional_edges(
            "intent_scope_node",
            self.route_after_intent_node,
            {
                "clarify": "clarify_node",
                "continue": "memory_read_node",
            },
        )
        graph.add_edge("clarify_node", END)
        graph.add_edge("memory_read_node", "tool_orchestrator_node")
        graph.add_conditional_edges(
            "tool_orchestrator_node",
            self.route_after_tool_orchestrator_node,
            {
                "call_price_tool": "price_data_node",
                "continue": "decompose_query_node",
            },
        )
        graph.add_edge("price_data_node", "tool_orchestrator_node")
        graph.add_edge("decompose_query_node", "multi_retrieve_node")
        graph.add_edge("multi_retrieve_node", "rerank_node")
        graph.add_edge("rerank_node", "answer_generate_node")
        graph.add_edge("answer_generate_node", "synthesis_node")
        graph.add_edge("synthesis_node", "memory_write_node")
        graph.add_edge("memory_write_node", "gc_node")
        graph.add_edge("gc_node", END)
        return graph.compile()

    @traceable(name="prepare_query_node")
    def prepare_query_node(self, state: GraphState) -> GraphState:
        raw = state.get("query", "")
        normalized = re.sub(r"\s+", " ", raw).strip()
        return {
            "normalized_query": normalized,
            "metadata_filter": self._extract_metadata_filter(normalized),
            "stats": {"pipeline": "langgraph_finance_v1"},
        }

    @traceable(name="intent_scope_node")
    def intent_scope_node(self, state: GraphState) -> GraphState:
        normalized_query = state.get("normalized_query", "")
        metadata_filter = state.get("metadata_filter", {})
        ambiguous_query = self._is_ambiguous_general_query(normalized_query, metadata_filter)
        return {"ambiguous_query": ambiguous_query}

    def route_after_intent_node(self, state: GraphState) -> str:
        return "clarify" if state.get("ambiguous_query", False) else "continue"

    @traceable(name="clarify_node")
    def clarify_node(self, state: GraphState) -> GraphState:
        query = state.get("normalized_query", "")
        clarification_question = (
            "Ta question est encore large. Tu veux une analyse sur quelle entreprise "
            "ou groupe d'entreprises, et sur quelle période (ex: 2024, 2023-2025) ?\n\n"
            f"Question reçue: {query}"
        )
        stats = state.get("stats", {})
        stats.update(
            {
                "chunks_used": 0,
                "gc_applied": False,
                "estimated_context_tokens": 0,
                "clarification_requested": True,
            }
        )
        return {"clarification_question": clarification_question, "stats": stats}

    @traceable(name="memory_read_node")
    def memory_read_node(self, state: GraphState) -> GraphState:
        conversation_id = state["conversation_id"]
        return {
            "memory_summary": self.memory_store.get_summary(conversation_id),
            "memory_window": self.memory_store.get_window(conversation_id),
        }

    @traceable(name="tool_orchestrator_node")
    def tool_orchestrator_node(self, state: GraphState) -> GraphState:
        query = state.get("normalized_query", "")
        metadata_filter = state.get("metadata_filter", {})
        messages = state.get("messages", [])
        attempts = state.get("price_tool_attempts", 0)
        current_price_context = state.get("price_context", "")

        should_try_price = self._should_fetch_price_context(query, metadata_filter, messages)
        enough_price_context = self._has_sufficient_price_context(current_price_context)
        tickers = self._extract_tickers_for_price_tool(query, metadata_filter)
        start_date, end_date = self._extract_price_date_window(query)
        if attempts > 0 and not enough_price_context:
            start_date, end_date = self._widen_price_window(start_date, end_date)

        decision = "continue"
        reason = "not_needed"

        if should_try_price and not enough_price_context:
            if not tickers:
                decision = "continue"
                reason = "no_ticker"
            elif attempts < self.price_max_attempts:
                decision = "call_price_tool"
                reason = "need_more_price_context"
            else:
                decision = "continue"
                reason = "max_attempts_reached"
        elif should_try_price and enough_price_context:
            decision = "continue"
            reason = "price_context_ready"
        elif not should_try_price:
            decision = "continue"
            reason = "not_needed"

        stats = state.get("stats", {})
        stats.update(
            {
                "price_tool_requested": should_try_price,
                "price_tool_attempts": attempts,
                "price_tool_decision": reason,
                "price_context_ready": enough_price_context,
                "price_tickers": tickers,
                "price_window_start": start_date,
                "price_window_end": end_date,
            }
        )

        return {
            "price_tool_decision": decision,
            "price_tickers": tickers,
            "price_window_start": start_date,
            "price_window_end": end_date,
            "stats": stats,
        }

    def route_after_tool_orchestrator_node(self, state: GraphState) -> str:
        return state.get("price_tool_decision", "continue")

    @traceable(name="price_data_node")
    def price_data_node(self, state: GraphState) -> GraphState:
        attempts = state.get("price_tool_attempts", 0) + 1
        tickers = state.get("price_tickers", [])
        start_date = state.get("price_window_start")
        end_date = state.get("price_window_end")

        if not tickers or not start_date or not end_date:
            stats = state.get("stats", {})
            stats.update(
                {
                    "price_tool_used": False,
                    "price_tool_reason": "orchestrator_missing_inputs",
                    "price_tool_attempts": attempts,
                }
            )
            return {
                "price_tool_used": False,
                "price_context": state.get("price_context", ""),
                "price_tool_attempts": attempts,
                "stats": stats,
            }

        summary = self._fetch_price_context(tickers, start_date, end_date)
        stats = state.get("stats", {})
        stats.update(
            {
                "price_tool_used": bool(summary),
                "price_tickers": tickers,
                "price_window_start": start_date,
                "price_window_end": end_date,
                "price_tool_attempts": attempts,
            }
        )
        return {
            "price_tool_used": bool(summary),
            "price_context": summary or state.get("price_context", ""),
            "price_tool_attempts": attempts,
            "stats": stats,
        }

    @traceable(name="decompose_query_node")
    def decompose_query_node(self, state: GraphState) -> GraphState:
        normalized_query = state["normalized_query"]
        metadata_filter = state.get("metadata_filter", {})
        if metadata_filter.get("ticker") and metadata_filter.get("year"):
            # Already very specific.
            decomposed = [normalized_query]
        else:
            decomposed = self._decompose_query(normalized_query)
        return {"decomposed_queries": decomposed}

    @traceable(name="multi_retrieve_node")
    def multi_retrieve_node(self, state: GraphState) -> GraphState:
        queries = state.get("decomposed_queries") or [state["normalized_query"]]
        metadata_filter = state.get("metadata_filter") or {}
        all_indices: list[int] = []
        for query in queries:
            retrieval = self.rag.retrieve(
                query,
                search_mode="vector",
                use_reranking=False,
                metadata_filter=metadata_filter or None,
                top_k=24,
                candidate_pool=24,
            )
            all_indices.extend(retrieval.chunk_indices)

            # If filter is too strict (often year), progressively relax it.
            if not retrieval.chunk_indices and metadata_filter:
                relaxed_filter = dict(metadata_filter)
                relaxed_filter.pop("year", None)
                if relaxed_filter:
                    relaxed_retrieval = self.rag.retrieve(
                        query,
                        search_mode="vector",
                        use_reranking=False,
                        metadata_filter=relaxed_filter,
                        top_k=24,
                        candidate_pool=24,
                    )
                    all_indices.extend(relaxed_retrieval.chunk_indices)

            if not retrieval.chunk_indices and metadata_filter:
                broad_retrieval = self.rag.retrieve(
                    query,
                    search_mode="vector",
                    use_reranking=False,
                    metadata_filter=None,
                    top_k=24,
                    candidate_pool=24,
                )
                all_indices.extend(broad_retrieval.chunk_indices)

        dedup_indices = self.rag._deduplicate_indices(all_indices)

        stats = state.get("stats", {})
        stats.update(
            {
                "decomposed_query_count": len(queries),
                "retrieval_candidate_count": len(dedup_indices),
            }
        )
        return {
            "candidate_indices": dedup_indices,
            "stats": stats,
        }

    @traceable(name="rerank_node")
    def rerank_node(self, state: GraphState) -> GraphState:
        candidates = state.get("candidate_indices", [])
        if not candidates:
            return {"final_indices": [], "final_chunks": [], "final_metadatas": []}

        top_indices = self.rag._rerank(state["normalized_query"], candidates, top_k=self.max_context_chunks)
        final_chunks = [self.rag.documents[idx] for idx in top_indices]
        final_meta = [self.rag.doc_metadata[idx] for idx in top_indices]
        return {
            "final_chunks": final_chunks,
            "final_metadatas": final_meta,
        }

    @traceable(name="answer_generate_node")
    def answer_generate_node(self, state: GraphState) -> GraphState:
        final_chunks = state.get("final_chunks", [])
        if not final_chunks:
            return {
                "draft_answer": (
                    "Je ne trouve pas assez de sources fiables pour répondre précisément. "
                    "Peux-tu préciser une entreprise, une période, ou un angle (risques, "
                    "catalyseurs, marges, guidance) ?"
                )
            }

        memory_context = self._format_memory_context(
            state.get("memory_summary", ""),
            state.get("memory_window", []),
        )
        message_context = self._format_chat_context(state.get("messages", []))
        chunks_context = "\n\n---\n\n".join(final_chunks)
        price_context = state.get("price_context", "")
        price_block = f"Contexte prix de marché:\n{price_context}\n\n" if price_context else ""
        prompt = (
            "Contexte conversationnel:\n"
            f"{memory_context}\n\n"
            "Historique de chat récent:\n"
            f"{message_context}\n\n"
            f"{price_block}"
            "Extraits financiers récupérés:\n"
            f"{chunks_context}\n\n"
            f"Question utilisateur: {state['normalized_query']}\n\n"
            "Réponds en français avec une analyse financière structurée."
        )
        draft = self.rag.provider.generate(prompt, temperature=0.1, max_tokens=900)
        return {"draft_answer": draft}

    @traceable(name="synthesis_node")
    def synthesis_node(self, state: GraphState) -> GraphState:
        chunk_count = len(state.get("final_chunks", []))
        if chunk_count < 2:
            return {
                "answer": (
                    "Le contexte source est limité pour une réponse ferme. "
                    "Je peux donner une vue générale, mais elle reste incertaine. "
                    "Si tu veux une réponse fiable, précise entreprise et période."
                )
            }

        prompt = (
            "Synthétise et clarifie cette réponse d'analyse financière.\n"
            "Conserve uniquement les éléments actionnables et factuels.\n"
            "Format strict en 5 sections:\n"
            "1) Synthèse\n2) Faits observés\n3) Interprétations\n4) Incertitudes\n5) Conclusion\n\n"
            f"Texte à synthétiser:\n{state.get('draft_answer', '')}"
        )
        final_answer = self.rag.provider.generate(prompt, temperature=0.0, max_tokens=700)
        return {"answer": final_answer}

    @traceable(name="memory_write_node")
    def memory_write_node(self, state: GraphState) -> GraphState:
        conversation_id = state["conversation_id"]
        self.memory_store.append_turn(conversation_id, "user", state.get("normalized_query", ""))
        self.memory_store.append_turn(conversation_id, "assistant", state.get("answer", ""))
        return {}

    @traceable(name="gc_node")
    def gc_node(self, state: GraphState) -> GraphState:
        conversation_id = state["conversation_id"]
        memory = self.memory_store.get_or_create(conversation_id)
        gc_applied = False

        if len(memory.turns) >= self.summarize_every_n_turns:
            transcript = "\n".join(f"{t['role']}: {t['content']}" for t in memory.turns[:-self.memory_store.window_size])
            if transcript.strip():
                summary_prompt = (
                    "Résume la conversation suivante en 8 lignes max pour mémoire long-terme d'un assistant financier.\n"
                    f"Mémoire existante: {memory.summary}\n\n"
                    f"Conversation à compresser:\n{transcript}"
                )
                new_summary = self.rag.provider.generate(summary_prompt, temperature=0.0, max_tokens=300)
                self.memory_store.update_summary(conversation_id, new_summary)
                self.memory_store.trim_turns(conversation_id, keep_last=self.memory_store.window_size)
                gc_applied = True

        deduped_chunks = []
        for chunk in state.get("final_chunks", []):
            if not self.memory_store.is_duplicate_chunk(conversation_id, chunk):
                deduped_chunks.append(chunk)
        if deduped_chunks:
            state["final_chunks"] = deduped_chunks

        # Hard cap context tokens to reduce API spend.
        final_chunks = state.get("final_chunks", [])
        while final_chunks and self.rag.count_context_tokens(final_chunks) > self.max_context_tokens:
            final_chunks = final_chunks[:-1]
            gc_applied = True
        state["final_chunks"] = final_chunks

        for chunk in state.get("final_chunks", []):
            self.memory_store.remember_chunk(conversation_id, chunk)

        stats = state.get("stats", {})
        stats.update(
            {
                "chunks_used": len(state.get("final_chunks", [])),
                "gc_applied": gc_applied,
                "estimated_context_tokens": self.rag.count_context_tokens(state.get("final_chunks", [])),
                "price_tool_used": state.get("price_tool_used", False),
            }
        )
        return {"gc_applied": gc_applied, "stats": stats}

    @staticmethod
    def _format_memory_context(summary: str, window: list[dict[str, str]]) -> str:
        parts = []
        if summary:
            parts.append(f"Résumé mémoire: {summary}")
        if window:
            turns = "\n".join(f"{t['role']}: {t['content']}" for t in window)
            parts.append(f"Derniers échanges:\n{turns}")
        return "\n\n".join(parts) if parts else "Aucun contexte mémorisé."

    @staticmethod
    def _extract_metadata_filter(query: str) -> dict[str, str]:
        filter_payload: dict[str, str] = {}

        ticker_match = re.search(r"\b([A-Z]{2,5})\b", query)
        if ticker_match:
            filter_payload["ticker"] = ticker_match.group(1)

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            filter_payload["year"] = year_match.group(1)

        return filter_payload

    def _decompose_query(self, query: str) -> list[str]:
        prompt = (
            "Décompose la requête finance suivante en sous-requêtes ciblées pour retrieval RAG.\n"
            f"Requête: {query}\n\n"
            f"Rends STRICTEMENT un JSON array de {self.decompose_query_count} à "
            f"{self.decompose_query_count + 2} chaînes courtes, sans autre texte."
        )
        raw = self.rag.provider.generate(prompt, temperature=0.0, max_tokens=350)
        parsed = self._parse_query_list(raw)
        if not parsed:
            parsed = [query]
        if query not in parsed:
            parsed.insert(0, query)
        return parsed[: self.decompose_query_count + 2]

    @staticmethod
    def _parse_query_list(raw: str) -> list[str]:
        text = raw.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except Exception:
            pass
        # Fallback line parsing if model did not produce valid JSON.
        lines = [ln.strip("-• \t") for ln in text.splitlines()]
        return [ln for ln in lines if ln]

    @staticmethod
    def _is_ambiguous_general_query(query: str, metadata_filter: dict[str, str]) -> bool:
        if metadata_filter.get("ticker") or metadata_filter.get("year"):
            return False
        q = query.lower()
        domain_keywords = [
            "risque",
            "catalyseur",
            "croissance",
            "marge",
            "profit",
            "guidance",
            "valorisation",
            "concurrence",
            "opportunit",
            "secteur",
            "gpu",
            "ia",
            "semi",
            "entreprise",
        ]
        has_domain_signal = any(token in q for token in domain_keywords)
        return len(q) < 20 or not has_domain_signal

    @staticmethod
    def _format_chat_context(messages: list[dict[str, str]], keep_last: int = 6) -> str:
        if not messages:
            return "Aucun historique de chat."
        selected = messages[-keep_last:]
        formatted = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in selected]
        return "\n".join(formatted)

    @staticmethod
    def _has_sufficient_price_context(price_context: str) -> bool:
        if not price_context:
            return False
        # Minimal signal: has global window + at least one sampled points line.
        return "Fenetre prix:" in price_context and "points[" in price_context

    def _should_fetch_price_context(
        self,
        query: str,
        metadata_filter: dict[str, str],
        messages: list[dict[str, str]],
    ) -> bool:
        if os.getenv("PRICE_TOOL_ENABLED", "true").lower() in {"0", "false", "no"}:
            return False

        lower_q = query.lower()
        explicit_price_keywords = [
            "prix",
            "cours",
            "performance",
            "rendement",
            "drawdown",
            "volatilit",
            "return",
            "returns",
            "stock price",
            "chart",
            "graphique",
        ]
        contextual_finance_keywords = [
            "risque",
            "catalyseur",
            "thèse",
            "these",
            "allocation",
            "compar",
            "opportunit",
            "investissement",
            "positionn",
            "court terme",
            "momentum",
            "sentiment",
        ]

        has_ticker = bool(metadata_filter.get("ticker")) or bool(
            re.search(r"\b[A-Z]{2,5}\b", query)
        )
        explicit_price = any(k in lower_q for k in explicit_price_keywords)
        contextual_finance = any(k in lower_q for k in contextual_finance_keywords)

        if explicit_price:
            return True
        if has_ticker and contextual_finance:
            return True
        if "semiconduct" in lower_q or "semi-conduct" in lower_q or "semi conduct" in lower_q:
            return True

        # If user just asked for a comparison in follow-up, price context can help.
        if messages:
            last_user_turns = [
                m.get("content", "").lower() for m in messages[-3:] if m.get("role") == "user"
            ]
            if any("compare" in t or "compar" in t for t in last_user_turns) and has_ticker:
                return True
        return False

    def _extract_tickers_for_price_tool(self, query: str, metadata_filter: dict[str, str]) -> list[str]:
        tracked = ["NVDA", "INTC", "AMD", "PLTR", "GOOGL", "META", "AMZN", "MSFT", "AVGO", "ORCL"]
        found = []
        if metadata_filter.get("ticker"):
            found.append(metadata_filter["ticker"].upper())

        for ticker in re.findall(r"\b[A-Z]{2,5}\b", query):
            candidate = ticker.upper()
            if candidate in tracked and candidate not in found:
                found.append(candidate)

        if not found:
            q = query.lower()
            if "semiconduct" in q or "semi-conduct" in q or "semi conduct" in q:
                found = ["NVDA", "AMD", "INTC"]

        return found[: self.price_max_tickers]

    def _extract_price_date_window(self, query: str) -> tuple[str, str]:
        today = datetime.utcnow().date()
        default_start = today - timedelta(days=self.price_default_days)
        start_date = default_start
        end_date = today

        # Explicit ISO dates.
        explicit_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
        if len(explicit_dates) >= 2:
            try:
                d1 = datetime.strptime(explicit_dates[0], "%Y-%m-%d").date()
                d2 = datetime.strptime(explicit_dates[1], "%Y-%m-%d").date()
                start_date, end_date = (d1, d2) if d1 <= d2 else (d2, d1)
            except Exception:
                pass
        elif re.search(r"\b20\d{2}\b", query):
            year = int(re.search(r"\b(20\d{2})\b", query).group(1))
            start_date = datetime(year, 1, 1).date()
            end_date = datetime(year, 12, 31).date()

        # Relative periods if provided.
        rel_match = re.search(r"\b(\d+)\s*(jour|jours|day|days|mois|month|months|an|ans|year|years)\b", query.lower())
        if rel_match:
            value = int(rel_match.group(1))
            unit = rel_match.group(2)
            if "jour" in unit or "day" in unit:
                delta_days = value
            elif "mois" in unit or "month" in unit:
                delta_days = value * 30
            else:
                delta_days = value * 365
            delta_days = min(delta_days, self.price_max_days)
            start_date = today - timedelta(days=delta_days)
            end_date = today

        # Hard cap for token/cost control.
        if (end_date - start_date).days > self.price_max_days:
            start_date = end_date - timedelta(days=self.price_max_days)

        return start_date.isoformat(), end_date.isoformat()

    def _fetch_price_context(self, tickers: list[str], start_date: str, end_date: str) -> str:
        lines: list[str] = []
        per_ticker_point_budget = max(5, self.price_max_points // max(1, len(tickers)))
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
            except Exception:
                continue
            if df is None or df.empty:
                continue

            if "Close" not in df.columns:
                continue
            close = df["Close"]
            # yfinance can return a DataFrame (e.g. MultiIndex columns) even for one ticker.
            if getattr(close, "ndim", 1) == 2:
                if close.shape[1] == 0:
                    continue
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) == 0:
                continue

            returns = close.pct_change().dropna()
            perf = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 1 else 0.0
            vol = returns.std() * (252**0.5) * 100 if not returns.empty else 0.0
            rolling_max = close.cummax()
            drawdown = ((close / rolling_max) - 1).min() * 100 if len(close) > 1 else 0.0

            lines.append(
                f"- {ticker}: perf={perf:.2f}%, vol_ann={vol:.2f}%, max_drawdown={drawdown:.2f}%, "
                f"close_min={close.min():.2f}, close_max={close.max():.2f}, "
                f"close_last={close.iloc[-1]:.2f}"
            )

            # Sample points for factual anchoring with controlled token footprint.
            step = max(1, len(close) // per_ticker_point_budget)
            sampled = close.iloc[::step].tail(per_ticker_point_budget)
            points = ", ".join(
                f"{self._format_price_index(idx)}={float(val):.2f}" for idx, val in sampled.items()
            )
            lines.append(f"  points[{ticker}]: {points}")

        if not lines:
            return ""
        return (
            f"Fenetre prix: {start_date} -> {end_date}\n"
            + "\n".join(lines)
        )

    def _widen_price_window(self, start_date: str, end_date: str) -> tuple[str, str]:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception:
            return start_date, end_date

        widened_start = start - timedelta(days=self.price_default_days)
        max_start = end - timedelta(days=self.price_max_days)
        if widened_start < max_start:
            widened_start = max_start
        return widened_start.isoformat(), end.isoformat()

    @staticmethod
    def _format_price_index(idx: Any) -> str:
        if hasattr(idx, "date"):
            return idx.date().isoformat()
        return str(idx)

    @traceable(name="finance_langgraph_run")
    def run(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> GraphState:
        convo_id = conversation_id or str(uuid.uuid4())
        initial_state: GraphState = {
            "conversation_id": convo_id,
            "query": query,
            "messages": messages or [],
        }
        return self.graph.invoke(initial_state)
