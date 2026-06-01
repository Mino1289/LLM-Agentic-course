from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

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
    memory_summary: str
    memory_window: list[dict[str, str]]
    candidate_indices: list[int]
    candidate_chunks: list[str]
    candidate_metadatas: list[dict[str, Any]]
    final_indices: list[int]
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
    ):
        self.rag = rag
        self.memory_store = MemoryStore(window_size=memory_window_size)
        self.summarize_every_n_turns = max(4, summarize_every_n_turns)
        self.max_context_chunks = max(4, max_context_chunks)
        self.max_context_tokens = max(1200, max_context_tokens)
        self.decompose_query_count = max(3, decompose_query_count)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_query_node", self.prepare_query_node)
        graph.add_node("intent_scope_node", self.intent_scope_node)
        graph.add_node("clarify_node", self.clarify_node)
        graph.add_node("memory_read_node", self.memory_read_node)
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
        graph.add_edge("memory_read_node", "decompose_query_node")
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
        candidate_chunks = [self.rag.documents[idx] for idx in dedup_indices]
        candidate_metadatas = [self.rag.doc_metadata[idx] for idx in dedup_indices]

        stats = state.get("stats", {})
        stats.update(
            {
                "decomposed_query_count": len(queries),
                "retrieval_candidate_count": len(dedup_indices),
            }
        )
        return {
            "candidate_indices": dedup_indices,
            "candidate_chunks": candidate_chunks,
            "candidate_metadatas": candidate_metadatas,
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
            "final_indices": top_indices,
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
        prompt = (
            "Contexte conversationnel:\n"
            f"{memory_context}\n\n"
            "Historique de chat récent:\n"
            f"{message_context}\n\n"
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
