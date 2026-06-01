from __future__ import annotations

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
    normalized_query: str
    metadata_filter: dict[str, str]
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
    ):
        self.rag = rag
        self.memory_store = MemoryStore(window_size=memory_window_size)
        self.summarize_every_n_turns = max(4, summarize_every_n_turns)
        self.max_context_chunks = max(4, max_context_chunks)
        self.max_context_tokens = max(1200, max_context_tokens)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_query_node", self.prepare_query_node)
        graph.add_node("memory_read_node", self.memory_read_node)
        graph.add_node("retrieve_node", self.retrieve_node)
        graph.add_node("rerank_node", self.rerank_node)
        graph.add_node("answer_generate_node", self.answer_generate_node)
        graph.add_node("synthesis_node", self.synthesis_node)
        graph.add_node("memory_write_node", self.memory_write_node)
        graph.add_node("gc_node", self.gc_node)

        graph.set_entry_point("prepare_query_node")
        graph.add_edge("prepare_query_node", "memory_read_node")
        graph.add_edge("memory_read_node", "retrieve_node")
        graph.add_edge("retrieve_node", "rerank_node")
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

    @traceable(name="memory_read_node")
    def memory_read_node(self, state: GraphState) -> GraphState:
        conversation_id = state["conversation_id"]
        return {
            "memory_summary": self.memory_store.get_summary(conversation_id),
            "memory_window": self.memory_store.get_window(conversation_id),
        }

    @traceable(name="retrieve_node")
    def retrieve_node(self, state: GraphState) -> GraphState:
        retrieval = self.rag.retrieve(
            state["normalized_query"],
            search_mode="vector",
            use_reranking=False,
            metadata_filter=state.get("metadata_filter"),
            top_k=24,
            candidate_pool=24,
        )
        return {
            "candidate_indices": retrieval.chunk_indices,
            "candidate_chunks": retrieval.chunks,
            "candidate_metadatas": retrieval.metadatas,
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
        memory_context = self._format_memory_context(
            state.get("memory_summary", ""),
            state.get("memory_window", []),
        )
        chunks_context = "\n\n---\n\n".join(state.get("final_chunks", []))
        prompt = (
            "Contexte conversationnel:\n"
            f"{memory_context}\n\n"
            "Extraits financiers récupérés:\n"
            f"{chunks_context}\n\n"
            f"Question utilisateur: {state['normalized_query']}\n\n"
            "Réponds en français avec une analyse financière structurée."
        )
        draft = self.rag.provider.generate(prompt, temperature=0.1, max_tokens=900)
        return {"draft_answer": draft}

    @traceable(name="synthesis_node")
    def synthesis_node(self, state: GraphState) -> GraphState:
        prompt = (
            "Synthétise et clarifie cette réponse d'analyse financière.\n"
            "Conserve uniquement les éléments actionnables et factuels, en 4 sections:\n"
            "1) Synthèse\n2) Signaux positifs\n3) Risques\n4) Conclusion\n\n"
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

    @traceable(name="finance_langgraph_run")
    def run(self, query: str, conversation_id: str | None = None) -> GraphState:
        convo_id = conversation_id or str(uuid.uuid4())
        initial_state: GraphState = {"conversation_id": convo_id, "query": query}
        return self.graph.invoke(initial_state)
