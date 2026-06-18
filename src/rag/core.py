"""Classe principale HybridRAG — point d'entrée pour la RAG financière."""

from __future__ import annotations

from typing import Generator, Optional

import chromadb

from src.config import TRACKED_TICKERS
from src.llm.provider import LLMProvider
from src.paths import CHROMA_DB_DIR
from src.rag.types import (
    DEFAULT_RERANKER,
    RetrievalResult,
    ChunkStrategy,
)
from src.rag.search.vector import (
    vector_search,
    apply_metadata_filter,
    deduplicate_indices,
)
from src.rag.search.rerank import rerank
from src.rag.corpus import build_corpus
from src.rag.indexing import load_and_index_data, get_embedding_plan


class HybridRAG:
    """Gestionnaire RAG hybride : indexation vectorielle, recherche et re-ranking."""

    def __init__(
        self,
        chunk_strategy: ChunkStrategy = "semantic",
        search_mode: str = "vector",
        use_reranking: bool = True,
        reranker_model: str = DEFAULT_RERANKER,
        collection_name: Optional[str] = None,
        max_candidate_cap: int = 30,
    ):
        self.chunk_strategy = chunk_strategy
        self.search_mode = search_mode
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model
        self.max_candidate_cap = max_candidate_cap

        self.documents: list[str] = []
        self.doc_metadata: list[dict] = []
        self.chunk_ids: list[str] = []
        self._reranker = None

        self.provider = LLMProvider()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        resolved_collection = collection_name or "finance_rag_semantic_vector"
        self.collection = self.chroma_client.get_or_create_collection(
            name=resolved_collection
        )

    def get_embedding_plan(self, **kwargs):
        return get_embedding_plan(
            chunk_strategy=self.chunk_strategy,
            collection=self.collection,
            corpus_fn=lambda **_: build_corpus(
                chunk_strategy=self.chunk_strategy,
                **(
                    {"max_files": kwargs.get("max_files")}
                    if "max_files" in kwargs
                    else {}
                ),
            ),
            **kwargs,
        )

    def load_and_index_data(
        self,
        max_files: Optional[int] = None,
        max_new_embeddings: Optional[int] = None,
        **kwargs,
    ):
        plan, all_chunks, all_metadata, all_ids = load_and_index_data(
            provider=self.provider,
            chunk_strategy=self.chunk_strategy,
            collection=self.collection,
            max_files=max_files,
            max_new_embeddings=max_new_embeddings,
            **kwargs,
        )
        if all_chunks:
            self.documents = all_chunks
            self.doc_metadata = all_metadata
            self.chunk_ids = all_ids
        return plan

    def retrieve(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
        top_k: int = 10,
        candidate_pool: int = 50,
    ) -> RetrievalResult:
        import time

        rerank_enabled = self.use_reranking if use_reranking is None else use_reranking

        start = time.perf_counter()
        candidate_indices = vector_search(
            self.provider,
            self.collection,
            self.chunk_ids,
            self.doc_metadata,
            query,
            top_k=min(candidate_pool, self.max_candidate_cap),
            metadata_filter=metadata_filter,
        )

        candidate_indices = deduplicate_indices(candidate_indices)
        candidate_indices = apply_metadata_filter(
            candidate_indices, metadata_filter, self.doc_metadata
        )

        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        rerank_start = time.perf_counter()
        if rerank_enabled and candidate_indices:
            final_indices = rerank(
                query,
                candidate_indices,
                self.documents,
                self.reranker_model,
                top_k=top_k,
            )
        else:
            final_indices = candidate_indices[:top_k]

        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

        return RetrievalResult(
            chunks=[self.documents[i] for i in final_indices],
            metadatas=[self.doc_metadata[i] for i in final_indices],
            chunk_indices=final_indices,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            search_mode="vector_rerank",
            reranking_enabled=rerank_enabled,
        )

    def _deduplicate_indices(self, indices: list[int]) -> list[int]:
        return deduplicate_indices(indices)

    def _rerank(self, query: str, indices: list[int], top_k: int = 10) -> list[int]:
        return rerank(query, indices, self.documents, self.reranker_model, top_k=top_k)

    def count_context_tokens(self, chunks: list[str]) -> int:
        context = "\n\n---\n\n".join(chunks)
        return self.provider.estimate_tokens(context)

    def build_prompt(self, query: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return f"""Tu es un analyste buy-side spécialisé actions.
En te basant uniquement sur les extraits suivants, fais une synthèse financière structurée.

Extraits:
{context}

Question: {query}

Format attendu:
1) Résumé exécutif (3-5 lignes)
2) Signaux positifs
3) Signaux de risque
4) Conclusion orientée décision (surveiller/renforcer/réduire) avec justification
"""

    def answer_question(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[str, RetrievalResult]:
        retrieval = self.retrieve(
            query,
            search_mode=search_mode,
            use_reranking=use_reranking,
            metadata_filter=metadata_filter,
        )
        prompt = self.build_prompt(query, retrieval.chunks)
        response = self.provider.generate(prompt)
        return response, retrieval

    def answer_question_stream(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
    ) -> Generator[str, None, RetrievalResult]:
        retrieval = self.retrieve(
            query,
            search_mode=search_mode,
            use_reranking=use_reranking,
            metadata_filter=metadata_filter,
        )
        prompt = self.build_prompt(query, retrieval.chunks)
        for chunk in self.provider.generate_stream(prompt):
            yield chunk
        return retrieval
