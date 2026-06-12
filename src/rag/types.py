"""Types communs pour le module RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ChunkStrategy = Literal["semantic"]

DEFAULT_MAX_CHUNK_SIZE = 1500
DEFAULT_MIN_CHUNK_SIZE = 80
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DAILY_EMBEDDING_LIMIT = 0
DEFAULT_EMBEDDING_RPM = 100
DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_EMBEDDING_RETRIES = 3


@dataclass
class EmbeddingPlan:
    chunk_strategy: str
    collection_name: str
    section_files: int
    total_chunks: int
    already_indexed: int
    missing_chunks: int
    daily_limit: int
    daily_used: int
    daily_remaining: int
    embeddable_now: int
    deferred_chunks: int
    estimated_minutes: float
    rpm_limit: int
    unlimited_quota: bool = False

    def summary(self) -> str:
        quota_line = (
            f"Quota journalier      : {self.daily_used:,} / {self.daily_limit:,} utilisés"
            if not self.unlimited_quota
            else "Quota journalier      : illimite (desactive)"
        )
        remaining_line = (
            f"Quota restant aujourd'hui : {self.daily_remaining:,}"
            if not self.unlimited_quota
            else "Quota restant aujourd'hui : illimite"
        )
        lines = [
            f"=== Plan d'embedding ({self.chunk_strategy}) ===",
            f"Collection ChromaDB : {self.collection_name}",
            f"Fichiers .txt source  : {self.section_files}",
            f"Chunks totaux         : {self.total_chunks:,}",
            f"Déjà indexés          : {self.already_indexed:,}",
            f"Manquants             : {self.missing_chunks:,}",
            "",
            quota_line,
            remaining_line,
            f"Embeddings prévus now : {self.embeddable_now:,}",
            f"Reportés (quota)      : {self.deferred_chunks:,}",
            f"Durée estimée         : ~{self.estimated_minutes:.1f} min "
            f"({self.rpm_limit} req/min)",
        ]
        if self.missing_chunks == 0:
            lines.append("\n✅ Rien à embedder — index vectoriel à jour.")
        elif not self.unlimited_quota and self.missing_chunks > self.daily_remaining:
            days = (self.deferred_chunks // max(self.daily_remaining, 1)) + (
                1 if self.deferred_chunks % max(self.daily_remaining, 1) else 0
            )
            lines.append(
                f"\n⚠️  Indexation complète : ~{days} jour(s) supplémentaire(s) au rythme actuel."
            )
        elif self.deferred_chunks > 0:
            lines.append(
                "\nℹ️ Des chunks restent reportés par la limite configurée pour ce run."
            )
        else:
            lines.append("\n✅ Tout peut être indexé aujourd'hui avec le quota restant.")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    chunks: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    search_mode: str = "vector_rerank"
    reranking_enabled: bool = False


def embedding_sleep_seconds(rpm_limit: int = DEFAULT_EMBEDDING_RPM) -> float:
    return 60.0 / max(rpm_limit, 1)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
