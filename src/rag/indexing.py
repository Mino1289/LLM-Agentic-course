"""Indexation vectorielle et embedding des chunks."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.rag.types import (
    DEFAULT_DAILY_EMBEDDING_LIMIT,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_RETRIES,
    DEFAULT_EMBEDDING_RPM,
    EmbeddingPlan,
    embedding_sleep_seconds,
)
from src.rag.corpus import build_corpus
from src.embeddings.backoff import BackoffConfig, with_exponential_backoff
from src.embeddings.quota import QuotaState
from src.paths import DATA_DIR

DEFAULT_QUOTA_STATE_PATH = DATA_DIR / "embedding_quota_state.json"


def iter_batches(items: list[int], batch_size: int):
    size = max(1, batch_size)
    for start in range(0, len(items), size):
        yield items[start: start + size]


def remove_stale_index_entries(collection: Any, valid_ids: list[str]) -> int:
    existing_ids = set(collection.get()["ids"])
    stale_ids = sorted(existing_ids - set(valid_ids))
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(stale_ids)


def get_embedding_plan(
    chunk_strategy: str,
    collection,
    corpus_fn,
    max_files: Optional[int] = None,
    daily_quota_used: int = 0,
    daily_quota_limit: int = DEFAULT_DAILY_EMBEDDING_LIMIT,
    max_new_embeddings: Optional[int] = None,
    rpm_limit: int = DEFAULT_EMBEDDING_RPM,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> EmbeddingPlan:
    all_chunks, _, all_ids, section_files = corpus_fn(max_files=max_files)
    total_chunks = len(all_chunks)

    existing_ids = set(collection.get()["ids"])
    missing_count = sum(1 for cid in all_ids if cid not in existing_ids)
    already_indexed = total_chunks - missing_count

    unlimited_quota = daily_quota_limit <= 0
    if unlimited_quota:
        daily_remaining = missing_count
        budget = missing_count if max_new_embeddings is None else max_new_embeddings
        budget = max(0, budget)
    else:
        daily_remaining = max(0, daily_quota_limit - daily_quota_used)
        budget = daily_remaining if max_new_embeddings is None else max_new_embeddings
        budget = max(0, min(budget, daily_remaining))
    embeddable_now = min(missing_count, budget)
    deferred = max(0, missing_count - embeddable_now)

    estimated_requests = embeddable_now / max(1, embedding_batch_size)
    return EmbeddingPlan(
        chunk_strategy=chunk_strategy,
        collection_name=collection.name,
        section_files=section_files,
        total_chunks=total_chunks,
        already_indexed=already_indexed,
        missing_chunks=missing_count,
        daily_limit=daily_quota_limit,
        daily_used=daily_quota_used,
        daily_remaining=daily_remaining,
        embeddable_now=embeddable_now,
        deferred_chunks=deferred,
        estimated_minutes=estimated_requests / max(rpm_limit, 1),
        rpm_limit=rpm_limit,
        unlimited_quota=unlimited_quota,
    )


def load_and_index_data(
    provider,
    chunk_strategy: str,
    collection,
    max_files: Optional[int] = None,
    max_new_embeddings: Optional[int] = None,
    daily_quota_used: Optional[int] = None,
    daily_quota_limit: int = DEFAULT_DAILY_EMBEDDING_LIMIT,
    rpm_limit: int = DEFAULT_EMBEDDING_RPM,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_embedding_retries: int = DEFAULT_EMBEDDING_RETRIES,
    quota_state_path: Optional[Path] = None,
    backoff_config: Optional[BackoffConfig] = None,
    dry_run: bool = False,
) -> tuple[EmbeddingPlan, list[str], list[dict], list[str]]:
    """Charge et indexe les données dans ChromaDB."""
    print(f"Loading and indexing data (chunk_strategy={chunk_strategy})...")

    all_chunks, all_metadata, all_ids, _ = build_corpus(chunk_strategy=chunk_strategy, max_files=max_files)

    state_path = quota_state_path or DEFAULT_QUOTA_STATE_PATH
    quota_state = QuotaState(state_path)
    if daily_quota_used is None:
        daily_quota_used = quota_state.quota_used()
        print(f"♻️  Reprise quota depuis l'état persisté : {daily_quota_used:,}")

    if not dry_run:
        removed = remove_stale_index_entries(collection, all_ids)
        if removed:
            print(f"Removed {removed} stale chunk(s) from ChromaDB.")

    plan = get_embedding_plan(
        chunk_strategy=chunk_strategy,
        collection=collection,
        corpus_fn=lambda **_: build_corpus(chunk_strategy=chunk_strategy, max_files=max_files),
        max_files=max_files,
        daily_quota_used=daily_quota_used,
        daily_quota_limit=daily_quota_limit,
        max_new_embeddings=max_new_embeddings,
        rpm_limit=rpm_limit,
        embedding_batch_size=embedding_batch_size,
    )
    print(plan.summary())

    if not all_chunks:
        print("No documents to index.")
        return plan, [], [], []

    if dry_run:
        print("\nMode --plan : aucun embedding lancé.")
        return plan, all_chunks, all_metadata, all_ids

    if plan.missing_chunks == 0:
        print("All chunks are already indexed in ChromaDB.")
        return plan, all_chunks, all_metadata, all_ids

    if plan.embeddable_now == 0:
        print("\nℹ️ Aucun embedding prévu avec la configuration actuelle.")
        return plan, all_chunks, all_metadata, all_ids

    existing_ids = set(collection.get()["ids"])
    missing_indices = [i for i, cid in enumerate(all_ids) if cid not in existing_ids]
    indices_to_embed = missing_indices[: plan.embeddable_now]

    print(f"\nEmbedding {len(indices_to_embed)} / {plan.missing_chunks} chunks manquants...")

    sleep_sec = embedding_sleep_seconds(rpm_limit)
    config = backoff_config or BackoffConfig(max_retries=max_embedding_retries)
    embedded = 0
    last_error: Optional[str] = None
    for batch_num, batch_indices in enumerate(
        iter_batches(indices_to_embed, embedding_batch_size), start=1
    ):
        texts = [all_chunks[idx] for idx in batch_indices]
        metadatas = [all_metadata[idx] for idx in batch_indices]
        ids = [all_ids[idx] for idx in batch_indices]

        try:
            embeddings = with_exponential_backoff(
                lambda t=texts: provider.embed(t),
                config=config,
            )
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding response size mismatch: {len(embeddings)} vs {len(texts)}"
                )
        except Exception as e:
            last_error = f"batch {batch_num}: {e}"
            print(f"❌ Batch {batch_num} abandonné : {e}")
            quota_state.update(batch_size=0, last_error=last_error)
            break

        try:
            with_exponential_backoff(
                lambda t=texts, e=embeddings, m=metadatas, i=ids: collection.upsert(
                    documents=t, embeddings=e, metadatas=m, ids=i
                ),
                config=config,
            )
        except Exception as e:
            last_error = f"batch {batch_num} upsert: {e}"
            print(f"❌ Écriture ChromaDB batch {batch_num} échouée : {e}")
            quota_state.update(batch_size=0, last_error=last_error)
            break

        embedded += len(batch_indices)
        quota_state.update(batch_size=len(batch_indices), last_error=None)
        if batch_num % 5 == 0 or embedded == len(indices_to_embed):
            print(f"  Progression : {embedded}/{len(indices_to_embed)} (quota-used={quota_state.quota_used():,})")
        time.sleep(sleep_sec)

    print(f"Vector indexing complete. Total in DB: {collection.count()}")
    if plan.deferred_chunks > 0:
        print(f"⏳ {plan.deferred_chunks} chunks restants — l'état quota est persisté dans {state_path}.")
    return plan, all_chunks, all_metadata, all_ids
