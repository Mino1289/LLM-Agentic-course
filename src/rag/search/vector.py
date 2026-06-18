"""Recherche vectorielle dans ChromaDB."""

from __future__ import annotations

from typing import Optional


def vector_search(
    provider,
    collection,
    chunk_ids: list[str],
    doc_metadata: list[dict],
    query: str,
    top_k: int = 10,
    metadata_filter: Optional[dict] = None,
) -> list[int]:
    """Recherche vectorielle avec filtres metadata optionnels."""
    try:
        query_embedding = provider.embed([query])[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if metadata_filter:
            filters = [{key: value} for key, value in metadata_filter.items()]
            query_kwargs["where"] = (
                filters[0] if len(filters) == 1 else {"$and": filters}
            )

        results = collection.query(**query_kwargs)
        if not results["ids"] or not results["ids"][0]:
            return []

        returned_ids = results["ids"][0]
        id_to_index = {cid: i for i, cid in enumerate(chunk_ids)}
        indices = [id_to_index[cid] for cid in returned_ids if cid in id_to_index]
        return indices
    except Exception as e:
        print(f"Vector search warning/error: {e}")
        return []


def apply_metadata_filter(
    indices: list[int], metadata_filter: Optional[dict], doc_metadata: list[dict]
) -> list[int]:
    if not metadata_filter:
        return indices
    filtered = []
    for idx in indices:
        metadata = doc_metadata[idx]
        if all(metadata.get(k) == v for k, v in metadata_filter.items()):
            filtered.append(idx)
    return filtered


def deduplicate_indices(indices: list[int]) -> list[int]:
    seen = set()
    deduped = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped
