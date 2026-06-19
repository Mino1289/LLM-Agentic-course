"""Re-ranking des résultats de recherche avec Cross-Encoder."""

from __future__ import annotations

from typing import Optional


def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def rerank(
    query: str,
    indices: list[int],
    documents: list[str],
    reranker_model: str,
    top_k: int = 10,
) -> list[int]:
    """Re-rank les indices des documents avec un Cross-Encoder."""
    if not indices:
        return []
    try:
        reranker = _get_reranker(reranker_model)
        pairs = [(query, documents[idx]) for idx in indices]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in ranked[:top_k]]
    except Exception as e:
        print(f"Reranker warning/error: {e}")
        return indices[:top_k]
