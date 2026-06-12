"""Module embeddings - Gestion des embeddings, backoff et cache."""
from src.embeddings.backoff import BackoffConfig, is_permanent_error, with_exponential_backoff
from src.embeddings.quota import QuotaState
from src.embeddings.cache import EmbeddingCache, build_embedding_cache_from_env

__all__ = [
    "BackoffConfig",
    "is_permanent_error",
    "with_exponential_backoff",
    "QuotaState",
    "EmbeddingCache",
    "build_embedding_cache_from_env",
]
