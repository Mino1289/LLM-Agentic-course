"""Persistent cache for query embeddings."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Iterable

from src.paths import DATA_DIR

DEFAULT_CACHE_FILE = DATA_DIR / "embedding_query_cache.json"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 1 week
DEFAULT_MAX_ENTRIES = 5000


class EmbeddingCache:
    """Persistent JSON cache for query embeddings."""

    def __init__(
        self,
        cache_file: Path | str = DEFAULT_CACHE_FILE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.cache_file = Path(cache_file)
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._cache: dict[str, dict] = self._load()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, dict]:
        if not self.cache_file.exists():
            return {}
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        now = time.time()
        return {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and v.get("ts", 0) + self.ttl_seconds > now
        }

    def _save(self) -> None:
        if len(self._cache) > self.max_entries:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda item: item[1].get("ts", 0),
                reverse=True,
            )
            self._cache = dict(sorted_entries[: self.max_entries])
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache_file)

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.get("ts", 0) + self.ttl_seconds < time.time():
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return list(entry.get("vector", []))

    def put(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        with self._lock:
            self._cache[key] = {"ts": time.time(), "vector": list(vector)}
            self._save()

    def get_many(self, texts: Iterable[str]) -> list[list[float] | None]:
        return [self.get(text) for text in texts]

    def put_many(self, pairs: list[tuple[str, list[float]]]) -> None:
        with self._lock:
            now = time.time()
            for text, vector in pairs:
                key = self._key(text)
                self._cache[key] = {"ts": now, "vector": list(vector)}
            self._save()

    def stats(self) -> dict[str, int | str]:
        return {
            "size": len(self._cache),
            "max": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "file": str(self.cache_file),
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()


def build_embedding_cache_from_env() -> EmbeddingCache:
    """Build an EmbeddingCache from environment variables."""
    cache_file = Path(os.getenv("EMBEDDING_CACHE_FILE", str(DEFAULT_CACHE_FILE)))
    try:
        ttl = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_SECONDS
    try:
        max_entries = int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES)))
    except (TypeError, ValueError):
        max_entries = DEFAULT_MAX_ENTRIES
    if os.getenv("EMBEDDING_CACHE_DISABLED", "0") == "1":
        cache_file = Path(tempfile.mkdtemp(prefix="embedding_cache_disabled_")) / "cache.json"
    return EmbeddingCache(cache_file, ttl_seconds=ttl, max_entries=max_entries)
