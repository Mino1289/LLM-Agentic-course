"""Construction des métadonnées et identifiants de chunks."""
from __future__ import annotations

import hashlib
from typing import Any

from src.rag.metadata.extract import extract_file_type_from_source, extract_year_from_source, extract_ticker_from_source


def build_chunk_id(source: str, section: str, strategy: str, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
    return f"{source}__{section}__{strategy}__{chunk_index}__{digest}"


def build_metadata(source: str, section: str, chunk_index: int, chunk_strategy: str) -> dict[str, Any]:
    return {
        "source": source,
        "section": section,
        "chunk_index": chunk_index,
        "file_type": extract_file_type_from_source(source),
        "year": extract_year_from_source(source),
        "ticker": extract_ticker_from_source(source),
        "chunk_strategy": chunk_strategy,
    }
