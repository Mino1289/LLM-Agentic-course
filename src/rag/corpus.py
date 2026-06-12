"""Construction du corpus depuis les fichiers prétraités."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.paths import PROCESSED_DATA_DIR
from src.rag.metadata.build import build_chunk_id, build_metadata
from src.rag.metadata.extract import parse_processed_filename
from src.rag.chunk.semantic import chunk_text_semantic


def build_corpus(
    chunk_strategy: str,
    max_files: Optional[int] = None,
) -> tuple[list[str], list[dict], list[str], int]:
    """Parcours les fichiers .txt prétraités et construit le corpus chunké."""
    files = sorted(PROCESSED_DATA_DIR.glob("*.txt"))
    if max_files:
        files = files[:max_files]

    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    all_ids: list[str] = []

    for file_path in files:
        source, section = parse_processed_filename(str(file_path))
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            continue

        chunker = chunk_text_semantic
        for i, chunk in enumerate(chunker(text)):
            chunk_id = build_chunk_id(source, section, chunk_strategy, i, chunk)
            all_chunks.append(chunk)
            all_metadata.append(build_metadata(source, section, i, chunk_strategy))
            all_ids.append(chunk_id)

    return all_chunks, all_metadata, all_ids, len(files)
