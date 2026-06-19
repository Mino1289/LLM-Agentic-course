"""Découpage sémantique des textes — chunking par paragraphes + phrases."""

from __future__ import annotations

import re
from typing import Optional

from src.rag.types import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_MIN_CHUNK_SIZE


def split_semantic_blocks(text: str) -> list[str]:
    """Split into paragraphs and markdown tables without breaking table rows."""
    blocks: list[str] = []
    current_lines: list[str] = []
    in_table = False

    for line in text.split("\n"):
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if current_lines and not in_table:
                block = "\n".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
            in_table = True
            current_lines.append(line)
            continue

        if in_table:
            block = "\n".join(current_lines).strip()
            if block:
                blocks.append(block)
            current_lines = []
            in_table = False

        if not stripped:
            if current_lines:
                block = "\n".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        block = "\n".join(current_lines).strip()
        if block:
            blocks.append(block)

    if not blocks and text.strip():
        blocks = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]

    return blocks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])', text)
    return [part.strip() for part in parts if part.strip()]


def split_markdown_table(table: str, max_size: int) -> list[str]:
    lines = [line for line in table.split("\n") if line.strip()]
    if len(lines) <= 2:
        return [table]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]
    prefix = f"{header}\n{separator}"
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = prefix + "\n" + "\n".join(candidate_rows)
        if len(candidate) > max_size and current_rows:
            chunks.append(prefix + "\n" + "\n".join(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append(prefix + "\n" + "\n".join(current_rows))

    return chunks or [table]


def split_block_by_sentences(block: str, max_size: int) -> list[str]:
    is_table = block.split("\n", 1)[0].strip().startswith("|")
    if is_table:
        return split_markdown_table(block, max_size)

    sentences = split_sentences(block)
    if len(sentences) <= 1:
        return [
            block[start : start + max_size] for start in range(0, len(block), max_size)
        ]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_size:
                current = sentence
            else:
                chunks.extend(
                    sentence[start : start + max_size]
                    for start in range(0, len(sentence), max_size)
                )
                current = ""
    if current:
        chunks.append(current)
    return chunks


def merge_semantic_blocks(
    blocks: list[str],
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > max_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_block_by_sentences(block, max_size))
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk) < min_size
            and len(merged[-1]) + 2 + len(chunk) <= max_size
        ):
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    return [chunk for chunk in merged if chunk.strip()]


def chunk_text_semantic(
    text: str,
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[str]:
    """Chunk by paragraphs/tables first, then sentences for oversized blocks."""
    if not text.strip():
        return []
    blocks = split_semantic_blocks(text)
    return merge_semantic_blocks(blocks, max_size=max_size, min_size=min_size)
