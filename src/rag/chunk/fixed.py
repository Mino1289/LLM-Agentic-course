"""Découpage fixe et récursif."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_fixed(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks


def chunk_text_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n| ", "\n\n", "\n| ", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
