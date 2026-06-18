"""Extraction de métadonnées depuis les noms de fichiers et sources.
Réutilise les fonctions de classification depuis src/preprocess/classify.
"""

from __future__ import annotations

import os
import re

from src.preprocess.classify import (
    extract_year_from_filename as extract_year_from_source,
)
from src.preprocess.classify import (
    extract_ticker_from_filename as extract_ticker_from_source,
)


def parse_processed_filename(file_path: str) -> tuple[str, str]:
    basename = os.path.basename(file_path)
    stem = basename[:-4] if basename.endswith(".txt") else basename
    if "__" in stem:
        source, section = stem.rsplit("__", 1)
        return source, section
    return stem, "unknown"


def extract_file_type_from_source(source: str) -> str:
    lower = source.lower()
    if re.search(r"20[-_]?f", lower):
        return "20-F"
    if re.search(r"6[-_]?k", lower):
        return "6-K"
    if re.search(r"10[-_]?k", lower):
        return "10-K"
    if re.search(r"10[-_]?q", lower):
        return "10-Q"
    if re.search(r"8[-_]?k", lower):
        return "8-K"
    if "earnings_call" in lower:
        return "EARNINGS_CALL"
    ext = os.path.splitext(source)[1].lower().lstrip(".")
    return ext or "txt"
