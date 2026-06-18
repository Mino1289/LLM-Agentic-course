"""Entrées/sorties pour le prétraitement SEC."""

from __future__ import annotations

import os
import re

from src.paths import PROCESSED_DATA_DIR, raw_input_dirs, ensure_dir
from src.preprocess.config import SUPPORTED_EXTENSIONS


def output_filename(source_name: str, section: str) -> str:
    safe_source = re.sub(r"[^\w.\-]+", "_", source_name)
    return f"{safe_source}__{section}.txt"


def write_section_files(source_name: str, sections: dict[str, str]) -> int:
    written = 0
    for section_name, section_text in sections.items():
        if not section_text or len(section_text) < 100:
            continue
        out_path = PROCESSED_DATA_DIR / output_filename(source_name, section_name)
        out_path.write_text(section_text, encoding="utf-8")
        written += 1
    return written


def collect_input_files() -> list[str]:
    files: list[str] = []
    for directory in raw_input_dirs():
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(str(p) for p in directory.glob(f"*{ext}"))
    return sorted(set(files))


def clean_processed_output() -> int:
    removed = 0
    for out_path in PROCESSED_DATA_DIR.glob("*.txt"):
        out_path.unlink()
        removed += 1
    return removed
