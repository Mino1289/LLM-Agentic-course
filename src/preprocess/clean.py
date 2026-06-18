"""Nettoyage et normalisation de texte SEC."""

from __future__ import annotations

import html as html_module
import re

from src.preprocess.config import MAX_TABLE_ROWS, SECTION_MAX_CHARS


def normalize_sec_text(text: str) -> str:
    text = html_module.unescape(text)
    text = (
        text.replace("\u00a0", " ")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cap_section_text(section_name: str, text: str) -> str:
    limit = SECTION_MAX_CHARS.get(section_name)
    if not limit or len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n\n" in cut:
        cut = cut.rsplit("\n\n", 1)[0]
    return (
        cut.rstrip()
        + f"\n\n[... section tronquée à {limit:,} caractères pour le quota d'embedding ...]"
    )


def compress_markdown_tables(text: str, max_rows: int = MAX_TABLE_ROWS) -> str:
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"\|\s*[-:| ]+\|", lines[i + 1])
        ):
            table_lines = [line, lines[i + 1]]
            i += 2
            body_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body_rows.append(lines[i])
                i += 1
            if len(body_rows) > max_rows:
                omitted = len(body_rows) - max_rows
                table_lines.extend(body_rows[:max_rows])
                table_lines.append(f"| ... | ({omitted} lignes de tableau omises) |")
            else:
                table_lines.extend(body_rows)
            result.extend(table_lines)
        else:
            result.append(line)
            i += 1
    return "\n".join(result)
