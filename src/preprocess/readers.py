"""Lecture et conversion vers texte brut des formats SEC (HTML, PDF, CSV, JSON)."""

from __future__ import annotations

import csv
import json
import os
import re

from bs4 import BeautifulSoup

from src.preprocess.clean import (
    clean_text,
    normalize_sec_text,
    compress_markdown_tables,
)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def read_text_file(file_path: str) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode text file", file_path, 0, 1, "unknown")


def table_to_markdown(table: Any) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [
            clean_text(cell.get_text(separator=" ", strip=True))
            for cell in tr.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    header = rows[0]
    separator = ["---"] * max_cols
    body = rows[1:] if len(rows) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def html_to_text(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for table in soup.find_all("table"):
        markdown = table_to_markdown(table)
        replacement = soup.new_string(f"\n\n{markdown}\n\n" if markdown else "\n")
        table.replace_with(replacement)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = clean_text(normalize_sec_text(text))
    return compress_markdown_tables(text)


def pdf_to_text(file_path: str) -> str:
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is required for PDF parsing. pip install pdfplumber"
        )
    parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            tables_md = []
            for table in page.extract_tables() or []:
                if not table:
                    continue
                rows = [[str(cell or "").strip() for cell in row] for row in table]
                if not rows:
                    continue
                max_cols = max(len(row) for row in rows)
                rows = [row + [""] * (max_cols - len(row)) for row in rows]
                header = rows[0]
                lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * max_cols) + " |",
                ]
                for row in rows[1:]:
                    lines.append("| " + " | ".join(row) + " |")
                tables_md.append("\n".join(lines))
            if page_text:
                parts.append(page_text)
            if tables_md:
                parts.append("\n\n".join(tables_md))
    return clean_text("\n\n".join(parts))


def csv_to_text(file_path: str, max_rows: int = 60) -> str:
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    body = rows[1 : max_rows + 1]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    if len(rows) - 1 > max_rows:
        lines.append(f"\n... ({len(rows) - 1 - max_rows} lignes omises)")
    return "\n".join(lines)


def json_to_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            lines = [
                "| " + " | ".join(keys) + " |",
                "| " + " | ".join(["---"] * len(keys)) + " |",
            ]
            for item in data[:500]:
                lines.append(
                    "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
                )
            if len(data) > 500:
                lines.append(f"\n... ({len(data) - 500} rows omitted)")
            return "\n".join(lines)
        return clean_text(json.dumps(data, ensure_ascii=False, indent=2))
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"## {key}\n{value}")
        return "\n\n".join(lines)
    return str(data)


def parse_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".html", ".htm"):
        return html_to_text(read_text_file(file_path))
    if ext == ".pdf":
        return pdf_to_text(file_path)
    if ext == ".txt":
        return clean_text(read_text_file(file_path))
    if ext == ".csv":
        return csv_to_text(file_path)
    if ext == ".json":
        return json_to_text(file_path)
    raise ValueError(f"Unsupported file extension: {ext}")
