import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import html as html_module
import json
import os
import re

from bs4 import BeautifulSoup

from rag.paths import PROCESSED_DATA_DIR, raw_input_dirs, ensure_dir

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

SUPPORTED_EXTENSIONS = (".html", ".htm", ".pdf", ".txt", ".csv", ".json")

MIN_SECTION_CHARS = 500
MAX_8K_CHARS = 12_000
MAX_TABLE_ROWS = 12

# Limites par section pour rester dans le quota d'embeddings (narratif > tableaux).
SECTION_MAX_CHARS = {
    "Item_1A": 45_000,
    "Item_7": 55_000,
    "Item_8": 20_000,
}

DEFAULT_SECTIONS = ("Item_1A", "Item_7")

SECTION_SPECS = {
    "Item_1A": {
        "starts": [
            r"ITEM\s+1A\.?\s+RISK\s+FACTORS",
            r"Item\s+1A\.?\s+Risk\s+Factors",
            r"\|\s*Item\s*1A\.?\s*\|\s*Risk\s+Factors\s*\|\s*\n\|\s*---",
            r"\|\s*\|\s*Risk\s+Factors\s*\|\s*\d+\s*\|\s*\n\|\s*---",
        ],
        "ends": [
            r"ITEM\s+1B\.?",
            r"Item\s+1B\.?",
            r"\|\s*Item\s*1B\.?\s*\|",
            r"\|\s*\|\s*Other\s+Key\s+Information\s*\|",
            r"ITEM\s+2\.?\s+PROPERTIES",
            r"Item\s+2\.?\s+Properties",
        ],
    },
    "Item_7": {
        "starts": [
            r"ITEM\s+7\.?\s+MANAGEMENT[\'\u2019]?S\s+DISCUSSION",
            r"Item\s+7\.?\s+Management[\'\u2019]?s\s+Discussion",
            r"Management[\'\u2019]?s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition",
            r"\|\s*Item\s*7\.?\s*\|\s*Management[\'\u2019]?s\s+Discussion[^|]*\|\s*\n\|\s*---",
            r"Management[\'\u2019]?s\s+Discussion\s+and\s+Analysis\s*\|\s*\|\s*\n\|\s*---",
        ],
        "ends": [
            r"ITEM\s+7A\.?",
            r"Item\s+7A\.?",
            r"\|\s*Item\s*7A\.?\s*\|",
            r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk",
        ],
    },
    "Item_8": {
        "starts": [
            r"ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS",
            r"Item\s+8\.?\s+Financial\s+Statements",
            r"\|\s*Item\s*8\.?\s*\|\s*Financial\s+Statements[^|]*\|\s*\n\|\s*---",
            r"Financial\s+Statements\s+and\s+Supplementary\s+Data",
        ],
        "ends": [
            r"ITEM\s+9\.?\s+CHANGES",
            r"Item\s+9\.?\s+Changes",
            r"\|\s*Item\s*9\.?\s*\|",
            r"ITEM\s+9A\.?",
            r"Item\s+9A\.?",
        ],
    },
}


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


def cap_section_text(section_name: str, text: str) -> str:
    limit = SECTION_MAX_CHARS.get(section_name)
    if not limit or len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n\n" in cut:
        cut = cut.rsplit("\n\n", 1)[0]
    return cut.rstrip() + f"\n\n[... section tronquée à {limit:,} caractères pour le quota d'embedding ...]"


def parse_sections_arg(raw: str) -> tuple[str, ...]:
    mapping = {
        "1a": "Item_1A",
        "7": "Item_7",
        "8": "Item_8",
    }
    sections = []
    for part in raw.lower().replace(" ", "").split(","):
        if part in mapping:
            sections.append(mapping[part])
    return tuple(sections) if sections else DEFAULT_SECTIONS


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


def table_to_markdown(table) -> str:
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


def read_text_file(file_path: str) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode text file", file_path, 0, 1, "unknown")


def pdf_to_text(file_path: str) -> str:
    if pdfplumber is None:
        raise ImportError("pdfplumber is required for PDF parsing. pip install pdfplumber")

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
        lines.append(f"\n... ({len(rows) - 1 - max_rows} lignes de prix omises)")
    return "\n".join(lines)


def json_to_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
            for item in data[:500]:
                lines.append("| " + " | ".join(str(item.get(k, "")) for k in keys) + " |")
            if len(data) > 500:
                lines.append(f"\n... ({len(data) - 500} additional rows omitted)")
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


def extract_file_type(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")


def extract_year_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    patterns = [
        r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?(0[1-9]|[12]\d|3[01])",
        r"(20\d{2})",
        r"(19\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    return ""


def extract_ticker_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    match = re.match(r"^([a-z]{1,5})[-_]", stem)
    if match:
        return match.group(1).upper()
    return ""


def is_8k_filename(filename: str) -> bool:
    return bool(re.search(r"8[-_]?k", filename, re.I))


def is_10k_filename(filename: str) -> bool:
    return bool(re.search(r"10[-_]?k", filename, re.I))


def _extract_between(text: str, start_patterns: list[str], end_patterns: list[str]) -> str:
    best = ""
    for start_pat in start_patterns:
        for start_match in re.finditer(start_pat, text, re.IGNORECASE | re.DOTALL):
            start = start_match.start()
            search_from = start + MIN_SECTION_CHARS
            end = len(text)
            for end_pat in end_patterns:
                end_match = re.search(end_pat, text[search_from:], re.IGNORECASE | re.DOTALL)
                if end_match:
                    end = min(end, search_from + end_match.start())
            candidate = clean_text(text[start:end])
            if len(candidate) >= MIN_SECTION_CHARS and len(candidate) > len(best):
                best = candidate
    return best


def extract_sections(text: str, enabled: tuple[str, ...] = DEFAULT_SECTIONS) -> dict[str, str]:
    text = normalize_sec_text(text)
    sections = {}
    for section_name in enabled:
        spec = SECTION_SPECS.get(section_name)
        if not spec:
            continue
        content = _extract_between(text, spec["starts"], spec["ends"])
        if content:
            content = cap_section_text(section_name, content)
            sections[section_name] = content
    return sections


def extract_8k_excerpt(text: str) -> str:
    text = normalize_sec_text(text)
    triggers = [
        r"Item\s+2\.02[^\n]{0,120}Results\s+of\s+Operations",
        r"Results\s+of\s+Operations\s+and\s+Financial\s+Condition",
        r"Item\s+2\.02",
    ]
    for pat in triggers:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            return clean_text(text[match.start() : match.start() + MAX_8K_CHARS])
    return clean_text(text[:MAX_8K_CHARS])


def parse_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".html", ".htm"):
        content = read_text_file(file_path)
        return html_to_text(content)
    if ext == ".pdf":
        return pdf_to_text(file_path)
    if ext == ".txt":
        return clean_text(read_text_file(file_path))
    if ext == ".csv":
        return csv_to_text(file_path)
    if ext == ".json":
        return json_to_text(file_path)

    raise ValueError(f"Unsupported file extension: {ext}")


def output_filename(source_name: str, section: str) -> str:
    safe_source = re.sub(r"[^\w.\-]+", "_", source_name)
    return f"{safe_source}__{section}.txt"


def write_section_files(source_name: str, sections: dict[str, str]) -> int:
    """Write full section text; chunking for embeddings happens in hybrid_rag.py."""
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


def main():
    parser = argparse.ArgumentParser(description="Prétraiter les rapports SEC pour le RAG.")
    parser.add_argument(
        "--include-8k",
        action="store_true",
        help="Inclure les 8-K (extrait limité). Par défaut: ignorés.",
    )
    parser.add_argument(
        "--sections",
        default="1a,7",
        help="Sections SEC à extraire (défaut: 1a,7 — sans Item 8/tableaux). Ex: 1a,7,8",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Ignorer les rapports antérieurs à cette année (ex: 2023).",
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Inclure les CSV de prix (désactivé par défaut).",
    )
    args = parser.parse_args()
    enabled_sections = parse_sections_arg(args.sections)

    ensure_dir(PROCESSED_DATA_DIR)

    files = collect_input_files()
    input_dirs = ", ".join(str(d) for d in raw_input_dirs())
    print(f"Found {len(files)} files in: {input_dirs}")
    print(f"Sections actives: {', '.join(enabled_sections)}")
    if args.min_year:
        print(f"Filtre année: >= {args.min_year}")

    stats = {
        "processed": 0,
        "sections_written": 0,
        "skipped": 0,
        "skipped_8k": 0,
        "no_sections_10k": 0,
        "skipped_year": 0,
        "skipped_csv": 0,
    }

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        try:
            if args.min_year and is_10k_filename(filename):
                year = extract_year_from_filename(filename)
                if year and int(year) < args.min_year:
                    print(f"  Skip: année {year} < {args.min_year}.")
                    stats["skipped_year"] += 1
                    continue

            if is_8k_filename(filename) and not args.include_8k:
                print(f"  Skip: 8-K ignoré (utilisez --include-8k pour un extrait court).")
                stats["skipped_8k"] += 1
                continue

            text = parse_file(file_path)
            if not text:
                print(f"  Warning: empty content for {filename}")
                stats["skipped"] += 1
                continue

            ext = extract_file_type(filename)
            if ext in ("csv", "json"):
                if ext == "csv" and not args.include_csv:
                    print("  Skip: CSV prix (utilisez --include-csv).")
                    stats["skipped_csv"] += 1
                    continue
                section_name = (
                    "market_data"
                    if "prix" in filename.lower() or "price" in filename.lower()
                    else "structured_data"
                )
                out_path = PROCESSED_DATA_DIR / output_filename(filename, section_name)
                out_path.write_text(text, encoding="utf-8")
                stats["sections_written"] += 1
            elif is_8k_filename(filename):
                excerpt = extract_8k_excerpt(text)
                if len(excerpt) < 100:
                    print(f"  Skip: 8-K trop court après extraction.")
                    stats["skipped"] += 1
                    continue
                out_path = PROCESSED_DATA_DIR / output_filename(filename, "earnings_8k")
                out_path.write_text(excerpt, encoding="utf-8")
                stats["sections_written"] += 1
                print(f"  8-K → extrait earnings ({len(excerpt):,} car.)")
            else:
                sections = extract_sections(text, enabled=enabled_sections)
                if sections:
                    written = write_section_files(filename, sections)
                    stats["sections_written"] += written
                    detail = ", ".join(
                        f"{name} ({len(body):,} car.)" for name, body in sections.items()
                    )
                    print(f"  Sections: {detail}")
                    if written == 0:
                        print(f"  Warning: matched sections were too short for {filename}")
                        stats["skipped"] += 1
                elif is_10k_filename(filename):
                    print(f"  Skip: aucune section Item 1A/7/8 trouvée dans ce 10-K.")
                    stats["no_sections_10k"] += 1
                else:
                    print(f"  Skip: document sans sections SEC reconnues.")

            stats["processed"] += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            stats["skipped"] += 1

    print(
        f"\nDone. Processed={stats['processed']}, "
        f"sections_written={stats['sections_written']}, "
        f"skipped_8k={stats['skipped_8k']}, "
        f"skipped_year={stats['skipped_year']}, "
        f"skipped_csv={stats['skipped_csv']}, "
        f"no_sections_10k={stats['no_sections_10k']}, "
        f"skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
