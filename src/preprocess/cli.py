"""CLI pour le prétraitement des rapports SEC."""

from __future__ import annotations

import argparse
import os

from src.paths import PROCESSED_DATA_DIR, ensure_dir
from src.preprocess.io import (
    clean_processed_output,
    collect_input_files,
    write_section_files,
    output_filename,
)
from src.preprocess.readers import parse_file
from src.preprocess.classify import (
    is_8k_filename,
    is_10k_filename,
    is_10q_filename,
    is_20f_filename,
    is_6k_filename,
    is_earnings_call_filename,
    extract_file_type,
    is_in_year_range,
    extract_year_from_filename,
)
from src.preprocess.sections import (
    parse_sections_arg,
    extract_sections,
    extract_8k_excerpt,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prétraiter les rapports SEC pour le RAG."
    )
    parser.add_argument(
        "--exclude-8k", action="store_true", help="Exclure les 8-K (item 2.02)."
    )
    parser.add_argument(
        "--sections", default="1a,7", help="Sections SEC (défaut: 1a,7). Ex: 1a,7,8"
    )
    parser.add_argument(
        "--min-year", type=int, default=None, help="Ignorer les rapports antérieurs."
    )
    parser.add_argument(
        "--max-year", type=int, default=None, help="Ignorer les rapports postérieurs."
    )
    parser.add_argument(
        "--no-clean-output", action="store_true", help="Conserver les anciens .txt."
    )
    parser.add_argument(
        "--include-csv", action="store_true", help="Inclure les CSV de prix."
    )
    args = parser.parse_args()

    if (
        args.min_year is not None
        and args.max_year is not None
        and args.min_year > args.max_year
    ):
        parser.error("--min-year doit être inférieur ou égal à --max-year")

    enabled_sections = parse_sections_arg(args.sections)
    ensure_dir(PROCESSED_DATA_DIR)

    if not args.no_clean_output:
        removed = clean_processed_output()
        print(f"Nettoyage sorties preprocess: {removed} fichier(s) supprimé(s).")

    files = collect_input_files()
    stats = {
        "processed": 0,
        "sections_written": 0,
        "skipped": 0,
        "skipped_8k": 0,
        "earnings_calls_written": 0,
        "no_sections_10k": 0,
        "skipped_year": 0,
        "skipped_unknown_year": 0,
        "skipped_csv": 0,
    }

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        try:
            if not is_in_year_range(filename, args.min_year, args.max_year):
                year = extract_year_from_filename(filename)
                if year:
                    print(f"  Skip: année {year} hors plage.")
                    stats["skipped_year"] += 1
                else:
                    print("  Skip: année absente.")
                    stats["skipped_unknown_year"] += 1
                continue

            if is_8k_filename(filename) and args.exclude_8k:
                print("  Skip: 8-K ignoré.")
                stats["skipped_8k"] += 1
                continue

            text = parse_file(file_path)
            if not text:
                print("  Warning: empty content.")
                stats["skipped"] += 1
                continue

            ext = extract_file_type(filename)
            if ext in ("csv", "json"):
                if ext == "csv" and not args.include_csv:
                    print("  Skip: CSV prix (--include-csv).")
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
                    print("  Skip: 8-K trop court.")
                    stats["skipped"] += 1
                    continue
                out_path = PROCESSED_DATA_DIR / output_filename(filename, "earnings_8k")
                out_path.write_text(excerpt, encoding="utf-8")
                stats["sections_written"] += 1
            elif ext == "txt" and is_earnings_call_filename(filename):
                if len(text) < 200:
                    print("  Skip: transcript trop court.")
                    stats["skipped"] += 1
                    continue
                out_path = PROCESSED_DATA_DIR / output_filename(
                    filename, "earnings_call"
                )
                out_path.write_text(text, encoding="utf-8")
                stats["sections_written"] += 1
                stats["earnings_calls_written"] += 1
            else:
                sections = extract_sections(text, enabled=enabled_sections)
                if sections:
                    written = write_section_files(filename, sections)
                    stats["sections_written"] += written
                elif is_10k_filename(filename):
                    stats["no_sections_10k"] += 1
                elif is_10q_filename(filename):
                    out_path = PROCESSED_DATA_DIR / output_filename(
                        filename, "quarterly_report"
                    )
                    out_path.write_text(text, encoding="utf-8")
                    stats["sections_written"] += 1
                elif is_20f_filename(filename):
                    out_path = PROCESSED_DATA_DIR / output_filename(
                        filename, "foreign_annual_report"
                    )
                    out_path.write_text(text, encoding="utf-8")
                    stats["sections_written"] += 1
                elif is_6k_filename(filename):
                    out_path = PROCESSED_DATA_DIR / output_filename(
                        filename, "foreign_interim_report"
                    )
                    out_path.write_text(text, encoding="utf-8")
                    stats["sections_written"] += 1
                else:
                    print("  Skip: document non reconnu.")
            stats["processed"] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            stats["skipped"] += 1

    print(
        f"\nDone. Processed={stats['processed']}, sections_written={stats['sections_written']}, "
        f"skipped_8k={stats['skipped_8k']}, earnings_calls={stats['earnings_calls_written']}, "
        f"skipped_year={stats['skipped_year']}, skipped_csv={stats['skipped_csv']}, "
        f"no_sections_10k={stats['no_sections_10k']}, skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
