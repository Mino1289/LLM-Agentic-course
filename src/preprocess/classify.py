"""Classification des fichiers par type SEC et extraction de métadonnées."""

from __future__ import annotations

import os
import re


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


def is_in_year_range(filename: str, min_year: int | None, max_year: int | None) -> bool:
    if min_year is None and max_year is None:
        return True
    raw_year = extract_year_from_filename(filename)
    if not raw_year:
        return False
    year = int(raw_year)
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def is_8k_filename(filename: str) -> bool:
    return bool(re.search(r"8[-_]?k", filename, re.I))


def is_10k_filename(filename: str) -> bool:
    return bool(re.search(r"10[-_]?k", filename, re.I))


def is_10q_filename(filename: str) -> bool:
    return bool(re.search(r"10[-_]?q", filename, re.I))


def is_20f_filename(filename: str) -> bool:
    return bool(re.search(r"20[-_]?f", filename, re.I))


def is_6k_filename(filename: str) -> bool:
    return bool(re.search(r"6[-_]?k", filename, re.I))


def is_earnings_call_filename(filename: str) -> bool:
    stem = os.path.splitext(filename)[0].lower()
    patterns = [
        r"earnings[_\- ]?call",
        r"conference[_\- ]?call",
        r"transcript",
        r"results[_\- ]?call",
    ]
    return any(re.search(pattern, stem, re.I) for pattern in patterns)
