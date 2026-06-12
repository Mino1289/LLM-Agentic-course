"""Extraction des sections réglementaires SEC (Item 1A, 7, 8)."""
from __future__ import annotations

import re

from src.preprocess.clean import normalize_sec_text, clean_text, cap_section_text
from src.preprocess.config import MIN_SECTION_CHARS, MAX_8K_CHARS, DEFAULT_SECTIONS

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


def _extract_between(text: str, start_patterns: list[str], end_patterns: list[str]) -> str:
    start_positions = sorted(
        {
            match.start()
            for pattern in start_patterns
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        }
    )
    distinct_starts: list[int] = []
    for position in start_positions:
        if not distinct_starts or position - distinct_starts[-1] > 200:
            distinct_starts.append(position)

    best = ""
    for start in distinct_starts:
        search_from = start + MIN_SECTION_CHARS
        end = len(text)
        for end_pat in end_patterns:
            end_match = re.search(end_pat, text[search_from:], re.IGNORECASE | re.DOTALL)
            if end_match:
                end = min(end, search_from + end_match.start())

        if any(start < later_start < end for later_start in distinct_starts):
            continue

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
