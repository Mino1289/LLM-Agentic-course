"""Constantes et configuration du prétraitement SEC."""

SUPPORTED_EXTENSIONS = (".html", ".htm", ".pdf", ".txt", ".csv", ".json")

MIN_SECTION_CHARS = 500
MAX_8K_CHARS = 12_000
MAX_TABLE_ROWS = 12

SECTION_MAX_CHARS = {
    "Item_1A": None,
    "Item_7": None,
    "Item_8": None,
}

DEFAULT_SECTIONS = ("Item_1A", "Item_7")
