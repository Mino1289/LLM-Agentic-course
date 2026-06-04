"""
Chemins absolus du pipeline RAG — indépendants du répertoire de lancement.

Layout:
  {PROJECT_ROOT}/
    data/                  ← fichiers bruts (SEC, CSV, PDF…)
    rag/
      processed_data/      ← sections .txt après preprocess
      chroma_db/           ← index vectoriel
      sec_filings_metadata.json
"""
from __future__ import annotations

import sys
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent

# Permet `from rag.xxx import` depuis n'importe quel script du repo
_root = str(PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Données brutes (même niveau que rag/)
DATA_DIR = PROJECT_ROOT / "data"

# Ancien dossier temp/ à la racine (compatibilité)
LEGACY_TEMP_DIR = PROJECT_ROOT / "temp"

# Données prétraitées et index
PROCESSED_DATA_DIR = RAG_DIR / "processed_data"
CHROMA_DB_DIR = RAG_DIR / "chroma_db"
SEC_FILINGS_METADATA = RAG_DIR / "sec_filings_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
ENV_FILE = PROJECT_ROOT / ".env"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_input_dirs() -> list[Path]:
    """Répertoires à scanner pour le preprocess (data/ + temp/ legacy)."""
    dirs = [DATA_DIR]
    if LEGACY_TEMP_DIR.is_dir():
        dirs.append(LEGACY_TEMP_DIR)
    return dirs


def raw_input_glob(ext: str) -> list[Path]:
    """Liste tous les fichiers bruts pour une extension donnée."""
    pattern = f"*{ext}"
    found: list[Path] = []
    for directory in raw_input_dirs():
        found.extend(directory.glob(pattern))
    return sorted(set(found))
