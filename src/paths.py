"""
Chemins absolus du projet — indépendants du répertoire de lancement.

Layout:
  {PROJECT_ROOT}/
    data/                  ← fichiers bruts (SEC, CSV, PDF…)
    src/                   ← code source
    ui/                    ← interface Streamlit
    tests/                 ← tests
    reports/               ← rapports générés
"""

from __future__ import annotations

import sys
from pathlib import Path

# Détecte la racine du projet depuis ce fichier (src/paths.py → racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ajoute la racine au sys.path pour les imports absolus
_root = str(PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

DATA_DIR = PROJECT_ROOT / "data"
LEGACY_TEMP_DIR = PROJECT_ROOT / "temp"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
SEC_FILINGS_METADATA = PROJECT_ROOT / "data" / "sec_filings_metadata.json"
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
