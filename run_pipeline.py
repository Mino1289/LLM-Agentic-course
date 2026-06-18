#!/usr/bin/env python3
"""Pipeline RAG complète : download SEC → preprocess → chunk → embed → index.

Usage:
    python run_pipeline.py                        # tout (sauf download si fichiers déjà présents)
    python run_pipeline.py --download              # force le téléchargement SEC
    python run_pipeline.py --min-year 2023         # année min (défaut: 2024)
    python run_pipeline.py --max-year 2026         # année max (défaut: 2026)
    python run_pipeline.py --sections 1a,7,8       # sections à extraire
    python run_pipeline.py --strategy semantic     # stratégie de chunking
    python run_pipeline.py --batch-size 32         # taille de batch embedding
    python run_pipeline.py --rpm 120               # rate limit embeddings
    python run_pipeline.py --dry-run               # affiche le plan sans exécuter
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import ENV_FILE, load_project_env


def run(cmd: list[str], description: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n  ERREUR: {description} a échoué (code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline RAG complète")
    parser.add_argument(
        "--download", action="store_true", help="Forcer le téléchargement SEC"
    )
    parser.add_argument("--min-year", default="2024", help="Année min (défaut: 2024)")
    parser.add_argument("--max-year", default="2026", help="Année max (défaut: 2026)")
    parser.add_argument(
        "--sections", default="1a,7", help="Sections à extraire (défaut: 1a,7)"
    )
    parser.add_argument(
        "--strategy",
        default="semantic",
        help="Stratégie de chunking (défaut: semantic)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Taille de batch embedding"
    )
    parser.add_argument(
        "--rpm", type=int, default=120, help="Rate limit embeddings (req/min)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Affiche le plan sans exécuter"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Ignorer le téléchargement même si --download",
    )
    args = parser.parse_args()

    load_project_env()

    data_dir = PROJECT_ROOT / "data"
    htm_files = sorted(data_dir.glob("*.htm")) if data_dir.is_dir() else []

    if args.download or not htm_files:
        if args.skip_download:
            print("  Téléchargement ignoré (--skip-download)")
        else:
            run(
                [
                    sys.executable,
                    "-m",
                    "src.fetchers.download_SEC_reports",
                    "--min-year",
                    args.min_year,
                    "--max-year",
                    args.max_year,
                ],
                "Étape 1/4 — Téléchargement des rapports SEC",
            )

    run(
        [
            sys.executable,
            "-m",
            "src.preprocess.cli",
            "--sections",
            args.sections,
            "--min-year",
            args.min_year,
            "--max-year",
            args.max_year,
        ],
        "Étape 2/4 — Prétraitement (extraction sections → .txt)",
    )

    if args.dry_run:
        run(
            [
                sys.executable,
                "-m",
                "src.rag.cli",
                "--plan",
                "--strategy",
                args.strategy,
            ],
            "Étape 3/4 — Plan d'indexation (dry-run)",
        )
    else:
        run(
            [
                sys.executable,
                "-m",
                "src.rag.cli",
                "--embed",
                "--strategy",
                args.strategy,
                "--quota-used",
                "0",
                "--batch-size",
                str(args.batch_size),
                "--rpm",
                str(args.rpm),
            ],
            "Étape 3/4 — Indexation vectorielle (chunk + embed + ChromaDB)",
        )

    print(f"\n{'=' * 70}")
    print(f"  Pipeline terminée !")
    print(f"{'=' * 70}")
    print(f"\n  Lance l'interface (deux terminaux) :")
    print(f"    # Terminal 1 — API FastAPI")
    print(f"    source .venv/bin/activate")
    print(
        f"    uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8000"
    )
    print(f"")
    print(f"    # Terminal 2 — UI Next.js")
    print(f"    cd frontend && npm install && npm run dev")
    print(f"")
    print(f"  Interface : http://localhost:3000")
    print(f"  API       : http://localhost:8000")


if __name__ == "__main__":
    main()
