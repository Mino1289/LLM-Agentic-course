import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

from rag.hybrid_rag import (
    DEFAULT_DAILY_EMBEDDING_LIMIT,
    HybridRAG,
    chunk_text_fixed,
    chunk_text_recursive,
    chunk_text_semantic,
    parse_processed_filename,
)
from rag.paths import PROCESSED_DATA_DIR

CHUNKERS = {
    "semantic": chunk_text_semantic,
    "fixed": chunk_text_fixed,
    "recursive": chunk_text_recursive,
}


def analyze_chunk_counts():
    files = sorted(PROCESSED_DATA_DIR.glob("*.txt"))
    print(f"Fichiers .txt dans {PROCESSED_DATA_DIR} : {len(files)}\n")

    for strategy_name, chunker in CHUNKERS.items():
        total_chunks = 0
        total_chars = 0
        file_stats = []

        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            if not text:
                continue

            file_chars = len(text)
            file_chunks = len(chunker(text))
            total_chars += file_chars
            total_chunks += file_chunks
            file_stats.append(
                {
                    "filename": file_path.name,
                    "chars": file_chars,
                    "chunks": file_chunks,
                }
            )

        print(f"--- Découpage {strategy_name} ---")
        print(f"  Chunks totaux : {total_chunks:,}")
        print(f"  Caractères    : {total_chars:,}")
        print(f"  Appels API    : {total_chunks:,} (si indexation complète)")
        print("  Top 3 fichiers les plus chunkés :")
        for stat in sorted(file_stats, key=lambda x: x["chunks"], reverse=True)[:3]:
            print(
                f"    - {stat['filename']}: {stat['chunks']} chunks "
                f"({stat['chars']:,} chars)"
            )
        print()


def analyze_embedding_plans(quota_used: int, quota_limit: int):
    print(f"=== Plans d'embedding (quota {quota_used}/{quota_limit}) ===\n")
    for strategy in CHUNKERS:
        rag = HybridRAG(chunk_strategy=strategy)
        plan = rag.get_embedding_plan(
            daily_quota_used=quota_used,
            daily_quota_limit=quota_limit,
        )
        print(plan.summary())
        print()


def main():
    parser = argparse.ArgumentParser(description="Estimer chunks et quota embedding.")
    parser.add_argument(
        "--quota-used",
        type=int,
        default=int(os.getenv("EMBEDDING_DAILY_USED", "0")),
    )
    parser.add_argument(
        "--quota-limit",
        type=int,
        default=int(os.getenv("EMBEDDING_DAILY_LIMIT", str(DEFAULT_DAILY_EMBEDDING_LIMIT))),
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Afficher seulement le décompte par stratégie (sans ChromaDB).",
    )
    args = parser.parse_args()

    analyze_chunk_counts()
    if not args.chunks_only:
        analyze_embedding_plans(args.quota_used, args.quota_limit)


if __name__ == "__main__":
    main()
