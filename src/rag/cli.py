"""CLI pour le module RAG (planification et indexation)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.rag.core import HybridRAG
from src.rag.indexing import DEFAULT_QUOTA_STATE_PATH
from src.rag.types import (
    DEFAULT_DAILY_EMBEDDING_LIMIT,
    DEFAULT_EMBEDDING_RPM,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_RETRIES,
)
from src.embeddings.backoff import BackoffConfig


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG — planifier ou lancer l'indexation vectorielle."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Afficher le nombre de chunks et le quota sans embedder.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Lancer les embeddings (respecte le quota restant).",
    )
    parser.add_argument("--strategy", choices=["semantic"], default="semantic")
    parser.add_argument("--quota-used", type=int, default=None)
    parser.add_argument("--quota-state", type=Path)
    parser.add_argument(
        "--quota-limit",
        type=int,
        default=int(
            os.getenv("EMBEDDING_DAILY_LIMIT", str(DEFAULT_DAILY_EMBEDDING_LIMIT))
        ),
    )
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("EMBEDDING_RPM", str(DEFAULT_EMBEDDING_RPM))),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("EMBEDDING_MAX_RETRIES", str(DEFAULT_EMBEDDING_RETRIES))),
    )
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if not args.embed:
        args.plan = True

    rag = HybridRAG(chunk_strategy=args.strategy)
    quota_state_path = args.quota_state or DEFAULT_QUOTA_STATE_PATH
    rag.load_and_index_data(
        max_files=args.max_files,
        max_new_embeddings=args.max_new,
        daily_quota_used=args.quota_used,
        daily_quota_limit=args.quota_limit,
        rpm_limit=args.rpm,
        embedding_batch_size=args.batch_size,
        max_embedding_retries=args.retries,
        quota_state_path=quota_state_path,
        backoff_config=BackoffConfig.from_env(),
        dry_run=args.plan and not args.embed,
    )

    if args.plan and not args.embed:
        print(
            f"\nPour lancer l'embedding : python3 src/rag/cli.py --embed --strategy {args.strategy} --quota-state {quota_state_path}"
        )


if __name__ == "__main__":
    main()
