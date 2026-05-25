import argparse
import subprocess
import sys
from pathlib import Path

from chunking import CHUNKING_METHODS
from embedding_models import EMBEDDING_MODELS


SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str, script_args: list[str] | None = None, description: str = ""):
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if script_args:
        cmd.extend(script_args)
    print(f"\n{'=' * 60}")
    print(f"[PIPELINE] {description or script_name}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[PIPELINE] {script_name} failed (exit code {result.returncode})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline unifié : téléchargement + construction de l'index vectoriel."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip all download steps.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the index build step.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate the Chroma collection from scratch.",
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_METHODS,
        default="paragraph",
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Extract and index tables from PDF files.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=list(EMBEDDING_MODELS.keys()),
        default="bge-small",
        help="Embedding model for the vector index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_download:
        run_script("download_reports.py", description="Download annual reports / 10-K")
        run_script("download_earnings_calls.py", description="Download earnings calls")
        run_script(
            "download_financial_csvs.py",
            ["--start-year", "2021"],
            description="Download SEC financial CSVs",
        )
        run_script(
            "download_stock_prices.py",
            ["--days", "23"],
            description="Download daily stock prices via yfinance",
        )

    if not args.skip_build:
        build_args = [
            f"--chunking={args.chunking}",
            f"--chunk-size={args.chunk_size}",
            f"--overlap={args.overlap}",
            f"--embedding-model={args.embedding_model}",
        ]
        if args.rebuild:
            build_args.append("--rebuild")
        if args.include_tables:
            build_args.append("--include-tables")
        run_script("build_index.py", build_args, description="Build ChromaDB vector index")

    print(f"\n{'=' * 60}")
    print("[PIPELINE] Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
