#!/usr/bin/env bash
set -euo pipefail

echo "== Finance RAG bootstrap =="

SKIP_DOWNLOAD_IF_EXISTS="${SKIP_DOWNLOAD_IF_EXISTS:-true}"
SKIP_PREPROCESS_IF_EXISTS="${SKIP_PREPROCESS_IF_EXISTS:-false}"
BOOTSTRAP_MIN_YEAR="${BOOTSTRAP_MIN_YEAR:-2021}"
BOOTSTRAP_SECTIONS="${BOOTSTRAP_SECTIONS:-1a,7}"
EMBEDDING_DAILY_USED="${EMBEDDING_DAILY_USED:-0}"
EMBEDDING_DAILY_LIMIT="${EMBEDDING_DAILY_LIMIT:-1000}"
EMBEDDING_RPM="${EMBEDDING_RPM:-100}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
EMBEDDING_MAX_RETRIES="${EMBEDDING_MAX_RETRIES:-3}"

mkdir -p data rag/processed_data rag/chroma_db

raw_count="$(ls -1 data/* 2>/dev/null | wc -l || true)"
if [[ "${raw_count}" -gt 0 && "${SKIP_DOWNLOAD_IF_EXISTS}" == "true" ]]; then
  echo "Skip download: data/ already contains files (${raw_count})."
else
  echo "Downloading SEC filings..."
  python3 rag/download_SEC_reports.py
fi

processed_count="$(ls -1 rag/processed_data/*.txt 2>/dev/null | wc -l || true)"
if [[ "${processed_count}" -gt 0 && "${SKIP_PREPROCESS_IF_EXISTS}" == "true" ]]; then
  echo "Skip preprocess: rag/processed_data already contains files (${processed_count})."
else
  echo "Preprocessing source files..."
  python3 rag/preprocess.py --sections "${BOOTSTRAP_SECTIONS}" --min-year "${BOOTSTRAP_MIN_YEAR}"
fi

echo "Embedding missing chunks (semantic strategy)..."
python3 rag/hybrid_rag.py \
  --embed \
  --strategy semantic \
  --quota-used "${EMBEDDING_DAILY_USED}" \
  --quota-limit "${EMBEDDING_DAILY_LIMIT}" \
  --rpm "${EMBEDDING_RPM}" \
  --batch-size "${EMBEDDING_BATCH_SIZE}" \
  --retries "${EMBEDDING_MAX_RETRIES}"

echo "Bootstrap complete."
