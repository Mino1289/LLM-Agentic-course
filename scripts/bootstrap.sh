#!/usr/bin/env bash
set -euo pipefail

# Bootstrap RAG — pipeline complète depuis l'image Docker
# Usage: docker compose run --rm bootstrap

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
[ -f .env ] && set -a && source .env && set +a

: "${SKIP_DOWNLOAD_IF_EXISTS:=false}"
: "${SKIP_PREPROCESS_IF_EXISTS:=false}"
: "${BOOTSTRAP_MIN_YEAR:=2024}"
: "${BOOTSTRAP_MAX_YEAR:=2026}"
: "${BOOTSTRAP_SECTIONS:=1a,7}"
: "${BOOTSTRAP_EXCLUDE_8K:=false}"
: "${EMBEDDING_DAILY_LIMIT:=0}"
: "${EMBEDDING_BATCH_SIZE:=32}"
: "${EMBEDDING_RPM:=120}"

# Assurer que les répertoires de données existent
mkdir -p data data/processed_data data/chroma_db

# Étape 1 — Téléchargement SEC
if [ "$SKIP_DOWNLOAD_IF_EXISTS" = "true" ] && [ -n "$(ls data/*.htm 2>/dev/null)" ]; then
    echo "SEC: déjà téléchargé, skip."
else
    echo "SEC: téléchargement…"
    python3 -m src.fetchers.download_SEC_reports \
        --min-year "$BOOTSTRAP_MIN_YEAR" \
        --max-year "$BOOTSTRAP_MAX_YEAR"
fi

# Étape 2 — Prétraitement
EXCLUDE_ARG=""
[ "$BOOTSTRAP_EXCLUDE_8K" = "true" ] && EXCLUDE_ARG="--exclude-8k"

if [ "$SKIP_PREPROCESS_IF_EXISTS" = "true" ] && [ -n "$(ls data/processed_data/*.txt 2>/dev/null)" ]; then
    echo "Preprocess: déjà fait, skip."
else
    echo "Preprocess: extraction sections $BOOTSTRAP_SECTIONS…"
    python3 -m src.preprocess.cli \
        --sections "$BOOTSTRAP_SECTIONS" \
        --min-year "$BOOTSTRAP_MIN_YEAR" \
        --max-year "$BOOTSTRAP_MAX_YEAR" \
        $EXCLUDE_ARG
fi

# Étape 3 — Indexation vectorielle
echo "Indexation: embedding + ChromaDB…"
python3 -m src.rag.cli \
    --embed \
    --strategy semantic \
    --quota-used 0 \
    --batch-size "$EMBEDDING_BATCH_SIZE" \
    --rpm "$EMBEDDING_RPM"

echo "Bootstrap terminé."
