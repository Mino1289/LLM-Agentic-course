# Finance RAG LangGraph

Assistant financier conversationnel avec RAG sur rapports SEC (10-K/10-Q/8-K/20-F/6-K),
prix temps réel, validation d'affirmations, et export de rapports.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  PIPELINE OFF LINE                          PIPELINE ON LINE     │
│                                                   │
│  SEC EDGAR ─→ download_SEC_reports.py         Query utilisateur  │
│       │                                              │           │
│       ▼                                              ▼           │
│  preprocess/cli.py ─→ .txt ─→ corpus.py        guard_node (LLM) │
│       │                           │                 │            │
│       ▼                           ▼                 ▼            │
│  chunk/semantic.py          rag/cli.py ─→   agent_node (LLM +   │
│       │                     embed + index      outils MCP)      │
│       ▼                           │                 │            │
│  data/processed_data/        ChromaDB  ◄───  tools_node          │
│                               ▲                                  │
│                               │                                  │
│                        retrieve + rerank                          │
│                        (similarité vectorielle                   │
│                         + cross-encoder)                          │
└──────────────────────────────────────────────────────────────────┘
```

LangGraph ([`src/graph/flow.py`](src/graph/flow.py)) :
1. `prepare` — normalisation requête (tickers, années)
2. `guard` — classification d'intention via LLM léger
3. `decompose` — décomposition en sous-requêtes
4. `retrieve` → `rerank` — recherche vectorielle + re-ranking
5. `agent` ⇄ `tools` — boucle LLM avec 5 outils MCP
6. `finalize` — écriture mémoire, nettoyage

## Prérequis

- Python ≥ 3.11
- [Clé API OpenAI](https://platform.openai.com/api-keys) **ou** [GitHub Models PAT](https://github.com/settings/tokens)
- (optionnel) Clé API Gemini pour le chat

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Éditer .env : choisir LLM_PROVIDER et renseigner la clé API
```

## Pipeline complète (une commande)

```bash
python run_pipeline.py
```

Cette commande enchaîne :
1. **Téléchargement** des rapports SEC depuis EDGAR → `data/`
2. **Prétraitement** (extraction sections Item 1A/7) → `data/processed_data/`
3. **Chunking + Embedding + Indexation** ChromaDB → `data/chroma_db/`

Options :

```bash
python run_pipeline.py --sections 1a,7,8 --min-year 2023 --max-year 2026
python run_pipeline.py --batch-size 64 --rpm 200        # + rapide si quota large
python run_pipeline.py --dry-run                         # plan sans indexer
python run_pipeline.py --download                        # forcer le téléchargement
```

## Lancement de l'interface

```bash
streamlit run ui/app_rag.py
```

## Docker

```bash
docker compose run --rm bootstrap    # pipeline complète
docker compose up finance-rag-ui     # interface sur http://localhost:8501
```

Variables `.env` utiles pour Docker : `SKIP_DOWNLOAD_IF_EXISTS`, `BOOTSTRAP_MIN_YEAR`, `BOOTSTRAP_SECTIONS`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_RPM`, etc.

## Pipeline pas à pas

```bash
# 1. Télécharger les rapports SEC
python -m src.fetchers.download_SEC_reports --min-year 2024 --max-year 2026

# 2. Prétraiter (extraire sections → .txt)
python -m src.preprocess.cli --sections 1a,7 --min-year 2024 --max-year 2026

# 3. Planifier l'indexation
python -m src.rag.cli --plan --strategy semantic

# 4. Indexer les embeddings dans ChromaDB
python -m src.rag.cli --embed --strategy semantic --quota-used 0 --batch-size 32 --rpm 120

# 5. Lancer l'interface
streamlit run ui/app_rag.py
```

## Outils agent (5 outils de type MCP)

| Outil | Fichier | Description |
|---|---|---|
| `sec_filings_rag` | `src/tools/sec_filings.py` | Recherche RAG dans les rapports SEC |
| `market_price` | `src/tools/market_price.py` | Prix et performance yfinance (temps réel) |
| `validate_claims` | `src/tools/validate_claims.py` | Validation d'affirmations (NLI) |
| `simulate_portfolio` | `src/tools/portfolio.py` | Allocation fictive et métriques |
| `export_report` | `src/tools/export_report.py` | Export markdown/PDF dans `reports/` |

## Backends LLM supportés

- **OpenAI** — chat + embeddings
- **GitHub Models** — chat + embeddings (même API qu'OpenAI, quota gratuit)
- **Gemini** — chat uniquement (embeddings via OpenAI/GitHub)

## Univers suivi

Tickers : `NVDA`, `ASML`, `AMD`, `ARM`, `MSFT` (configurable dans `src/config/__init__.py`).

Documents : 10-K, 10-Q, 8-K (item 2.02), 20-F, 6-K, transcripts earnings.

## Structure du projet

```
src/
├── config/          # Configuration (tickers)
├── rag/             # Corpus, chunking, indexation, retrieval, reranking
│   ├── chunk/       # Découpage sémantique
│   ├── metadata/    # Métadonnées et IDs de chunks
│   └── search/      # Recherche vectorielle + re-ranking
├── preprocess/      # Extraction sections depuis HTML SEC
├── graph/           # LangGraph (état, noeuds, flow, mémoire)
├── tools/           # 5 outils agent (MCP-like)
├── llm/             # Providers LLM (OpenAI, GitHub Models, Gemini)
├── embeddings/      # Cache, quota, backoff
├── alpaca/          # Client Alpaca (portfolio/trading)
├── data/            # Scripts de download SEC et prix
├── analytics/       # Statistiques d'indexation
├── prompts/         # Prompt système agent
├── mcp/             # Serveur MCP stdio
└── paths.py         # Chemins du projet
ui/
├── app_rag.py       # Interface Streamlit
└── streaming.py     # Streaming temps réel
data/                # Rapports SEC bruts + index ChromaDB
reports/             # Rapports exportés
```

## Tests

```bash
pytest tests/
```

Tests principales : `test_async_nodes`, `test_async_stream`, `test_decompose_quick_wins`, `test_embedding_batching`, `test_provider_stream`, `test_regressions`.
