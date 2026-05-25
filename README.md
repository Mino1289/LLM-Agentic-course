# RAG Semi-conducteurs

Projet RAG sur des entreprises de semi-conducteurs: NVIDIA, AMD, Intel, TSMC et ASML.

Le corpus peut contenir plusieurs types de documents:

- rapports annuels / 10-K en PDF
- transcripts d'earnings calls en TXT/PDF
- tableaux extraits de PDF
- CSV financiers structurés issus de SEC Company Facts
- CSV de prix journaliers issus de Yahoo Finance via yfinance
- fichiers `.txt`, `.md`, `.csv`, `.tsv` ajoutés manuellement dans `data/raw`

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Configuration des clés API:

```bash
cp .env.example .env
```

Ensuite renseigner dans `.env`:

```bash
OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

## Télécharger les données

Rapports annuels et 10-K:

```bash
.venv/bin/python src/download_reports.py
```

Earnings calls:

```bash
.venv/bin/python src/download_earnings_calls.py
```

CSV financiers SEC:

```bash
.venv/bin/python src/download_financial_csvs.py --start-year 2021
```

Prix journaliers des actions:

```bash
.venv/bin/python src/download_stock_prices.py --days 23
```

Les CSV déjà présents par entreprise sont ignorés au prochain lancement. Pour retélécharger volontairement tous les fichiers:

```bash
.venv/bin/python src/download_stock_prices.py --days 23 --force
```

Les fichiers sont sauvegardés dans `data/raw`.

## Construire l'index vectoriel

Build simple avec le modèle par défaut:

```bash
.venv/bin/python src/build_index.py --rebuild
```

Build avec chunking par paragraphes:

```bash
.venv/bin/python src/build_index.py --rebuild --chunking paragraph
```

Build avec extraction des tableaux dans les PDF:

```bash
.venv/bin/python src/build_index.py --rebuild --include-tables
```

Build complet recommandé pour tester plusieurs formats:

```bash
.venv/bin/python src/build_index.py --rebuild --chunking paragraph --include-tables
```

## Options d'indexation à comparer

### Chunking

Méthode simple, par taille fixe:

```bash
.venv/bin/python src/build_index.py --rebuild --chunking simple
```

Méthode par paragraphes:

```bash
.venv/bin/python src/build_index.py --rebuild --chunking paragraph
```

Changer la taille et l'overlap:

```bash
.venv/bin/python src/build_index.py --rebuild --chunking paragraph --chunk-size 1200 --overlap 200
```

### Embeddings

Modèle léger:

```bash
.venv/bin/python src/build_index.py --rebuild --embedding-model bge-small
```

Modèle plus gros:

```bash
.venv/bin/python src/build_index.py --rebuild --embedding-model bge-base
```

Modèle encore plus gros:

```bash
.venv/bin/python src/build_index.py --rebuild --embedding-model bge-large
```

Les collections Chroma sont séparées automatiquement:

- `bge-small` -> `semiconductor_reports`
- `bge-base` -> `semiconductor_reports_bge_base`
- `bge-large` -> `semiconductor_reports_bge_large`

Il faut interroger avec le même modèle que celui utilisé au build.

## Interroger le RAG

Question simple avec vector search + reranking:

```bash
.venv/bin/python src/query_rag.py "What did NVIDIA say about Blackwell demand?"
```

Utiliser un index construit avec `bge-base`:

```bash
.venv/bin/python src/query_rag.py --embedding-model bge-base "Compare NVIDIA and AMD revenue growth"
```

Limiter le nombre de résultats:

```bash
.venv/bin/python src/query_rag.py --top-k 5 "Which company had the highest revenue?"
```

Décomposer une question large en plusieurs requêtes ciblées avec un LLM:

```bash
.venv/bin/python src/query_rag.py --decompose --search-mode hybrid --rerank "Which company is best positioned for AI infrastructure growth?"
```

Contrôler le nombre maximum de sous-requêtes:

```bash
.venv/bin/python src/query_rag.py --decompose --query-count 7 --search-mode hybrid "Compare NVIDIA, AMD, TSMC and ASML exposure to AI demand."
```

Le LLM ne génère pas la réponse finale dans ce mode. Il produit seulement des requêtes de recherche plus ciblées, puis le RAG récupère et rerank les passages depuis ChromaDB.

## Méthodes de recherche à comparer

Vectoriel seul:

```bash
.venv/bin/python src/query_rag.py --search-mode vector --no-rerank "What did NVIDIA say about Blackwell demand?"
```

BM25 seul:

```bash
.venv/bin/python src/query_rag.py --search-mode bm25 --no-rerank "What did NVIDIA say about Blackwell demand?"
```

Hybride vectoriel + BM25:

```bash
.venv/bin/python src/query_rag.py --search-mode hybrid --no-rerank "What did NVIDIA say about Blackwell demand?"
```

Hybride + reranking:

```bash
.venv/bin/python src/query_rag.py --search-mode hybrid --rerank "What did NVIDIA say about Blackwell demand?"
```

Comparer automatiquement les stratégies:

```bash
.venv/bin/python src/query_rag.py --compare --top-k 5 "What did NVIDIA say about Blackwell demand?"
```

Comparer les stratégies avec décomposition de requête:

```bash
.venv/bin/python src/query_rag.py --compare --decompose "Which company seems best positioned for AI infrastructure growth?"
```

Le mode `--compare` écrit aussi les passages sélectionnés dans `comparison.txt`.
Pour choisir un autre fichier:

```bash
.venv/bin/python src/query_rag.py --compare --comparison-output results/blackwell_comparison.txt "What did NVIDIA say about Blackwell demand?"
```

Le mode `--compare` affiche:

- vector only
- BM25 only
- hybrid vector + BM25
- hybrid + reranking

## Questions utiles pour les tests

Earnings calls:

```bash
.venv/bin/python src/query_rag.py --compare "What did NVIDIA management say about demand for Blackwell during the Q4 2025 earnings call?"
```

Risques dans les rapports:

```bash
.venv/bin/python src/query_rag.py --compare "What are the main supply chain risks mentioned by NVIDIA and AMD?"
```

Tableaux et CSV financiers:

```bash
.venv/bin/python src/query_rag.py --compare "Which semiconductor company reported the highest revenue?"
```

Comparaison R&D:

```bash
.venv/bin/python src/query_rag.py --compare "Compare research and development expenses for NVIDIA, AMD, Intel, TSMC and ASML."
```

Capex:

```bash
.venv/bin/python src/query_rag.py --compare "Which company reported the highest capital expenditures?"
```

Termes exacts:

```bash
.venv/bin/python src/query_rag.py --compare "What do the documents say about export controls?"
```

## Ce qui est mesuré ou observable

Pour comparer les stratégies, regarder:

- documents remontés dans le top-k
- entreprise / fichier source / page
- type de contenu: `text` ou `table`
- source type: `pdf`, `txt`, `csv`, etc.
- score vectoriel
- score BM25
- score hybride
- score multi-query si `--decompose` est activé
- score de reranking

## Structure des scripts

- `src/download_reports.py`: télécharge les rapports PDF
- `src/download_earnings_calls.py`: télécharge les earnings calls
- `src/download_financial_csvs.py`: génère les CSV financiers SEC
- `src/download_stock_prices.py`: génère les CSV de prix journaliers via yfinance
- `src/chunking.py`: chunking `simple` et `paragraph`
- `src/ingest.py`: lit PDF/TXT/MD/CSV/TSV et produit les documents à indexer
- `src/build_index.py`: calcule les embeddings et remplit ChromaDB
- `src/retrieval.py`: recherche vectorielle, BM25, hybride et reranking
- `src/query_rag.py`: CLI de requête et comparaison des méthodes
- `src/query_planner.py`: décomposition LLM des questions en requêtes ciblées
- `src/env_config.py`: chargement local du fichier `.env`
- `src/embedding_models.py`: configuration des modèles d'embedding

## Notes

Après un changement de chunking, d'embedding, ou d'extraction de tableaux, utiliser `--rebuild`.

Le premier lancement d'un nouveau modèle peut télécharger des poids Hugging Face et prendre du temps.

BM25 est construit à la volée depuis les chunks stockés dans ChromaDB. Il ne nécessite pas de rebuild séparé.
