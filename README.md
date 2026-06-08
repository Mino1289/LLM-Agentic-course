# Finance RAG LangGraph

Projet RAG financier cohérent de bout en bout pour analyser des rapports SEC / foreign issuer (10-K/10-Q/8-K/20-F/6-K) avec:

- stratégie unique: **semantic chunking + vector retrieval + reranking**
- agent **LangGraph v2** (boucle Agent ↔ Outils, style MCP)
- mémoire conversationnelle avec **garbage collector** de contexte
- backends LLM: **OpenAI**, **GitHub Models**, **Gemini** (chat ; embeddings séparés)
- export de rapports dans `reports/` + UI Streamlit

## Mode conversationnel

L'interface se comporte comme un chatbot:

- historique multi-tours visible dans la session (style ChatGPT/Gemini)
- mémoire courte + résumé interne du contexte
- trace des outils utilisés (« Réflexion de l'agent »)
- affichage des sources et métriques par réponse
- bouton de téléchargement si un rapport est exporté
- l'agent choisit dynamiquement RAG SEC, prix yfinance et export de rapport

## Structure

- `rag/download_SEC_reports.py`: téléchargement SEC dans `data/`
- `rag/preprocess.py`: extraction sections (Item 1A/7/8) + fallback foreign issuer vers `rag/processed_data/`
- `rag/hybrid_rag.py`: indexation vectorielle Chroma + retrieval/reranking
- `rag/langgraph_flow.py`: graphe agent v2 (prepare → memory → guard → agent ⇄ tools → gc)
- `rag/tool_executor.py`: validation, dispatch async/sync et normalisation des résultats d'outils
- `rag/mcp_server.py`: serveur MCP stdio exposant les outils du projet
- `rag/tools.py`: 5 outils agent (`sec_filings_rag_tool`, `market_price_tool`, `validate_claims_tool`, `simulate_portfolio_tool`, `export_investment_report_tool`)
- `PHASE2.md`: démo, bilan phase 2, approche type MCP
- `rag/llm_provider.py`: providers OpenAI / GitHub Models / Gemini (tool calling)
- `ui/app_rag.py`: chatbot Streamlit branché sur LangGraph
- `ARCHITECTURE_REVIEW.md`: revue académique V1 vs V2

## Univers suivi et documents utilises

### Entreprises suivies (tickers)

Univers temporairement limité au mode test/debug: `NVDA`, `ASML`, `AMD`, `ARM`, `MSFT`.

### Documents ingeres actuellement

- `10-K` (rapport annuel)
- `10-Q` (rapport trimestriel)
- `8-K` limites a l'item `2.02` (publication de resultats)
- `20-F` (rapport annuel foreign private issuer, ex: ASML/ARM)
- `6-K` (interim foreign private issuer, ex: ASML/ARM)
- transcripts d'earnings calls en `.txt` (si le nom contient `earnings_call`, `conference_call` ou `transcript`)
- sections extraites au preprocess (par defaut): `Item 1A` et `Item 7`
- section optionnelle: `Item 8` (si activee via `--sections 1a,7,8`)

### Documents non ingeres dans cette version

- investor presentations / communiques hors SEC

### Consequence sur la pertinence

Le chatbot est coherent sur l'analyse fondamentale long-terme et intermediaire (10-K/10-Q/8-K/20-F/6-K), mais reste limite au perimetre des documents SEC/foreign issuer + transcripts disponibles.

## Configuration

1. Copier le template:

```bash
cp .env.example .env
```

2. Renseigner les variables nécessaires dans `.env`:

- `LLM_PROVIDER=openai`, `github_models`, ou `gemini`
- OpenAI:
  - `OPENAI_API_KEY`
  - `OPENAI_CHAT_MODEL` (défaut: `gpt-4o-mini`)
  - `OPENAI_EMBEDDING_MODEL` (défaut: `text-embedding-3-small`)
- GitHub Models:
  - `GITHUB_MODELS_API_KEY`
  - `GITHUB_CHAT_MODEL` (défaut: `gpt-4.1-mini` — sans préfixe `openai/`)
  - `GITHUB_EMBEDDING_MODEL` (défaut: `text-embedding-3-small`)
- Gemini (chat uniquement ; embeddings via `EMBEDDING_PROVIDER`, défaut `openai`):
  - `GEMINI_API_KEY`
  - `GEMINI_CHAT_MODEL` (défaut: `gemini-2.0-flash`)
  - `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY` pour Chroma
- LangSmith (visualisation):
  - `LANGSMITH_TRACING=true`
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_PROJECT`
  - `LANGSMITH_REGION=eu` et `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` si compte sur [eu.smith.langchain.com](https://eu.smith.langchain.com)
- SEC:
  - `SEC_USER_AGENT` (obligatoire pour crawler SEC)

## Lancement local

### 1) Installer les dépendances

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Ingestion SEC (optionnel si vous avez déjà des fichiers dans `data/`)

```bash
python3 rag/download_SEC_reports.py
# Transcripts earnings (optionnel)
python3 fetch/download_earnings_calls.py
```

### 3) Pré-traitement

```bash
python3 rag/preprocess.py --sections 1a,7 --min-year 2024 --max-year 2026
```

Le preprocess reconstruit `rag/processed_data/` à chaque exécution afin de ne conserver
que les sections et années demandées. Utilisez `--no-clean-output` uniquement pour un
debug incrémental volontaire.

Par defaut, les `8-K` (item `2.02`) sont inclus.  
Les documents `20-F` et `6-K` sont conserves via fallback texte integral quand les
sections SEC standard `Item 1A/7/8` ne sont pas detectees.
Pour les exclure explicitement:

```bash
python3 rag/preprocess.py --sections 1a,7 --min-year 2024 --max-year 2026 --exclude-8k
```

### 4) Planifier l'indexation

```bash
python3 rag/hybrid_rag.py --plan --strategy semantic
```

### 5) Indexer les embeddings

```bash
python3 rag/hybrid_rag.py --embed --strategy semantic --quota-used 0
```

### Serveur MCP

Installer les dépendances puis lancer le serveur stdio:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m rag.mcp_server
```

Ne collez pas la configuration JSON ci-dessous dans le terminal du serveur.
En mode `stdio`, le serveur attend des messages JSON-RPC MCP envoyés par un
client compatible, pas un fichier de configuration.

Exemple de configuration à mettre dans le client MCP:

```json
{
  "mcpServers": {
    "finance-rag": {
      "command": "/home/julien/Documents/UQAC(nogit)/LLM-Agentic-course/.venv/bin/python",
      "args": ["-m", "rag.mcp_server"],
      "cwd": "/home/julien/Documents/UQAC(nogit)/LLM-Agentic-course"
    }
  }
}
```

Le serveur expose les 5 outils du projet via MCP:

- `sec_filings_rag_tool`
- `market_price_tool`
- `validate_claims_tool`
- `simulate_portfolio_tool`
- `export_investment_report_tool`

Le serveur charge Chroma avec `max_new_embeddings=0`: il n'embedde pas de
nouveaux chunks au démarrage. Lancez l'indexation séparément si le plan indique
des chunks manquants.

Pour un temps d'indexation raisonnable, utilisez les embeddings en batch:

```bash
python3 rag/hybrid_rag.py --embed --strategy semantic --quota-used 0 --batch-size 32 --rpm 120
```

### 6) Lancer l'UI

```bash
streamlit run ui/app_rag.py
```

## Lancement Docker (recommandé)

```bash
docker compose run --rm bootstrap
docker compose up finance-rag-ui
```

L'UI est disponible sur `http://localhost:8501`.

Le service `bootstrap` exécute:

1. téléchargement SEC (optionnel selon flags)
2. preprocess
3. embedding

Flags `.env` utiles:

- `SKIP_DOWNLOAD_IF_EXISTS=false` (le crawler ignore déjà les fichiers téléchargés)
- `SKIP_PREPROCESS_IF_EXISTS=false`
- `BOOTSTRAP_MIN_YEAR=2024`
- `BOOTSTRAP_MAX_YEAR=2026`
- `BOOTSTRAP_SECTIONS=1a,7`
- `BOOTSTRAP_EXCLUDE_8K=false` (par defaut les 8-K sont inclus)
- `BOOTSTRAP_EARNINGS=false` (télécharge les transcripts si `true`)
- `MAX_TOOL_ITERATIONS=6`
- `EMBEDDING_DAILY_LIMIT=0` (`0` = quota illimite)
- `EMBEDDING_BATCH_SIZE=32`
- `EMBEDDING_MAX_RETRIES=3`
- `EMBEDDING_RPM=120`
- `QUERY_DECOMPOSE_COUNT=4` (nombre de sous-requêtes générées par LangGraph)
- `PRICE_TOOL_ENABLED=true`
- `PRICE_MAX_DAYS=180`
- `PRICE_MAX_POINTS=40`
- `PRICE_MAX_TICKERS=3`
- `PRICE_DEFAULT_DAYS=90`
- `PRICE_MAX_ATTEMPTS=2`

## Architecture LangGraph v2 (Agent ↔ Tools)

Le graphe exécute:

1. `prepare_query_node` — normalisation requête / tickers / années
2. `memory_read_node` — résumé conversationnel
3. `guard_node` — décision LLM légère avec accès mémoire : hors sujet, coverage info, clarification évidente
4. `agent_node` — LLM avec tool calling (boucle)
5. `tools_node` — orchestration minimale, déléguée à `ToolExecutor`
6. `finalize_node` → `memory_write_node` → `gc_node`

Exemple de requête complexe (démo phase 2) :

> Compare MSFT et NVDA (risques SEC 2024 + performance 6 mois), valide les affirmations clés, propose une allocation fictive 50/50, puis sauvegarde le rapport.

```bash
streamlit run ui/app_rag.py
```

Le `gc_node` compresse l'historique pour limiter le coût token/API.

Documentation phase 2 : [`PHASE2.md`](PHASE2.md). Revue architecture : [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).

Tests agent (sans API) :

```bash
python -m unittest tests.test_regressions.AgentToolsTests -v
```

## Visualisation LangSmith

Les noeuds du graphe sont instrumentés via `@traceable`.  
Une fois `LANGSMITH_TRACING=true` et `LANGSMITH_API_KEY` définis, chaque run est visible dans le projet `LANGSMITH_PROJECT`.

### Lancer et visualiser

1. Activer les variables dans `.env`:
   - `LANGSMITH_TRACING=true`
   - `LANGSMITH_API_KEY=...`
   - `LANGSMITH_PROJECT=finance-rag-langgraph`
2. Lancer l'app Streamlit et poser des questions.
3. Ouvrir [https://eu.smith.langchain.com](https://eu.smith.langchain.com) (EU) ou [smith.langchain.com](https://smith.langchain.com) (US), puis le projet `LANGSMITH_PROJECT`.

### LangGraph Studio (optionnel)

Pour visualiser localement le graphe en mode dev:

```bash
pip install langgraph-cli
langgraph dev --config langgraph.json
```
