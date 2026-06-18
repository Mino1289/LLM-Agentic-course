# Finance RAG · Hub-and-Spoke

Assistant financier conversationnel pour le cours **8INF829** — du RAG sur rapports SEC à l'orchestration multi-agents (architecture Hub-and-Spoke), avec interface web Next.js et API FastAPI.

Capacités principales :

- RAG sur filings SEC (10-K, 10-Q, 8-K, 20-F, 6-K) et transcripts
- Prix de marché, news, portefeuille paper trading (Alpaca)
- Validation d'affirmations et export de rapports (Markdown/PDF)
- Routage automatique **agent simple** vs **multi-agents** selon la requête
- Approbation humaine avant exécution d'un ordre de trade

---

## Démarrage rapide (nouveaux utilisateurs)

### 1. Prérequis

- **Python** ≥ 3.11
- **Node.js** ≥ 20 (pour le frontend)
- Une clé API LLM : [OpenAI](https://platform.openai.com/api-keys), [GitHub Models](https://github.com/settings/tokens) (gratuit), Gemini, Azure OpenAI ou NVIDIA NIM
- (optionnel) Clés [Alpaca Paper Trading](https://alpaca.markets/) pour le portefeuille et les ordres simulés

### 2. Installation

```bash
git clone <url-du-repo>
cd LLM-Agentic-course

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Éditer .env : LLM_PROVIDER + clé API correspondante
# Exemple minimal : LLM_PROVIDER=openai et OPENAI_API_KEY=sk-...
```

### 3. Indexer les données SEC (première fois)

```bash
python run_pipeline.py
```

Cette commande télécharge les rapports SEC, les prétraite et indexe les embeddings dans ChromaDB (`data/chroma_db/`). Comptez plusieurs minutes selon votre quota d'embeddings.

Options utiles :

```bash
python run_pipeline.py --dry-run          # voir le plan sans indexer
python run_pipeline.py --min-year 2023    # ajuster la période
python run_pipeline.py --download         # forcer le re-téléchargement
```

### 4. Lancer l'interface

**Deux terminaux** :

```bash
# Terminal 1 — API FastAPI
source .venv/bin/activate
uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8000

# Terminal 2 — UI Next.js
cd frontend && npm install && npm run dev
```


| Service          | URL                                                                  |
| ---------------- | -------------------------------------------------------------------- |
| **Interface**    | [http://localhost:3000](http://localhost:3000)                       |
| **API**          | [http://localhost:8000](http://localhost:8000)                       |
| **Health check** | [http://localhost:8000/api/health](http://localhost:8000/api/health) |


Le frontend proxifie les appels `/api/`* vers le backend (voir `frontend/next.config.ts`).

> L'ancienne interface Streamlit (`ui/app_rag.py`) est **dépréciée**. Utiliser Next.js + FastAPI.

### Ports alternatifs

Si les ports 8000 (API) ou 3000 (UI) sont déjà utilisés :

```bash
# Terminal 1 — API sur 8080
uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8080

# Terminal 2 — UI sur 3030
cd frontend && npm run dev -- -p 3030
```

Configurer `[frontend/.env.local](frontend/.env.local)` :

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8080
API_PROXY_TARGET=http://127.0.0.1:8080
PORT=3030
```

Et dans `.env` (racine), autoriser l'origine du frontend :

```env
CORS_ORIGINS=http://localhost:3030,http://127.0.0.1:3030
```

Redémarrer les deux serveurs après modification des variables d'environnement.

---

## Interface web

L'UI Next.js (`frontend/`) propose :


| Zone                      | Description                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| **Barre latérale**        | Historique des conversations, nouvelle conversation, bascule FR/EN                         |
| **Zone de chat**          | Questions en langage naturel, réponses streamées en temps réel                             |
| **Pipeline multi-agents** | LEDs de statut par agent (Intent Router, PM, Analystes, Compliance…) pendant le traitement |
| **Raisonnements**         | Panneau déroulant des étapes intermédiaires (mode streaming)                               |
| **Artefacts**             | Sources RAG, étapes outils, stats debug, rapports exportés                                 |
| **Approbation trade**     | Carte d'approbation humaine si un ordre buy/sell est proposé (pas pour une simple analyse) |
| **Paramètres**            | Panneau de config (chunks max, sous-requêtes, fenêtre prix, itérations agent…)             |


### Exemples de questions

```
Compare les risques SEC et la perf 6 mois de NVDA et AMD
Quels risques NVDA mentionne-t-il dans son 10-K 2024 ?
Quel est le prix de MSFT sur les 3 derniers mois ?
Quelles sont les dernières news sur NVIDIA ?
Montre mon portefeuille paper
Achète 5 actions NVDA          → déclenche le flux trade + approbation humaine
```

### Routage automatique

L'**Intent Router** choisit le mode d'exécution :


| Mode              | Quand                                                             | Comportement                                                                         |
| ----------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Agent simple**  | Une seule famille d'outils (news seule, prix seul, SEC seul…)     | Agent LangGraph direct avec outils                                                   |
| **Hub-and-Spoke** | Plusieurs familles d'outils (ex. SEC + perf) ou action de trading | PM → analystes en parallèle → synthèse → compliance (si trade) → approbation humaine |


Familles d'outils détectées : filings SEC, news, prix/perf, portefeuille, trading, export rapport.

---

## Architecture

### Pipeline offline (données)

```
SEC EDGAR → download_SEC_reports.py → preprocess/cli.py → chunk + embed → ChromaDB
```

### Pipeline online (requête utilisateur)

```
Requête → Intent Router
              ├─ simple  → Agent LangGraph (prepare → guard → retrieve → agent ⇄ tools → finalize)
              └─ complex → Hub-and-Spoke :
                              PM (plan) → Analystes parallèles → PM (synthèse)
                              → Compliance (si ordre proposé) → Approbation humaine → Executor
```

### Agents Hub-and-Spoke


| Agent                    | Outils                                                                 | Rôle                                               |
| ------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------- |
| **Portfolio Manager**    | —                                                                      | Planifie, délègue, synthétise                      |
| **Analyste fondamental** | `sec_filings_rag_tool`, `get_news_tool`                                | Risques SEC, news, sentiment                       |
| **Analyste quantitatif** | `market_price_tool`, `portfolio_history_tool`                          | Prix, perf, volatilité                             |
| **Compliance Validator** | `validate_claims_tool`, `portfolio_info_tool`, `account_activity_tool` | Garde-fous avant trade                             |
| **Executor Trader**      | `place_trade_tool`, `close_position_tool`                              | Exécution paper trading (après PASS + approbation) |


Fichiers clés :

- Graphe simple : `[src/graph/flow.py](src/graph/flow.py)`
- Graphe Hub-and-Spoke : `[src/orchestration/hub_graph.py](src/orchestration/hub_graph.py)`
- API streaming SSE : `[api/services/hub_runner.py](api/services/hub_runner.py)`

---

## Configuration (`.env`)

Copier `.env.example` vers `.env`. Variables essentielles :


| Variable                               | Description                                                              |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `LLM_PROVIDER`                         | `openai`, `github_models`, `gemini`, `azure_openai`, `nvidia_nim`        |
| `OPENAI_API_KEY`                       | Clé OpenAI (chat + embeddings par défaut)                                |
| `GITHUB_MODELS_API_KEY`                | Alternative gratuite (chat + embeddings)                                 |
| `EMBEDDING_PROVIDER`                   | Provider embeddings si différent du chat (ex. `openai` avec chat Gemini) |
| `SEC_USER_AGENT`                       | Identité requise par la SEC (`NomApp email@domaine.com`)                 |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Paper trading (optionnel)                                                |
| `LANGSMITH_`*                          | Traçabilité LangSmith (optionnel)                                        |


Voir `.env.example` pour la liste complète (quota embeddings, fenêtres prix, mémoire conversationnelle…).

---

## Docker

```bash
docker compose run --rm bootstrap              # pipeline complète (indexation)
docker compose up finance-rag-api finance-rag-web
```

- UI : [http://localhost:3000](http://localhost:3000)
- API : [http://localhost:8000](http://localhost:8000)

Variables utiles : `SKIP_DOWNLOAD_IF_EXISTS`, `BOOTSTRAP_MIN_YEAR`, `BOOTSTRAP_SECTIONS`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_RPM`.

---

## Outils agent


| Outil                                      | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `sec_filings_rag_tool`                     | Recherche RAG dans les rapports SEC indexés |
| `market_price_tool`                        | Prix et performance (yfinance)              |
| `get_news_tool`                            | Actualités récentes par ticker              |
| `validate_claims_tool`                     | Validation d'affirmations vs sources RAG    |
| `portfolio_info_tool`                      | État du compte paper Alpaca                 |
| `portfolio_history_tool`                   | Historique de performance du portefeuille   |
| `account_activity_tool`                    | Activité récente du compte                  |
| `place_trade_tool` / `close_position_tool` | Ordres paper trading                        |
| `export_investment_report_tool`            | Export Markdown/PDF dans `reports/`         |


---

## Backends LLM supportés


| Provider      | Chat | Embeddings               |
| ------------- | ---- | ------------------------ |
| OpenAI        | ✓    | ✓                        |
| GitHub Models | ✓    | ✓                        |
| Azure OpenAI  | ✓    | ✓                        |
| NVIDIA NIM    | ✓    | ✓                        |
| Gemini        | ✓    | via `EMBEDDING_PROVIDER` |


---

## Univers suivi

Tickers par défaut : `NVDA`, `ASML`, `AMD`, `ARM`, `MSFT` — configurables dans `[src/config/__init__.py](src/config/__init__.py)`.

Documents indexés : 10-K, 10-Q, 8-K (item 2.02), 20-F, 6-K, transcripts earnings.

---

## Structure du projet

```
api/                 # Backend FastAPI (chat SSE, config, rapports, trade approval)
frontend/            # Interface Next.js (i18n FR/EN, chat, artefacts, paramètres)
src/
├── config/          # Tickers, constantes
├── rag/             # Corpus, chunking, indexation, retrieval, reranking
├── preprocess/      # Extraction sections SEC (Item 1A, MD&A…)
├── graph/           # Agent LangGraph simple (nœuds, état, mémoire)
├── orchestration/   # Hub-and-Spoke (intent router, PM, analystes, compliance)
├── tools/           # Outils agent (SEC, prix, news, portfolio, trading…)
├── llm/             # Providers LLM
├── alpaca/          # Client Alpaca paper trading
└── embeddings/      # Cache, quota, backoff
ui/                  # Streamlit legacy (déprécié)
data/                # Rapports SEC bruts + ChromaDB
reports/             # Rapports exportés
tests/               # Tests pytest + unittest
```

---

## Tests

```bash
# Tests pytest
pytest tests/

# Tests routage / trade intent (unittest, sans dépendance langgraph lourde)
python3 -m unittest tests.test_intent_router tests.test_trade_intent -v
```

---

## Dépannage


| Problème                  | Piste                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| UI sans réponse           | Vérifier que l'API tourne ; si ports custom, vérifier `NEXT_PUBLIC_API_BASE_URL` et `CORS_ORIGINS` |
| Erreur embeddings / quota | Ajuster `EMBEDDING_RPM`, `EMBEDDING_BATCH_SIZE` ou attendre reset quota                            |
| ChromaDB vide             | Relancer `python run_pipeline.py`                                                                  |
| Trade non exécuté         | Normal si la requête est analytique ; l'approbation n'apparaît que pour un ordre explicite         |
| SEC 403                   | Renseigner `SEC_USER_AGENT` avec un email valide                                                   |


Pour le contexte pédagogique du cours (phases RAG → agent → orchestration), voir `[scope.md](scope.md)`.