# Revue d'architecture — Finance RAG Agent v2

## Contexte

Le projet est passé d'un pipeline LangGraph **séquentiel** (intent → scope → orchestrateur prix JSON → décomposition → retrieval → rerank → génération → synthèse) à une architecture **agentique orientée outils**, inspirée du Model Context Protocol (MCP).

## Ce qui a bien fonctionné

### Semantic chunking sur les rapports SEC

Le découpage sémantique préserve mieux la structure des tableaux Markdown et des sections Item 1A / Item 7 que des chunks de taille fixe. Les métadonnées (`ticker`, `year`, `file_type`, `section`) restent alignées avec ChromaDB, ce qui permet un filtrage fiable dans `sec_filings_rag_tool`.

### Reranker et fenêtre de contexte

La combinaison **recherche vectorielle large + reranking** (sentence-transformers) réduit le bruit avant injection dans la réponse. Le budget `max_context_chunks` et l'équilibrage multi-tickers évitent qu'une comparaison MSFT/NVDA ne ne retienne que l'entreprise dominante dans le top-k.

### Garbage collector de conversation

Le nœud `gc_node` résume les tours anciens et conserve une fenêtre glissante. Cela limite la dérive de coût API sur les sessions longues sans perdre le fil de la discussion.

## Ce qui a été difficile (V1)

### Routing et désambiguïsation statiques

Plusieurs nœuds (`intent_scope_node`, `query_scope_node`, `tool_orchestrator_node`) décidaient du parcours via heuristiques + JSON LLM fragile. Les conflits (question fondamentale vs besoin de prix, multi-tickers, types de documents) multipliaient les branches conditionnelles.

**Pourquoi l'agent Tool-Calling simplifie :** un seul LLM avec outils explicites planifie l'ordre des appels (RAG → prix → export) selon la requête utilisateur, au lieu d'un graphe figé.

### Quotas d'embeddings

L'indexation batch (`--embed`, RPM, retries) impose une planification manuelle des runs bootstrap. La séparation **Gemini pour le chat** et **OpenAI/GitHub pour les embeddings** évite une ré-indexation complète lors du changement de modèle conversationnel.

## Patrons robustes retenus en V2

| Patron | Rôle |
|--------|------|
| Hybride Vector + Rerank | Qualité des extraits SEC / earnings |
| Outils décorrélés (`rag/tools.py`) | Source de données ≠ logique agent |
| Boucle Agent ↔ Tools | Décision dynamique, plafond `max_tool_iterations` |
| Mémoire + GC | Sessions multi-tours sans explosion de tokens |
| `tool_events` + UI Streamlit | Transparence des actions agent |

## Architecture V2 (résumé)

```
prepare_query → memory_read → guard LLM → agent_node ⇄ tools_node → finalize → memory_write → gc
```

Outils exposés (phase 2 complète) :

1. `sec_filings_rag_tool` — ChromaDB, filtres 10-K / 10-Q / 8-K / 20-F / 6-K / **EARNINGS_CALL**
2. `market_price_tool` — yfinance
3. `validate_claims_tool` — juge NLI LLM borné aux extraits RAG
4. `simulate_portfolio_tool` — allocation fictive contrôlée sur l'univers debug, pas d'exécution réelle
5. `export_investment_report_tool` — fichiers dans `reports/`

Scénario de démo reproductible et tableau MCP : voir [`PHASE2.md`](PHASE2.md).

## Providers LLM

- **openai** / **github_models** : chat + tool calling via API OpenAI-compatible
- **gemini** : chat + function calling (`google-genai`) ; embeddings via `EMBEDDING_PROVIDER` (défaut `openai`)

## Exemple de requête complexe (PRD)

> Génère un rapport d'investissement comparant MSFT et NVDA incluant les risques SEC 2024 et la performance boursière des 6 derniers mois, puis sauvegarde-le.

Comportement attendu : appels successifs à `sec_filings_rag_tool`, `market_price_tool`, `validate_claims_tool`, éventuellement `simulate_portfolio_tool`, synthèse en français, puis `export_investment_report_tool` avec bouton de téléchargement Streamlit.

## Pistes d'amélioration

- Streaming des réponses agent pendant la boucle outils
- Export PDF natif (sans fallback MD)
- Tests d'intégration end-to-end avec corpus earnings réel (`BOOTSTRAP_EARNINGS=true`)
