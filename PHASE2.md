# Phase 2 — Agent LLM unique + RAG + outils (type MCP)

## Objectif

Démontrer comment un **agent unique** combine :

- raisonnement (LLM),
- récupération d'information (RAG SEC / earnings),
- utilisation d'outils (prix, validation, simulation, export).

Le graphe LangGraph v2 orchestre une boucle `agent_node` ⇄ `tools_node` sans routing statique figé.

## Approche « type MCP »

Le projet n'expose pas un serveur MCP (stdio/HTTP). Il reprend les **idées** du Model Context Protocol :

| MCP officiel | Ce projet |
|--------------|-----------|
| Serveur + transport réseau | Non |
| Contrat outil (nom, description, schéma JSON) | `get_tool_definitions()` dans [`rag/tools.py`](rag/tools.py) |
| Exécution des outils | `execute_tool()` + `tools_node` |
| Orchestration | LLM dans `agent_node` (function calling OpenAI / Gemini) |

Les outils sont **découplés** de la logique agent : le LLM choisit quand les appeler selon la requête.

## Catalogue des 5 outils

| Outil | Rôle | Quand l'utiliser |
|-------|------|------------------|
| `sec_filings_rag_tool` | Recherche ChromaDB (10-K, 10-Q, 8-K, EARNINGS_CALL) | Faits fondamentaux, risques, MD&A, transcripts |
| `market_price_tool` | Prix / performance via yfinance | Comparaisons de performance, fenêtres de dates |
| `validate_claims_tool` | Vérifie des affirmations vs extraits RAG déjà récupérés | Après RAG, avant conclusion sur des faits SEC |
| `simulate_portfolio_tool` | Allocation fictive (NVDA/AMD/MSFT, poids = 100 %) | Rebalancement / allocation pédagogique, **aucun ordre réel** |
| `export_investment_report_tool` | Sauvegarde Markdown dans `reports/` | Demande explicite de rapport fichier |

## Scénario de démo (Streamlit)

Lancer l'interface :

```bash
streamlit run ui/app_rag.py
```

Coller cette requête :

> Compare MSFT et NVDA (risques SEC 2024 + performance sur 6 mois), valide les affirmations clés, propose une allocation fictive 50/50, puis sauvegarde le rapport.

**Ordre attendu des outils** (indicatif) :

1. `sec_filings_rag_tool` (MSFT et NVDA, 2024, 10-K / risques)
2. `market_price_tool` (fenêtre ~6 mois)
3. `validate_claims_tool` (2–4 affirmations issues du RAG)
4. `simulate_portfolio_tool` (`{"MSFT": 50, "NVDA": 50}`)
5. `export_investment_report_tool` (corps Markdown complet)

Vérifier dans l'UI :

- expander **Réflexion de l'agent (outils utilisés)** ;
- sources SEC affichées ;
- bouton de téléchargement du rapport si export réussi.

## Bilan phase 2

### Ce qui a bien fonctionné

- **Boucle agent ↔ outils** : un seul LLM planifie RAG → prix → validation → simulation → export.
- **Outils décorrélés** dans `rag/tools.py` : faciles à tester et à étendre.
- **RAG réutilisé** : `sec_filings_rag_tool` encapsule décomposition, retrieval et rerank existants.
- **Transparence** : `tool_events` + debug Streamlit + LangSmith (`@traceable`).

### Ce qui a été difficile

- **Routing V1** (heuristiques + JSON fragile) remplacé par tool calling — migration conceptuelle plus que technique.
- **Validation sans LLM** : heuristique token-overlap rapide mais moins nuancée qu'un judge LLM.
- **Indexation** : quotas embeddings et corpus earnings optionnel restent des prérequis pour une démo riche.

### Patrons robustes

- Hybride **vector + rerank** avant injection dans l'agent.
- **Plafond** `max_tool_iterations` pour éviter les boucles coûteuses.
- **Mémoire + GC** pour sessions multi-tours.
- **Injection de contexte RAG** dans `validate_claims_tool` via le state (`final_chunks`).

Pour la comparaison détaillée V1 → V2, voir [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).

## Tests automatisés

```bash
source .venv/bin/activate
python -m unittest tests.test_regressions.AgentToolsTests -v
```

Couvre : validation, simulation, boucle agent mockée (sans clé API).
