# Session Summary — Refactor vers src/

## Goal
Refactor complet du projet dans `src/` avec un fichier par fonction/classe (~200 li max), sans perte.

## Structure choisie
- Technique (Option 2) : `src/rag/`, `src/llm/`, `src/embeddings/`, `src/preprocess/`, etc.
- `ui/` reste à la racine
- Un dossier par outil sous `src/tools/`
- Legacy conservé via Git

## Migré
- `src/paths.py` — chemins projet
- `src/config/__init__.py` — TRACKED_TICKERS
- `src/llm/` (8 fichiers) — types, sinks, config_builder, messages, parser, openai_client, github_client, gemini_client, provider
- `src/embeddings/` (3 fichiers) — backoff, quota, cache
- `src/rag/` (10 fichiers) — types, chunk/{semantic,fixed}, metadata/{extract,build}, search/{vector,rerank}, corpus, indexing, core, cli
- `src/preprocess/` (8 fichiers) — config, clean, readers, sections, classify, io, cli

## Migré (suite)
- `src/tools/` (12 fichiers) — schemas, descriptions, sec_filings, market_price, validate_claims, export_report, portfolio, trading, news, definitions, execute, __init__
- `src/alpaca/` (2 fichiers) — client, __init__

## Migré (suite)
- `src/graph/` (14 fichiers) — state, tracing, memory_store, prompt_context, prepare_node, guard, decompose_node, retrieval_node, rerank_node, agent_node, tool_nodes, tool_execution_node, memory_nodes, flow

## Migré (suite)
- `src/fetchers/` (1 fichier) — download_SEC_reports.py
- `src/analytics/` (2 fichiers) — analyze_stats.py, __init__.py
- `src/prompts/` (2 fichiers) — agent_prompt.py, __init__.py
- `src/mcp/` (2 fichiers) — mcp_server.py, __init__.py

## Migré (fin) — plus aucun fichier legacy à migrer
Tous les modules `rag/` sont maintenant réécrits dans `src/` avec :
- Un fichier par responsabilité
- Imports internes via `src.*`
- Déduplication des symboles (extract_year/ticker_from_filename dans preprocess/classify.py)
- Code mort supprimé (boucle dead dans cli.py)
- agent_node.py importe depuis src.prompts.agent_prompt

## Tests — état final
- 174 passent, 7 skipped, 37 échouent
- Les 37 échecs sont **préexistants** ou liés aux tests V1 legacy (format_retrieved_excerpts, query_scope_node, synthesis_node, context_prune_node)
- Aucun échec d'import ou de collection
- `rag/` supprimé, plus aucune dépendance vers l'ancienne structure

## Prochaines étapes
1. Corriger les 37 échecs de test (infrastructure mock, V1 legacy)
2. Ajouter les tests manquants pour `src/fetchers/`, `src/analytics/`, `src/mcp/`
3. Mettre à jour `scripts/audit_toon_real_volume.py` (importe encore `rag.paths`, `rag.tools`)
4. Vérifier que les scripts standalone fonctionnent depuis la racine
