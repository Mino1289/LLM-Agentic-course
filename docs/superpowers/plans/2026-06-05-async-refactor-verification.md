# Étape 5 — Rapport de vérification (Refactor Asynchrone)

**Date** : 2026-06-05
**Branche** : `etape2`
**Spec** : `docs/superpowers/specs/2026-06-05-async-refactor-design.md`
**Plan** : `docs/superpowers/plans/2026-06-05-async-refactor.md`

## Résumé

ÉTAPE 5 convertit le graphe LangGraph Finance de **synchrone** à **asynchrone** (asyncio) sans casser la compatibilité ascendante. Le moteur est maintenant non-bloquant et prêt pour l'étape 6 (streaming token-par-token) et la migration UI.

**5 commits TDD** sur `etape2` :
- G1 (red) : `47fec22` — 14 tests rouges async nodes + to_thread wrapping
- G2 (impl) : `2699ac1` — 8 nœuds convertis `async def` + `asyncio.to_thread`
- G3 (red) : `dab39d1` — 7 tests rouges arun/astream + délégation
- G4 (impl) : `5b0b901` — double API `run()`/`arun()`/`astream()` sur `FinanceLangGraphAgent`
- G5 (verify) : ce rapport + 3 tests e2e (ce commit)

## Résultats des tests

```
$ .venv/bin/python -m unittest discover tests
Ran 87 tests in 0.115s
OK
```

**87/87 tests verts** :
- 63 tests hérités (ÉTAPES 1-4) — tous adaptés pour `asyncio.run()` autour des appels `async def` nodes, **aucun supprimé**
- 14 nouveaux `test_async_nodes.py` (structure + `to_thread` wrapping)
- 7 nouveaux `test_async_stream.py` (3 structure + 4 behavior)
- 3 nouveaux `test_async_e2e.py` (smoke test end-to-end)

## Mapping des commits

| Commit | Type | Description | Tests cumulés |
|--------|------|-------------|---------------|
| `47fec22` | 🔴 red | async nodes + to_thread | 63 (existants) + 14 rouges |
| `2699ac1` | 🟢 impl | 8 nœuds async | 77 verts |
| `dab39d1` | 🔴 red | arun + astream | 77 + 7 rouges |
| `5b0b901` | 🟢 impl | double API FinanceLangGraphAgent | 84 verts |
| (ce commit) | 🔵 verify | e2e + rapport | 87 verts |

## Nœuds convertis

| Fichier | Fonctions asyncifiées | Appels wrappés via `to_thread` |
|---------|----------------------|-------------------------------|
| `prepare_node.py` | `prepare_query_node` | (aucun — regex sync) |
| `memory_nodes.py` | `memory_read_node`, `memory_write_node`, `context_prune_node`, `gc_node` | `memory_store.{get_summary,get_window,get_or_create,append_turn,update_summary,trim_turns,remember_chunk,is_duplicate_chunk}` + `provider.generate` (summary) |
| `agent_nodes.py` | `agent_node`, `tools_node`, `finalize_from_agent_state` | `provider.invoke_with_tools`, `execute_tool` |
| `decompose_node.py` | `decompose_query`, `decompose_query_node` | `provider.generate` |
| `retrieval_node.py` | `multi_retrieve_node`, `_retrieve_with_fallbacks` | `rag.retrieve`, `rag._deduplicate_indices` |
| `rerank_node.py` | `rerank_node`, `_balanced_rerank_indices` | `rag._rerank` |

**Pattern uniforme** : `def` → `async def` + `await asyncio.to_thread(blocking_call, ...)`. Là où c'est possible, `asyncio.gather()` parallélise des I/O indépendants (e.g. `memory_read_node` : `get_summary` et `get_window` en parallèle).

## FinanceLangGraphAgent — double API

```python
class FinanceLangGraphAgent:
    # Sync (backwards compat — Studio, scripts, tests sync)
    def run(self, query, conversation_id=None, messages=None) -> GraphState:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop and running_loop.is_running():
            raise RuntimeError(
                "agent.run() ne peut pas être appelé depuis une boucle asyncio. "
                "Utilisez 'await agent.arun(...)' à la place."
            )
        return asyncio.run(self.arun(query, conversation_id, messages))

    # Async (UI, scripts async, streaming)
    async def arun(self, query, conversation_id=None, messages=None) -> GraphState:
        initial_state = self._initial_state(query, conversation_id, messages)
        return await self.graph.ainvoke(initial_state)

    # Streaming (UI réactive sur tool events)
    async def astream(self, query, conversation_id=None, messages=None):
        initial_state = self._initial_state(query, conversation_id, messages)
        async for event in self.graph.astream_events(initial_state, version="v2"):
            yield event
```

## Critères PRD §5.4

| Critère | Statut | Notes |
|---------|--------|-------|
| Graphe async-compatible (`ainvoke`/`astream_events`) | ✅ | Compilé en détectant les `async def` via `inspect.iscoroutinefunction` |
| Outils lourds non-bloquants via `asyncio.to_thread` | ✅ | `provider.*`, `rag.retrieve`, `rag._rerank`, `execute_tool`, `memory_store.*` tous wrappés |
| Double API `run()` / `arun()` (backwards compat) | ✅ | 63 tests existants verts sans modification de leur contrat externe (juste `asyncio.run` ajouté) |
| `astream_events(version="v2")` opérationnel | ✅ | `astream()` yield les events — testé via mock + e2e |
| Tests async isolés via `unittest.IsolatedAsyncioTestCase` (stdlib) | ✅ | Aucune nouvelle dépendance ajoutée |

## Tests async (3 suites, 24 tests)

### `test_async_nodes.py` (14 tests)

**Structure** (10) : `inspect.iscoroutinefunction` retourne `True` pour les 10 nœuds/fonctions async.

**Thread wrapping** (4) : mock `asyncio.to_thread` et vérifie que :
- `agent_node` wrap `provider.invoke_with_tools`
- `tools_node` wrap `execute_tool`
- `memory_read_node` wrap `memory_store.get_summary` ou `.get_window`
- `multi_retrieve_node` wrap `rag.retrieve`

### `test_async_stream.py` (7 tests)

**Structure** (3) : `arun` est coroutine, `astream` est async generator, `run` reste sync.

**Behavior** (4) :
- `test_run_delegates_to_asyncio_run_via_thread` — appelle `run()` dans un thread (hors loop), vérifie qu'`asyncio.run` est invoqué.
- `test_arun_returns_state_via_ainvoke` — `arun()` retourne le state final via `graph.ainvoke`.
- `test_astream_yields_chain_events` — `astream()` yield au moins 1 `on_chain_start`.
- `test_run_raises_in_running_loop` — `run()` depuis une loop async lève une `RuntimeError` claire.

### `test_async_e2e.py` (3 tests)

Vrai agent (pas mocké), rag stub minimal :
- `test_arun_returns_state` — `await agent.arun("Question test async")` retourne un dict avec `conversation_id`.
- `test_astream_yields_events` — `agent.astream("Question streaming")` yield ≥ 1 event.
- `test_arun_preserves_conversation_id` — `conversation_id` est propagé.

## Modifications des tests existants (ÉTAPES 1-4)

7 tests existants ont été adaptés pour envelopper les appels aux nœuds async avec `asyncio.run(...)` :

| Fichier | Tests adaptés |
|---------|-------------|
| `test_regressions.py` | `RetrievalTests` (2), `PrepareTests` (1), `RerankTests` (1), `ContextPruneTests` (1), `AgentToolsTests` (1) |
| `test_tool_events_lifecycle.py` | 4 tests (tous appellent `tools_node`) |

Le contrat externe des nœuds est inchangé du point de vue du **résultat** (mêmes clés, mêmes types). Seul le mécanisme d'invocation passe de `node(agent, state)` à `asyncio.run(node(agent, state))`.

## Décisions techniques

| Choix | Rationale |
|-------|-----------|
| **Outils sync wrappés** (pas réécrits) | YAGNI — `asyncio.to_thread` suffit, on évite de réécrire 5 outils + `HybridRAG` + `LLMProvider` |
| **Provider sync inchangé** | Étape 6 ajoutera `AsyncOpenAI`/`google.genai` async si besoin pour streaming token |
| **UI Streamlit reste sur `run()` sync** | Migration vers `astream` = étape 6 (réduit scope, focus moteur) |
| **`asyncio.run` + check running_loop** | Évite `RuntimeError` cryptique de `asyncio.run` ; UX claire |
| **`asyncio.gather` pour I/O parallèles** | `memory_read_node` : get_summary + get_window en parallèle ; `context_prune_node` : is_duplicate_chunk × N en parallèle ; `memory_write_node` : 2 append_turn en parallèle ; `gc_node` : 2 updates + N remember_chunk en parallèle |

## Hors scope (volontairement)

- ❌ Vrai streaming token LLM (étape 6)
- ❌ Réécriture du provider en `AsyncOpenAI` / `google.genai` async (YAGNI)
- ❌ Migration UI Streamlit vers `agent.astream` (réduit scope, focus moteur)
- ❌ Multi-agent orchestration (autre étape)
- ❌ Migration des 5 outils vers async natif (YAGNI — `to_thread` suffit)

## Risques identifiés et mitigations

| Risque | Mitigation |
|--------|-----------|
| `asyncio.run` dans loop existante | Check explicite + message clair pointant vers `await agent.arun()` |
| `@traceable` sur fonction async | `langsmith.traceable` supporte async depuis 0.1.27 — pas testé runtime (pas dans cette étape) |
| Détection async LangGraph | `graph.compile()` utilise `inspect.iscoroutinefunction` — validé par les 87 tests |
| `astream_events(version="v2")` schema | Test e2e vérifie qu'au moins 1 event est yield avec la bonne forme |

## Prochaines étapes (ÉTAPE 6+)

1. **ÉTAPE 6** : streaming token LLM via `AsyncOpenAI` / `google.genai` ; intégration dans `astream()` avec events `on_llm_new_token`.
2. **Migration UI** : remplacer `agent.run()` par `agent.astream()` dans `ui/app_rag.py` ; affichage progressif des tool events.
3. **Tracing async** : valider que `@traceable` capture bien les events de span pour les fonctions `async def`.

## Fichiers modifiés

```
M  rag/nodes/agent_nodes.py            (3 functions async)
M  rag/nodes/decompose_node.py         (2 functions async)
M  rag/nodes/memory_nodes.py           (4 functions async)
M  rag/nodes/prepare_node.py           (1 function async)
M  rag/nodes/rerank_node.py            (2 functions async)
M  rag/nodes/retrieval_node.py         (2 functions async)
M  rag/langgraph_flow.py               (run/arun/astream)
M  tests/test_async_nodes.py           (NEW, 14 tests)
M  tests/test_async_stream.py          (NEW, 7 tests)
M  tests/test_async_e2e.py             (NEW, 3 tests)
M  tests/test_regressions.py           (7 tests adapted with asyncio.run)
M  tests/test_tool_events_lifecycle.py (4 tests adapted with asyncio.run)
```

## Conclusion

✅ **ÉTAPE 5 livrée et vérifiée**. Le moteur est maintenant :
- **Non-bloquant** (tous les I/O via `asyncio.to_thread` + `asyncio.gather`)
- **Streamable** (`agent.astream()` yield tool events pour UI réactive)
- **Backwards compatible** (`agent.run()` sync, 63 tests existants verts)
- **Testé** (24 nouveaux tests, 87/87 verts)
- **Prêt pour étape 6** (streaming token + migration UI)
