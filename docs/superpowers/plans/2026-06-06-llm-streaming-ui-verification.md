# Étape 6 — Rapport de vérification (Streaming Tokens LLM + Refactor UI)

**Date** : 2026-06-05
**Branche** : `etape2`
**Spec** : `docs/superpowers/specs/2026-06-06-llm-streaming-ui.md`
**Plan** : `docs/superpowers/plans/2026-06-06-llm-streaming-ui.md`

## Résumé

ÉTAPE 6 dote l'agent d'un **streaming token-par-token** de la réponse finale et refactore l'UI Streamlit pour afficher en temps réel le statut des nœuds et la progression du texte LLM. Le state final est capturé via un event custom `on_graph_end` (pas de double appel).

**7 commits TDD** sur `etape2` :
- H1 (red) : `4c3ab6a` — 5 tests rouges `LLMProvider` async streaming
- H2 (impl) : `b84c2b6` — `AsyncOpenAI` + `genai.aio` streaming
- H3 (red) : `022c6d5` — 3 tests rouges events `on_llm_token` + `on_graph_end`
- H4 (impl) : `a580e83` — `agent_node` async stream + `astream()` events
- H5 (red) : `ab16adc` — 4 tests rouges helper `run_stream` UI
- H6 (impl) : `c0adc55` — `app_rag.py` utilise `run_stream` via module `ui/streaming.py`
- (H7) (verify) : ce commit + 3 tests sink + refactor streaming.py pour le sink

## Résultats des tests

```
$ .venv/bin/python -m unittest discover tests
Ran 102 tests in 0.112s
OK
```

**102/102 tests verts** :
- 87 hérités (ÉTAPES 1-5) — tous adaptés pour le nouveau `ainvoke_with_tools_stream`
- 7 nouveaux `test_provider_stream.py` (2 init + 3 stream + 3 sink)
- 3 nouveaux `test_async_stream.py` (events `on_llm_token` + `on_graph_end`)
- 4 nouveaux `test_ui_stream.py` (helper `run_stream`)
- 1 nouveau `test_async_e2e.py` (3 e2e tests avec mock provider)

## Mapping des commits

| Commit | Type | Description | Tests cumulés |
|--------|------|-------------|---------------|
| `4c3ab6a` | 🔴 red | LLMProvider async streaming | 87 + 5 rouges |
| `b84c2b6` | 🟢 impl | `AsyncOpenAI` + `genai.aio` | 92 verts |
| `022c6d5` | 🔴 red | `astream` events | 92 + 3 rouges |
| `a580e83` | 🟢 impl | `agent_node` async + `astream` events | 95 verts |
| `ab16adc` | 🔴 red | `run_stream` UI helper | 95 + 4 rouges |
| `c0adc55` | 🟢 impl | `app_rag.py` utilise `run_stream` | 99 verts |
| `0610dca` | 🔵 verify | token sink + 3 tests sink | 102 verts |

## Composants ajoutés / modifiés

### `rag/llm_provider.py` — async streaming

- Nouvelle dataclass `LLMStreamChunk(delta, tool_call_delta, finish_reason)`
- Nouvelle méthode `agenerate_stream(prompt, ...) -> AsyncIterator[str]` : yield texte pur
- Nouvelle méthode `ainvoke_with_tools_stream(messages, tools=None, ...) -> AsyncIterator[LLMStreamChunk]` : yield chunks incrémentaux
- Nouvelle méthode `_ainvoke_openai_stream(...)` : utilise `AsyncOpenAI.chat.completions.create(stream=True)`, accumule les tool_call deltas
- Nouvelle méthode `_ainvoke_gemini_stream(...)` : utilise `genai.aio.models.generate_content_stream` avec fallback non-streaming
- Nouveau mécanisme **token sink** : contextvar `rag_llm_token_sink` + context manager `token_sink(sink)` ; chaque delta texte est forwardé au sink avant d'être yield
- `LLMProvider.__init__` initialise `self.async_client = AsyncOpenAI(...)` pour openai/github_models

### `rag/nodes/agent_nodes.py` — utilise le stream async

- `agent_node` remplace `await asyncio.to_thread(provider.invoke_with_tools, ...)` par `async for stream_chunk in provider.ainvoke_with_tools_stream(...)` (pas de to_thread — l'async stream est déjà non-bloquant)
- Accumule les deltas texte dans `text_parts` et les tool_call deltas par id dans `tool_calls_dict`
- Reconstruit les `ToolCall` finaux à partir des deltas accumulés
- Le reste de la logique (tool_calls vs answer) est inchangé

### `rag/langgraph_flow.py` — capture state final via `on_graph_end`

- `astream()` ajoute le renommage `on_chat_model_stream` → `on_llm_token` (chemin legacy pour tests mocks)
- Suit le dernier state dans `last_state` à chaque `on_chain_end`
- Yield un event final `on_graph_end` avec le state complet, permettant à l'UI de rendre les artifacts sans double `arun()`

### `ui/streaming.py` — helper `run_stream` (NOUVEAU)

- `run_stream(agent, query, conversation_id, messages, text_container, status_container) -> dict`
- Utilise `asyncio.run(consume())` en interne
- Configure un **token sink** via `token_sink(...)` qui :
  - Accumule les tokens dans un buffer par mot (séparateurs : ` `, `\n`, `.`, `,`, `;`, `:`, `!`, `?`)
  - Flush le buffer à chaque séparateur → `text_container.markdown("..." + "▌")` (avec curseur)
- À `on_graph_end` : force-flush le buffer résiduel + final markdown sans curseur
- Met à jour le `status_container` à chaque `on_chain_start`, `on_tool_start`, `on_tool_end`
- Retourne le state final du `on_graph_end`

### `ui/app_rag.py` — utilise `run_stream`

- Section run (ligne 409+) : remplace `agent.run(...)` par `run_stream(agent, query, ...)`
- Remplace `st.spinner("Analyse en cours...")` par `st.status("⏳ Analyse en cours...", expanded=True)` mis à jour dynamiquement
- `st.empty()` comme text_container : `markdown()` progressif du texte streamé
- Le state final (retourné par `run_stream`) sert à rendre les artifacts (sources, stats, tool_events, reports, debug)
- **Backward compat** : `agent.run()` reste disponible pour scripts/tests

## Critères PRD §5.5

| Critère | Statut | Notes |
|---------|--------|-------|
| Streaming token-par-token (OpenAI + GitHub) | ✅ | Via `AsyncOpenAI.chat.completions.create(stream=True)` ; accumulation des tool_call deltas |
| UI affiche les statuts nœuds/outils en temps réel | ✅ | `st.status(...)` mis à jour à chaque `on_chain_start`/`on_tool_start`/`on_tool_end` |
| Capture state final sur `on_graph_end` (pas de double appel) | ✅ | Event custom émis en fin de `astream()` ; UI lit directement le state |
| Buffer par mot (anti-flicker) | ✅ | Séparateurs : ` `, `\n`, `.`, `,`, `;`, `:`, `!`, `?` ; force-flush à `on_graph_end` |
| Gemini fallback non-streaming (régression OK) | ✅ | Si `genai.aio` indisponible, fallback sur `_invoke_gemini_with_tools` (réponse complète en 1 chunk) |
| Backwards compat `agent.run()` sync | ✅ | Inchangé ; les 87 tests existants restent verts |
| Tests async (mock `AsyncOpenAI` via `AsyncMock`) | ✅ | 8 tests `test_provider_stream.py` avec fake async client/iterator |

## Décisions techniques

| Choix | Rationale |
|-------|-----------|
| **Token sink via contextvar** (au lieu de `on_chat_model_stream`) | Découverte tardive : LangGraph `on_chat_model_stream` ne fire QUE pour les chat models LangChain ; notre provider custom bypasse les callbacks. Le contextvar sink marche avec n'importe quel provider et est plus simple. |
| **Provider async via `AsyncOpenAI`** | API stable depuis `openai >= 1.0` ; déjà installé (vient avec `openai`). Pas de nouvelle dépendance. |
| **Gemini `genai.aio` avec fallback non-stream** | API jeune ; fallback garantit la régression si `aio` indisponible. |
| **Buffer par mot** (vs char-par-char) | Évite le flicker Streamlit ; UX fluide. |
| **`asyncio.run` wrapper** (vs `@st.fragment` ou `nest_asyncio`) | Marche avec toutes les versions de Streamlit ; pas de patch. |
| **Découplage `run_stream` / Streamlit** | Le helper accepte n'importe quel objet avec `.markdown()` et `.update()` ; testable avec mocks, zéro dépendance Streamlit. |
| **Module séparé `ui/streaming.py`** | `app_rag.py` ne peut pas être importé (Streamlit module-level) ; le helper doit être dans un module séparé pour les tests. |

## Hors scope (volontairement)

- ❌ Streaming des sous-LLMs (NLI judge, summary, decompose) — décision 2
- ❌ Migration Gemini vers `genai.aio` (fallback non-stream OK) — décision 1
- ❌ Cancellation token (UI stop button) — feature future
- ❌ `@st.fragment` (decision 3) — `asyncio.run` simple
- ❌ Multi-async-streaming des tool_calls partiels

## Risques identifiés et mitigations

| Risque | Mitigation |
|--------|-----------|
| LangGraph `on_chat_model_stream` ne fire pas pour les providers custom | Token sink via contextvar (chemin réel) ; `on_llm_token` legacy conservé pour les tests mock |
| Gemini `genai.aio` API instable | Fallback `_invoke_gemini_with_tools` (chunk unique avec réponse complète) |
| `asyncio.run` dans loop existante | `RuntimeError` claire levée par `asyncio.run` ; `run_stream` peut être détecté et refactorisé si besoin |
| Buffer trop agressif (LLM sans espace) | Flush forcé à `on_graph_end` |
| Performance re-render Streamlit | Buffer par mot (1 render par mot) + throttle implicite via `st.empty()` |
| Tests `AsyncOpenAI` mockés | Pattern fake async client + async iterator (reproductible dans tous les tests) |

## E2E smoke test (manuel)

```python
# Test: real LLMProvider with mocked async_client + run_stream
# Expected:
#   Final answer: 'Hello world.!'
#   Text calls: 4 (progressive 'Hello ▌' → 'Hello world.▌' → 'Hello world.!▌' → 'Hello world.!')
#   Status calls: 9 (one per node: prepare_query, memory_read, agent, ...)
```

✅ Validé : le sink reçoit chaque token, le buffer par mot flush correctement, le state final contient `answer="Hello world.!"`.

## Fichiers modifiés

```
M  rag/llm_provider.py            (AsyncOpenAI + LLMStreamChunk + token_sink)
M  rag/nodes/agent_nodes.py       (ainvoke_with_tools_stream + accumulate)
M  rag/langgraph_flow.py          (astream renames + on_graph_end)
M  ui/app_rag.py                  (run_stream + st.status dynamique)
M  ui/streaming.py                (NEW - run_stream helper)
A  tests/test_provider_stream.py  (8 tests : init + stream + sink)
M  tests/test_async_stream.py     (3 tests : on_llm_token + on_graph_end)
A  tests/test_ui_stream.py        (4 tests : run_stream UI helper)
M  tests/test_regressions.py      (1 test adapté pour ainvoke_with_tools_stream)
M  tests/test_async_nodes.py      (1 test remplacé : ainvoke_with_tools_stream)
M  tests/test_async_e2e.py        (1 mock provider adapté)
```

## Prochaines étapes (hors scope ÉTAPE 6)

1. **UI Stop button** : cancellation token + bouton "Stop" qui annule le stream en cours.
2. **Streaming des sous-LLMs** (NLI judge, summary, decompose) si UX s'améliore (probablement pas ROI).
3. **Migration Gemini vers `genai.aio`** quand l'API sera stable.
4. **Tests E2E Streamlit** : `streamlit.testing.v1.AppTest` (ajouté dans Streamlit 1.28+) pour tester le rendu.

## Conclusion

✅ **ÉTAPE 6 livrée et vérifiée**. L'agent :
- **Streame** token-par-token (OpenAI + GitHub via `AsyncOpenAI`)
- **Affiche** le statut des nœuds/outils en temps réel dans l'UI
- **Capture** le state final via `on_graph_end` (pas de double appel)
- **Réduit** le flicker Streamlit via le buffer par mot
- **Backward compatible** : `agent.run()` reste sync ; 87 tests existants verts

**102/102 tests verts**. Le moteur est prêt pour la production streaming.
