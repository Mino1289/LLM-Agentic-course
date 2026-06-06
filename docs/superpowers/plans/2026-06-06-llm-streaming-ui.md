# Étape 6 — Plan d'implémentation TDD

**Date** : 2026-06-05
**Branche** : `etape2`
**Spec** : `docs/superpowers/specs/2026-06-06-llm-streaming-ui.md`

## Vue d'ensemble

7 commits TDD pour streamer token-par-token la réponse finale de l'agent et refactorer l'UI Streamlit.

| # | Type | Description | Fichiers | Tests |
|---|------|-------------|----------|-------|
| H1 | 🔴 red | Tests `agenerate_stream` / `ainvoke_with_tools_stream` | 1 nouveau test | +4 |
| H2 | 🟢 impl | `AsyncOpenAI` + `genai.aio` streaming | `rag/llm_provider.py` | 87 → 87 (mocks) |
| H3 | 🔴 red | Tests `astream` events `on_llm_token` + `on_graph_end` | 1 nouveau test | +3 |
| H4 | 🟢 impl | `agent_node` async stream + `astream` events | 2 fichiers | 94 verts |
| H5 | 🔴 red | Tests `run_stream` UI helper | 1 nouveau test | +3 |
| H6 | 🟢 impl | `app_rag.py` utilise `run_stream` | `ui/app_rag.py` | 97 verts |
| H7 | 🔵 verify | Rapport | 1 nouveau doc | 97 verts |

## H1 — Red tests : LLMProvider async streaming (4 tests)

**Fichier** : `tests/test_provider_stream.py`

```python
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

class LLMProviderAsyncInitTests(unittest.TestCase):
    def test_async_client_created_for_openai(self):
        os.environ["OPENAI_API_KEY"] = "fake"
        with patch("rag.llm_provider.AsyncOpenAI") as mock_async:
            from rag.llm_provider import LLMProvider, LLMConfig
            cfg = LLMConfig(provider="openai", chat_model="gpt-4o-mini",
                          embedding_model="text-embedding-3-small",
                          api_key="x", embedding_api_key="x")
            provider = LLMProvider(cfg)
            mock_async.assert_called_once()
            self.assertIsNotNone(provider.async_client)

    def test_no_async_client_for_gemini(self):
        # Gemini = fallback non-streaming
        ...


class LLMProviderAsyncStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_agenerate_stream_yields_text_deltas(self):
        # Mock async_client.chat.completions.create to return chunks
        ...

    async def test_ainvoke_with_tools_stream_yields_deltas(self):
        ...

    async def test_ainvoke_with_tools_stream_accumulates_tool_calls(self):
        ...
```

## H2 — Impl : LLMProvider async streaming

**Fichier** : `rag/llm_provider.py`

Modifications :
- Import `AsyncOpenAI` from `openai`
- Ajout dataclass `LLMStreamChunk`
- `LLMProvider.__init__` : créer `self.async_client = AsyncOpenAI(...)` pour OpenAI/GitHub
- 3 nouvelles méthodes : `agenerate_stream`, `ainvoke_with_tools_stream`, `_ainvoke_openai_stream`
- Gemini : utiliser `genai.aio.models.generate_content_stream` (avec fallback)
- `generate_stream` (sync existant) reste comme fallback (ne supprime pas, pour back-compat)

## H3 — Red tests : astream events (3 tests)

**Fichier** : `tests/test_async_stream.py` (ajout de 3 tests)

```python
class AStreamTokenEventsTests(unittest.IsolatedAsyncioTestCase):
    async def test_astream_yields_on_llm_token_for_each_token(self):
        # Mock agent_node via patch sur provider.ainvoke_with_tools_stream
        # pour yield "Hello" " " "world"
        ...

    async def test_astream_yields_on_graph_end_with_full_state(self):
        # Mock astream_events pour yield on_chain_end avec state
        ...

    async def test_astream_final_state_has_tool_events(self):
        # Vérifier que le state final contient tool_events
        ...
```

## H4 — Impl : agent_node async stream + astream events

**Fichiers** :
- `rag/nodes/agent_nodes.py` : `agent_node` appelle `ainvoke_with_tools_stream` au lieu de `invoke_with_tools` ; accumule `text_parts` et `tool_calls_dict`
- `rag/langgraph_flow.py` : `astream()` ajoute renommage `on_chat_model_stream → on_llm_token` et yield event `on_graph_end` à la fin avec state complet

## H5 — Red tests : run_stream UI helper (3 tests)

**Fichier** : `tests/test_ui_stream.py`

```python
class RunStreamHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_run_stream_invokes_asyncio_run(self):
        # Mock agent.astream
        ...

    def test_run_stream_flushes_word_buffer_on_separator(self):
        # Mock pour yield "Hello" " " "world"
        # Vérifier que le container.markdown est appelé avec "Hello " (avec séparateur)
        # puis "Hello world"
        ...

    def test_run_stream_returns_state_from_on_graph_end(self):
        # Mock pour yield on_graph_end avec state
        ...
```

## H6 — Impl : app_rag.py utilise run_stream

**Fichier** : `ui/app_rag.py`

Modifications :
- Ajouter helper `run_stream()` après les imports
- Section run (ligne 409+) : remplacer `agent.run(...)` par `run_stream(...)` qui retourne le state final
- Suppression du `st.spinner` au profit de `st.status(...)` mis à jour dynamiquement
- Rendu progressif du texte via `st.empty()` + `markdown`

## H7 — Verification

**Fichier** : `docs/superpowers/plans/2026-06-06-llm-streaming-ui-verification.md`

Contenus :
- 7 commits
- 97 tests verts (87 anciens + 10 nouveaux)
- Validation manuelle : lancer un agent avec `OPENAI_API_KEY=fake` et vérifier que `astream` yield les events attendus
- Critères PRD §5.5
- Note : l'UI Streamlit doit être testée manuellement (pas de test E2E Streamlit dans cette étape)
- Note : Gemini reste en fallback non-streaming (pas testé runtime sans clé API)

## Stratégie de tests

- **Tests sync existants (87)** : restent verts (pas de modif du contrat externe).
- **Tests async nouveaux (10)** : `unittest.IsolatedAsyncioTestCase` + `AsyncMock` pour mocker `AsyncOpenAI`.
- **Mock `AsyncOpenAI`** : `unittest.mock.AsyncMock` + async iterator (classe avec `__aiter__`/`__anext__`).
- **Vérif post-H6** : smoke test manuel via `streamlit run ui/app_rag.py` (hors CI).

## Critères de succès

- ✅ 97 tests verts
- ✅ `provider.ainvoke_with_tools_stream` est async generator
- ✅ `agent.astream` yield `on_llm_token` (1 par token) et `on_graph_end` (state complet)
- ✅ `run_stream` met à jour les containers Streamlit
- ✅ State final accessible sans double appel `arun()`
- ✅ Branche `etape2` propre, 7 nouveaux commits
