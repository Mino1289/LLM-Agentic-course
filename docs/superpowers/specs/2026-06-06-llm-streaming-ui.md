# Étape 6 — Streaming Tokens LLM + Refactor UI

**Date** : 2026-06-05
**Branche** : `etape2`
**Statut** : Design validé, en attente d'implémentation TDD

## Contexte et objectifs

L'UI actuelle (`ui/app_rag.py:409-451`) appelle `agent.run()` dans un `st.spinner` et n'affiche la réponse qu'à la fin. Conséquences :
- L'utilisateur attend 5-30s sans signal de progression.
- Aucun feedback nœud/outil en temps réel.
- Aucun streaming token LLM.
- `LLMProvider.generate_stream` (ligne 325) est un **faux stream** (yield unique du texte complet).

**Objectif** : streamer token-par-token la réponse finale de l'agent, afficher en temps réel les statuts nœuds/outils dans l'UI Streamlit, capturer le `GraphState` final via un event `on_graph_end` custom (pas de double appel).

## Décisions validées

| # | Décision | Choix |
|---|----------|-------|
| 1 | Providers streaming | OpenAI + GitHub Models via `AsyncOpenAI` ; Gemini = fallback non-streaming (réponse complète à la fin) |
| 2 | Scope | Agent final uniquement (dernier `agent_node` retournant `answer`) ; NLI judge, summary, decompose restent non-streamés |
| 3 | Bridge Streamlit sync | Helper `run_stream()` qui appelle `asyncio.run(consume_astream(...))` ; chaque yield → `container.markdown(...)` / `container.status(...)` |
| 4 | Buffer tokens | Par mot : accumuler jusqu'à ` `, `\n`, `.`, `,` ; évite le flicker Streamlit |

## Architecture cible

```
┌─────────────────────────────────────────────────────────────────┐
│  UI Streamlit (sync)                                            │
│  with st.chat_message("assistant"):                             │
│      text_container = st.empty()                                │
│      status_container = st.status("⏳ Analyse...")              │
│      final_state = run_stream(agent, query, ...)                │
│      # ↳ helper qui appelle asyncio.run(consume_astream)       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  FinanceLangGraphAgent.astream()                                │
│  async generator qui yield :                                    │
│    • on_chain_start(name) : nœud démarre                        │
│    • on_chain_end(name) : nœud termine                          │
│    • on_tool_start(tool) : outil démarre                        │
│    • on_tool_end(tool) : outil termine                          │
│    • on_llm_token(token) : token LLM (renommé on_chat_model)    │
│    • on_graph_end(state) : state final (event custom)           │
└────────────────┬────────────────────────────────────────────────┘
                 │ graph.astream_events(state, version="v2")
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Graphe LangGraph (async)                                       │
│  prepare → memory_read → agent ⇄ tools → finalize → ...        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  agent_node (modifié) : utilise ainvoke_with_tools_stream       │
│  Autres nœuds : inchangés (to_thread sur provider sync)         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLMProvider (sync inchangé + nouvelles méthodes async) :       │
│  - agenerate_stream(prompt) -> AsyncIterator[str]               │
│  - ainvoke_with_tools_stream(messages, tools) ->                 │
│    AsyncIterator[LLMStreamChunk]                                │
│  - AsyncOpenAI pour OpenAI + GitHub Models                      │
│  - genai.aio pour Gemini (fallback non-streaming acceptable)    │
└─────────────────────────────────────────────────────────────────┘
```

## Composants détaillés

### `rag/llm_provider.py` — async streaming

```python
@dataclass
class LLMStreamChunk:
    """Chunk incrémental d'un stream LLM."""
    delta: str = ""
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None

class LLMProvider:
    def __init__(self, config=None):
        # ... existant (sync) ...
        if self.config.provider in {"openai", "github_models"}:
            # NOUVEAU : client async en parallèle
            self.async_client = AsyncOpenAI(...)
        else:
            self.async_client = None

    async def agenerate_stream(self, prompt, system_prompt=None, ...):
        """Stream texte pur token-par-token."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}] if system_prompt else [{"role": "user", "content": prompt}]
        async for chunk in self.ainvoke_with_tools_stream(messages, tools=None, ...):
            if chunk.delta:
                yield chunk.delta

    async def ainvoke_with_tools_stream(self, messages, tools=None, ...):
        """Stream LLM. Yield LLMStreamChunk par chunk."""
        if self.config.provider == "gemini":
            async for chunk in self._ainvoke_gemini_stream(messages, tools, ...):
                yield chunk
            return
        async for chunk in self._ainvoke_openai_stream(messages, tools, ...):
            yield chunk

    async def _ainvoke_openai_stream(self, messages, tools, temperature, max_tokens):
        kwargs = {... "stream": True}
        response = await self.async_client.chat.completions.create(**kwargs)
        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            yield LLMStreamChunk(
                delta=delta.content or "",
                tool_call_delta=... if delta.tool_calls else None,
                finish_reason=choice.finish_reason,
            )
```

### `rag/nodes/agent_nodes.py` — utilise le stream async

```python
@traceable(name="agent_node")
async def agent_node(agent, state):
    # ... guard, iterations check ...
    lc_messages = _build_lc_messages(agent, state)
    text_parts = []
    tool_calls_dict = {}  # idx -> {id, name, args_parts}
    async for chunk in agent.rag.provider.ainvoke_with_tools_stream(
        lc_messages, tools=get_tool_definitions(), temperature=0.2, max_tokens=2500
    ):
        if chunk.delta:
            text_parts.append(chunk.delta)
        if chunk.tool_call_delta:
            # Accumulate tool calls by index
            ...
    final_text = "".join(text_parts)
    final_tool_calls = [...]  # rebuild
    # ... même logique de return qu'avant (tool_calls vs answer) ...
```

### `rag/langgraph_flow.py` — capture state final via `on_graph_end`

```python
async def astream(self, query, conversation_id=None, messages=None):
    initial_state = self._initial_state(query, conversation_id, messages)
    last_state = None
    async for event in self.graph.astream_events(initial_state, version="v2"):
        kind = event.get("event")
        # Renommer on_chat_model_stream → on_llm_token
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            delta = getattr(chunk, "content", "") or ""
            if not delta and isinstance(chunk, dict):
                delta = chunk.get("content", "")
            if delta:
                yield {"event": "on_llm_token", "token": delta}
                continue
        # Capturer le state dans chaque on_chain_end
        if kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, dict):
                last_state = output
        yield event
    # Event final avec state complet
    if last_state is not None:
        yield {"event": "on_graph_end", "state": last_state}
```

### `ui/app_rag.py` — helper `run_stream`

```python
def run_stream(agent, query, conversation_id, messages, text_container, status_container) -> dict:
    """Consomme agent.astream() via asyncio.run, met à jour les containers."""
    final_state = {}
    streamed_text = ""
    word_buffer = ""
    WORD_SEPARATORS = (" ", "\n", ".", ",")
    
    async def consume():
        nonlocal final_state, streamed_text, word_buffer
        async for event in agent.astream(query, conversation_id, messages):
            kind = event.get("event")
            if kind == "on_chain_start":
                name = event.get("name", "?")
                status_container.update(label=f"⏳ {name} en cours...")
            elif kind == "on_tool_start":
                tool = event.get("name", "?")
                status_container.update(label=f"⏳ Outil `{tool}` en cours...")
            elif kind == "on_tool_end":
                status_container.update(label="✅ Outil terminé")
            elif kind == "on_llm_token":
                token = event.get("token", "")
                word_buffer += token
                if any(sep in word_buffer for sep in WORD_SEPARATORS):
                    streamed_text += word_buffer
                    word_buffer = ""
                    text_container.markdown(streamed_text + "▌")
            elif kind == "on_graph_end":
                if word_buffer:
                    streamed_text += word_buffer
                    word_buffer = ""
                final_state = event.get("state", {})
    
    asyncio.run(consume())
    return final_state
```

## Tests rouges (TDD)

### H1 — `LLMProvider` async streaming (4 tests)

```python
class AsyncProviderStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_client_is_initialized_for_openai(self):
        provider = LLMProvider(LLMConfig(provider="openai", ...))
        self.assertIsNotNone(provider.async_client)
        self.assertIsInstance(provider.async_client, AsyncOpenAI)

    async def test_agenerate_stream_yields_text_deltas(self):
        # Mock async_client.chat.completions.create
        ...
        chunks = []
        async for token in provider.agenerate_stream("Hello"):
            chunks.append(token)
        self.assertGreater(len(chunks), 0)
        self.assertEqual("".join(chunks), "Hello world")

    async def test_ainvoke_with_tools_stream_yields_deltas(self):
        # Mock async_client pour yield LLMStreamChunk
        ...

    async def test_ainvoke_with_tools_stream_accumulates_tool_calls(self):
        # Mock pour répondre avec tool_calls
        ...
```

### H3 — `astream` events (3 tests)

```python
class AStreamTokenEventsTests(unittest.IsolatedAsyncioTestCase):
    async def test_astream_yields_on_llm_token_events(self):
        # Mock ainvoke_with_tools_stream pour yield "Hello" " " "world"
        ...
        tokens = []
        async for event in agent.astream("query"):
            if event.get("event") == "on_llm_token":
                tokens.append(event["token"])
        self.assertEqual("".join(tokens), "Hello world")

    async def test_astream_yields_on_graph_end_with_state(self):
        # Mock ainvoke_with_tools_stream + astream_events
        ...
        final = None
        async for event in agent.astream("query"):
            if event.get("event") == "on_graph_end":
                final = event["state"]
        self.assertIsInstance(final, dict)
        self.assertIn("answer", final)

    async def test_astream_final_state_has_tool_events(self):
        # Vérifie que le state final contient tool_events (pas de double appel)
        ...
```

### H5 — `run_stream` UI helper (3 tests)

```python
class RunStreamHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_run_stream_invokes_asyncio_run(self):
        # Mock agent.astream, container.status
        ...
        final = run_stream(...)
        self.assertIn("answer", final)

    def test_run_stream_flushes_word_buffer(self):
        # Mock agent.astream pour yield "Hello" " " "world"
        # Vérifier que le container.markdown est appelé avec "Hello " puis "Hello world"
        ...

    def test_run_stream_returns_state_from_on_graph_end(self):
        # Mock pour yield on_graph_end avec state
        ...
        final = run_stream(...)
        self.assertEqual(final.get("answer"), "test")
```

**Total** : ~10 nouveaux tests. **Cumul** : ~97 verts (87 actuels + 10).

## Commits prévus

| # | Type | Fichiers | Commit |
|---|------|----------|--------|
| H1 | 🔴 red | `tests/test_provider_stream.py` | `test(etape6): red tests for LLMProvider async streaming` |
| H2 | 🟢 impl | `rag/llm_provider.py` | `feat(etape6): LLMProvider async streaming via AsyncOpenAI + genai.aio` |
| H3 | 🔴 red | `tests/test_async_stream.py` | `test(etape6): red tests for astream on_llm_token + on_graph_end` |
| H4 | 🟢 impl | `rag/langgraph_flow.py`, `rag/nodes/agent_nodes.py` | `feat(etape6): agent_node uses async stream; astream yields on_llm_token + on_graph_end` |
| H5 | 🔴 red | `tests/test_ui_stream.py` | `test(etape6): red tests for run_stream UI helper` |
| H6 | 🟢 impl | `ui/app_rag.py` | `feat(etape6): UI consumes astream via run_stream + per-word buffer` |
| H7 | 🔵 verify | `docs/superpowers/plans/2026-06-06-llm-streaming-ui-verification.md` | `chore(etape6): verification report` |

## Critères PRD §5.5

- ✅ Streaming token-par-token (OpenAI + GitHub via `AsyncOpenAI`)
- ✅ UI affiche les statuts nœuds/outils en temps réel
- ✅ Capture state final sur `on_graph_end` (pas de double appel)
- ✅ Buffer par mot (anti-flicker)
- ✅ Gemini fallback non-streaming (régression OK)
- ✅ Backwards compat : `agent.run()` reste sync
- ✅ Tests async (mock `AsyncOpenAI` via `AsyncMock`)

## Hors scope

- ❌ Streaming des sous-LLMs (NLI, summary, decompose) — décision 2
- ❌ Migration Gemini vers `genai.aio` (fallback non-stream OK) — décision 1
- ❌ Cancellation token (UI stop button) — feature future
- ❌ `@st.fragment` (decision 3) — `asyncio.run` simple
- ❌ Multi-async-streaming des tool_calls partiels

## Risques

1. **Event schema `on_chat_model_stream`** : structure `event["data"]["chunk"]` peut varier. Mapping dans `astream()`.
2. **Gemini `genai.aio`** : API jeune. Fallback non-streaming = régression safe.
3. **`asyncio.run` dans Streamlit** : si UI dans loop existante → `RuntimeError`. À détecter.
4. **Token buffer trop agressif** : flush de sécurité tous les 50 chars.
5. **`AsyncOpenAI` non disponible** : lib `openai` >= 1.0 l'inclut. Vérifier version.
6. **Performance re-render Streamlit** : throttle 100ms minimum si trop lent.
7. **Tests `AsyncOpenAI` mocké** : `AsyncMock` + async iterator.
