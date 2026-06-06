# Étape 4 — Architecture Async & Multi-Agent Ready

**Date** : 2026-06-05
**Branche** : `etape2`
**Statut** : Design validé, en attente d'implémentation TDD

## Contexte et objectifs

L'agent LangGraph actuel expose 5 outils à l'LLM. Trois problèmes bloquent l'évolution vers une architecture async / multi-agent :

1. **Typage permissif** : les schémas d'outils sont des dicts JSON non validés. Des args malformés passent silencieusement.
2. **Couplage implicite** : `validate_claims_tool` reçoit son contexte RAG via un canal caché `rag_context=…` threadé par `tools_node`.
3. **tool_events plats** : un seul dict par appel, pas d'état `running`/`completed`/`failed`. L'UI ne peut pas afficher de spinner ni diagnostiquer un échec.

**Objectif** : poser les fondations typées et traçables nécessaires aux étapes suivantes (multi-agent, async).

## Décisions validées

| # | Décision | Choix |
|---|----------|-------|
| 1 | Typage strict (PRD §3.1) | **Pydantic BaseModel** par outil (Option 2) |
| 2 | Isolation contexte (PRD §3.2) | **Contexte explicite via args Pydantic** (Option 1) |
| 3 | tool_events (PRD §3.3) | **Lifecycle running/completed/failed** (Option 1) |
| 4 | Schéma LLM vs runtime | **Schéma public minimal + injection runtime** (chunks/metadatas invisibles au LLM) |
| 5 | Emplacement schémas | **Nouveau `rag/tool_schemas.py`** |
| 6 | Type `ToolEvent` | **`TypedDict`** (cohérent avec `GraphState`) |

## Architecture cible

```
                       ┌────────────────────────────────────┐
                       │   rag/tool_schemas.py (NEW)        │
                       │   - SecFilingsRAGArgs              │
                       │   - MarketPriceArgs                │
                       │   - ValidateClaimsLLMArgs (LLM)    │
                       │   - ValidateClaimsArgs (full)      │
                       │   - SimulatePortfolioArgs          │
                       │   - ExportReportArgs               │
                       │   - Inherit BaseModel              │
                       └──────────────┬─────────────────────┘
                                      │ Pydantic validation
                                      ▼
┌───────────────────┐    ┌────────────────────────────────────┐
│ GraphState        │    │   rag/tools.py                     │
│  tool_events:     │◄───┤   - run_*(args: BaseModel)        │
│  list[ToolEvent]  │    │   - get_tool_definitions()         │
│  (TypedDict)      │    │   - execute_tool(..., state)       │
│  final_chunks     │    └──────────────┬─────────────────────┘
│  final_metadatas  │                   │
└────────┬──────────┘                   │
         │                              │
         │ state read/write             │
         ▼                              ▼
┌──────────────────────────────────────────────────────────┐
│   rag/nodes/agent_nodes.py : tools_node                  │
│   1. emit ToolEvent(status=running)                      │
│   2. parse LLM args (JSON) → ValidateClaimsLLMArgs       │
│   3. resolve injected args from state                    │
│   4. instantiate full BaseModel (e.g. ValidateClaimsArgs)│
│   5. execute tool → result or exception                  │
│   6. emit ToolEvent(status=completed|failed)             │
│   7. update state (final_chunks, stats, ...)             │
└──────────────────────────────────────────────────────────┘
```

## Composants détaillés

### `rag/tool_schemas.py` (nouveau)

```python
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class SecFilingsRAGArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    tickers: list[str] | None = None
    years: list[str] | None = None
    doc_types: list[str] | None = None

    @field_validator("tickers", "years", "doc_types", mode="before")
    @classmethod
    def _accept_string_or_list(cls, v):
        # LLM may return a comma-separated string
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class MarketPriceArgs(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class ValidateClaimsLLMArgs(BaseModel):
    """Schéma PUBLIC envoyé à OpenAI — ne contient que ce que le LLM fournit."""
    claims: list[str] = Field(..., min_length=1)


class ValidateClaimsArgs(BaseModel):
    """Modèle RUNTIME complet — chunks/metadatas sont injectés par tools_node."""
    claims: list[str] = Field(..., min_length=1)
    chunks: list[str] = Field(default_factory=list)
    metadatas: list[dict[str, Any]] = Field(default_factory=list)


class SimulatePortfolioArgs(BaseModel):
    allocations: dict[str, float] = Field(..., min_length=1)
    notional_usd: float = Field(default=100_000, gt=0, le=1_000_000)


class ExportReportArgs(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    format: Literal["md", "pdf"] = "md"
```

### `rag/nodes/state.py` (modifié)

```python
from typing import Literal, TypedDict

class ToolEvent(TypedDict, total=False):
    id: str
    tool: str
    status: Literal["running", "completed", "failed"]
    started_at: str          # ISO UTC
    finished_at: str         # ISO UTC
    args: dict[str, Any]     # args validés côté serveur
    args_summary: str        # résumé pour l'UI
    result: dict[str, Any] | None
    error: str | None


class GraphState(TypedDict, total=False):
    # ... (clés existantes)
    tool_events: list[ToolEvent]  # type tightené
```

### `rag/tools.py` (refactor)

**Avant** :
```python
def run_validate_claims(agent, claims, chunks, metadatas) -> dict: ...
def execute_tool(agent, name, arguments, *, rag_context=None) -> dict: ...
def get_tool_definitions() -> list[dict]: ...
```

**Après** :
```python
def run_validate_claims(args: ValidateClaimsArgs, *, agent: Any) -> dict: ...
def run_sec_filings_rag(args: SecFilingsRAGArgs, *, agent: Any) -> dict: ...
def run_market_price_tool(args: MarketPriceArgs, *, agent: Any) -> dict: ...
def run_simulate_portfolio(args: SimulatePortfolioArgs) -> dict: ...
def run_export_investment_report(args: ExportReportArgs) -> dict: ...

def execute_tool(
    name: str,
    args: BaseModel,            # toujours un BaseModel validé
    *,
    agent: Any,
    state: dict[str, Any] | None = None,  # pour injection contexte
) -> dict[str, Any]: ...

def get_tool_definitions() -> list[dict[str, Any]]:
    """Génère le schéma OpenAI/Gemini. Pour validate_claims_tool, utilise
    ValidateClaimsLLMArgs.model_json_schema() (LLM ne voit que 'claims')."""
```

### `rag/nodes/agent_nodes.py` (refactor `tools_node`)

```python
@traceable(name="tools_node")
def tools_node(agent: Any, state: GraphState) -> GraphState:
    lc_messages = list(state.get("lc_messages") or [])
    pending: list[ToolCall] = state.get("pending_tool_calls") or []
    stats = dict(state.get("stats") or {})
    final_chunks = list(state.get("final_chunks") or [])
    final_metadatas = list(state.get("final_metadatas") or [])
    price_context = state.get("price_context", "")
    report_artifacts = list(state.get("report_artifacts") or [])
    tool_events = list(state.get("tool_events") or [])

    for tc in pending:
        event_id = str(uuid.uuid4())
        # 1. parse + emit running
        llm_args = _parse_llm_args(tc)
        event: ToolEvent = {
            "id": event_id,
            "tool": tc.name,
            "status": "running",
            "started_at": _now_utc(),
            "args": _safe_dict(llm_args),
            "args_summary": _summarize_tool_args(tc.name, tc.arguments),
        }
        tool_events.append(event)
        # 2. resolve injected + validate
        try:
            full_args = _resolve_full_args(tc.name, llm_args, state)
        except ValidationError as e:
            event["status"] = "failed"
            event["error"] = f"args_validation: {e}"
            event["finished_at"] = _now_utc()
            continue
        # 3. execute
        try:
            result = execute_tool(tc.name, full_args, agent=agent, state=state)
        except Exception as e:
            event["status"] = "failed"
            event["error"] = f"execution: {e}"
            event["finished_at"] = _now_utc()
            continue
        # 4. completed
        event["status"] = "completed"
        event["result"] = _safe_result(result)
        event["finished_at"] = _now_utc()
        # 5. update state side-effects
        ... (logique rag/price/export/validate/simulate)

    return {...}
```

### Fonctions helper

```python
def _parse_llm_args(tc: ToolCall) -> dict[str, Any]:
    try:
        return json.loads(tc.arguments or "{}")
    except json.JSONDecodeError:
        return {}

def _resolve_full_args(
    tool_name: str, llm_args: dict, state: GraphState
) -> BaseModel:
    if tool_name == "validate_claims_tool":
        llm = ValidateClaimsLLMArgs.model_validate(llm_args)
        return ValidateClaimsArgs(
            claims=llm.claims,
            chunks=list(state.get("final_chunks") or []),
            metadatas=list(state.get("final_metadatas") or []),
        )
    if tool_name == "sec_filings_rag_tool":
        return SecFilingsRAGArgs.model_validate(llm_args)
    if tool_name == "market_price_tool":
        return MarketPriceArgs.model_validate(llm_args)
    if tool_name == "simulate_portfolio_tool":
        return SimulatePortfolioArgs.model_validate(llm_args)
    if tool_name == "export_investment_report_tool":
        return ExportReportArgs.model_validate(llm_args)
    raise ValueError(f"Unknown tool: {tool_name}")
```

## Flux de données — exemple `validate_claims_tool`

```
LLM <tool_call> {"name": "validate_claims_tool", "arguments": '{"claims": ["MSFT supply chain risk", ...]}'}
                                  ↓
                            tools_node
                                  ↓
_parse_llm_args → {"claims": [...]}        (étape 1)
                                  ↓
ValidateClaimsLLMArgs.model_validate(llm_args)
                                  ↓
{claims: ["MSFT supply chain risk"]}       (validation LLM schema)
                                  ↓
_resolve_full_args → lit state["final_chunks"] + ["final_metadatas"]
                                  ↓
ValidateClaimsArgs(claims=..., chunks=state.chunks, metadatas=state.metadatas)
                                  ↓
execute_tool("validate_claims_tool", full_args, agent=agent, state=state)
                                  ↓
run_validate_claims(args, agent=agent) → {"validations": [...], "stats": {...}}
                                  ↓
ToolEvent.status = "completed", result = {"text": ..., "validations": ...}
```

## Tests rouges (TDD)

### F1 — schémas Pydantic (6 tests)

| Test | Vérifie |
|------|---------|
| `test_sec_filings_args_rejects_missing_query` | Pydantic `ValidationError` sans `query` |
| `test_sec_filings_args_accepts_string_list_for_tickers` | `tickers: "MSFT,NVDA"` accepté |
| `test_market_price_args_validates_date_format` | `start_date: "2024-01-01"` ok, `"01/01/2024"` rejeté |
| `test_validate_claims_llm_args_excludes_chunks` | Schéma JSON = `{claims}` seul (pas de `chunks`) |
| `test_validate_claims_args_requires_chunks_field` | `ValidateClaimsArgs` valide le champ `chunks` (list[str]) |
| `test_simulate_portfolio_args_caps_notional` | `notional_usd > 1_000_000` rejeté |

### F3 — tool_events lifecycle (3 tests)

| Test | Vérifie |
|------|---------|
| `test_tool_event_running_then_completed` | tools_node émet 2 events (running, completed) avec timestamps ordonnés |
| `test_tool_event_running_then_failed_on_exception` | execute_tool raise → event.status = "failed", event.error non vide |
| `test_tool_event_running_then_failed_on_validation` | Pydantic `ValidationError` → event.status = "failed" sans appeler execute_tool |

## Critères PRD

- ✅ §3.1 typage strict (Pydantic)
- ✅ §3.2 isolation contexte (args explicites, injection runtime)
- ✅ §3.3 tool_events lifecycle (running/completed/failed)
- ✅ §5.3 tests async-ready (TypedDict sérialisable, BaseModel sérialisable)

## Hors scope

- ❌ Refactor `asyncio` (les tools restent sync)
- ❌ Multi-agent orchestration (PRD §3.4)
- ❌ Persistence tool_events en base (UI display only)
- ❌ Rétrocompatibilité `execute_tool(..., rag_context=...)` — supprimé
- ❌ Migration des anciens tests qui mockent `execute_tool(agent, name, args, rag_context=…)` — adaptés

## Commits prévus

| # | Type | Fichier(s) | Commit |
|---|------|------------|--------|
| F1 | 🔴 red | `tests/test_tool_schemas.py` (nouveau) | `test(etape4): red tests for Pydantic tool schemas` |
| F2 | 🟢 impl | `rag/tool_schemas.py` (nouveau) + `requirements.txt` (ajout `pydantic`) | `feat(etape4): Pydantic tool args (BaseModel per tool)` |
| F3 | 🔴 red | `tests/test_tool_events_lifecycle.py` (nouveau) | `test(etape4): red tests for tool_events running/completed/failed` |
| F4 | 🟢 impl | `rag/nodes/state.py` + `rag/nodes/agent_nodes.py` + `rag/tools.py` (refactor) + adaptation des tests existants | `refactor(etape4): tool_events lifecycle + injected args` |
| F5 | 🔵 verify | rapport | `chore(etape4): verification report` |

## Risques identifiés

1. **Régressions tests existants** : `test_agent_tool_loop_mocked` (test_regressions.py:477) mocke `execute_tool` avec l'ancienne signature. À adapter.
2. **Pydantic manquant** : ajouter `pydantic` à `requirements.txt` (transitive dep of `openai` mais à expliciter).
3. **`get_tool_definitions` est appelé 2x** : une fois par `agent_node` (passé à OpenAI), une fois exporté pour debug. Le rendu doit être cohérent.
4. **Schema Gemini** : `_tool_definitions_to_gemini` lit `tool.get("function", {}).get("parameters")` — doit fonctionner avec le schéma Pydantic JSON Schema.

## Prochaines étapes

1. ✅ Design validé par l'utilisateur (decisions 1-6)
2. ⏭️ Écrire le plan d'implémentation détaillé via skill `writing-plans`
3. ⏭️ Exécuter F1-F5 en TDD strict
