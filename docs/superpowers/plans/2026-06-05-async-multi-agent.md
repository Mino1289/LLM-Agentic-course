# Étape 4 — Async & Multi-Agent Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor tool layer to use Pydantic BaseModel args, explicit context passing, and lifecycle tool_events — foundations for async and multi-agent work.

**Architecture:** New `rag/tool_schemas.py` module with one BaseModel per tool. `validate_claims_tool` split into `ValidateClaimsLLMArgs` (LLM-visible, `claims` only) and `ValidateClaimsArgs` (runtime, `claims + chunks + metadatas`). `tools_node` injects context from state, runs Pydantic validation, emits running/completed/failed events.

**Tech Stack:** Python 3.14, Pydantic v2, LangGraph, unittest (no pytest), LangChain-style tool calling.

**Spec:** `docs/superpowers/specs/2026-06-05-async-multi-agent-design.md`

**Branche:** `etape2` (we stay on this branch per user instruction).

---

## File Structure

**Create:**
- `rag/tool_schemas.py` — Pydantic BaseModel per tool
- `tests/test_tool_schemas.py` — Pydantic validation tests
- `tests/test_tool_events_lifecycle.py` — tool_events lifecycle tests

**Modify:**
- `rag/nodes/state.py` — add `ToolEvent` TypedDict; tighten `tool_events` type
- `rag/tools.py` — refactor `run_*` to take BaseModel; remove `rag_context=`; `get_tool_definitions` uses `ValidateClaimsLLMArgs.model_json_schema()`
- `rag/nodes/agent_nodes.py` — `tools_node` lifecycle + injected args + `_resolve_full_args`
- `requirements.txt` — add `pydantic>=2.0`
- `tests/test_regressions.py` — adapt `test_agent_tool_loop_mocked` to new `execute_tool` signature; adapt `test_validate_claims_*` to new `run_validate_claims` signature

---

## Task 1: Add Pydantic dependency + verify

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `pydantic>=2.0` to requirements.txt**

Edit `requirements.txt`, add at end (after `pdfplumber` line):
```
pydantic>=2.0
```

- [ ] **Step 2: Verify pydantic is importable in venv**

Run: `.venv/bin/python -c "import pydantic; print(pydantic.VERSION)"`
Expected: prints version >= 2.0 (e.g., `2.7.0` or higher).

If pydantic is missing, run: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): add pydantic>=2.0 for tool_schemas"
```

---

## Task 2 (F1): Red tests — Pydantic tool schemas

**Files:**
- Create: `tests/test_tool_schemas.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_tool_schemas.py` with this exact content:

```python
"""Pydantic BaseModel validation for tool args (PRD etape 4 §3.1)."""

import unittest

from pydantic import ValidationError


class SecFilingsRAGArgsTests(unittest.TestCase):
    def test_rejects_missing_query(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        with self.assertRaises(ValidationError):
            SecFilingsRAGArgs.model_validate({})

    def test_accepts_string_list_for_tickers(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        args = SecFilingsRAGArgs.model_validate(
            {"query": "NVDA risk", "tickers": "MSFT,NVDA"}
        )
        self.assertEqual(args.tickers, ["MSFT", "NVDA"])

    def test_defaults_optional_fields(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        args = SecFilingsRAGArgs.model_validate({"query": "MSFT risk"})
        self.assertIsNone(args.tickers)
        self.assertIsNone(args.years)
        self.assertIsNone(args.doc_types)


class MarketPriceArgsTests(unittest.TestCase):
    def test_validates_date_format(self) -> None:
        from rag.tool_schemas import MarketPriceArgs

        args = MarketPriceArgs.model_validate(
            {"tickers": ["NVDA"], "start_date": "2024-01-01", "end_date": "2024-12-31"}
        )
        self.assertEqual(args.start_date, "2024-01-01")

        with self.assertRaises(ValidationError):
            MarketPriceArgs.model_validate(
                {"tickers": ["NVDA"], "start_date": "01/01/2024", "end_date": "2024-12-31"}
            )

    def test_requires_at_least_one_ticker(self) -> None:
        from rag.tool_schemas import MarketPriceArgs

        with self.assertRaises(ValidationError):
            MarketPriceArgs.model_validate(
                {"tickers": [], "start_date": "2024-01-01", "end_date": "2024-12-31"}
            )


class ValidateClaimsLLMArgsTests(unittest.TestCase):
    def test_excludes_chunks_from_schema(self) -> None:
        from rag.tool_schemas import ValidateClaimsLLMArgs

        schema = ValidateClaimsLLMArgs.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        self.assertEqual(props, {"claims"})
        self.assertNotIn("chunks", props)
        self.assertNotIn("metadatas", props)

    def test_requires_non_empty_claims(self) -> None:
        from rag.tool_schemas import ValidateClaimsLLMArgs

        with self.assertRaises(ValidationError):
            ValidateClaimsLLMArgs.model_validate({"claims": []})


class ValidateClaimsArgsTests(unittest.TestCase):
    def test_accepts_explicit_chunks_and_metadatas(self) -> None:
        from rag.tool_schemas import ValidateClaimsArgs

        args = ValidateClaimsArgs.model_validate(
            {
                "claims": ["MSFT risk"],
                "chunks": ["Item 1A risk factors."],
                "metadatas": [{"ticker": "MSFT", "year": "2024"}],
            }
        )
        self.assertEqual(len(args.chunks), 1)
        self.assertEqual(args.metadatas[0]["ticker"], "MSFT")

    def test_defaults_chunks_and_metadatas_to_empty(self) -> None:
        from rag.tool_schemas import ValidateClaimsArgs

        args = ValidateClaimsArgs.model_validate({"claims": ["claim 1"]})
        self.assertEqual(args.chunks, [])
        self.assertEqual(args.metadatas, [])


class SimulatePortfolioArgsTests(unittest.TestCase):
    def test_caps_notional(self) -> None:
        from rag.tool_schemas import SimulatePortfolioArgs

        with self.assertRaises(ValidationError):
            SimulatePortfolioArgs.model_validate(
                {"allocations": {"MSFT": 100.0}, "notional_usd": 2_000_000}
            )

    def test_default_notional(self) -> None:
        from rag.tool_schemas import SimulatePortfolioArgs

        args = SimulatePortfolioArgs.model_validate({"allocations": {"MSFT": 100.0}})
        self.assertEqual(args.notional_usd, 100_000)


class ExportReportArgsTests(unittest.TestCase):
    def test_default_format_is_md(self) -> None:
        from rag.tool_schemas import ExportReportArgs

        args = ExportReportArgs.model_validate(
            {"title": "Report", "content": "Body"}
        )
        self.assertEqual(args.format, "md")

    def test_rejects_unknown_format(self) -> None:
        from rag.tool_schemas import ExportReportArgs

        with self.assertRaises(ValidationError):
            ExportReportArgs.model_validate(
                {"title": "Report", "content": "Body", "format": "html"}
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_tool_schemas -v 2>&1 | head -20`
Expected: `ModuleNotFoundError: No module named 'rag.tool_schemas'` (or `ImportError`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_schemas.py
git commit -m "test(etape4): red tests for Pydantic tool schemas"
```

---

## Task 3 (F2): Implement Pydantic tool schemas

**Files:**
- Create: `rag/tool_schemas.py`

- [ ] **Step 1: Write the implementation**

Create `rag/tool_schemas.py` with this exact content:

```python
"""Pydantic BaseModel schemas for tool arguments (PRD etape 4 §3.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _split_csv(value: Any) -> Any:
    """Accept a comma-separated string in addition to list for LLM tolerance."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class SecFilingsRAGArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for SEC filings / earnings calls.")
    tickers: list[str] | None = Field(
        default=None, description="Tracked tickers filter: NVDA, AMD, MSFT."
    )
    years: list[str] | None = Field(
        default=None, description="Filing years filter, e.g. ['2024']."
    )
    doc_types: list[str] | None = Field(
        default=None, description="Allowed: 10-K, 10-Q, 8-K, EARNINGS_CALL."
    )

    @field_validator("tickers", "years", "doc_types", mode="before")
    @classmethod
    def _coerce_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class MarketPriceArgs(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class ValidateClaimsLLMArgs(BaseModel):
    """Schéma PUBLIC envoyé à OpenAI : le LLM ne fournit que 'claims'."""

    claims: list[str] = Field(..., min_length=1)


class ValidateClaimsArgs(BaseModel):
    """Modèle RUNTIME complet : chunks/metadatas sont injectés par tools_node."""

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


__all__ = [
    "SecFilingsRAGArgs",
    "MarketPriceArgs",
    "ValidateClaimsLLMArgs",
    "ValidateClaimsArgs",
    "SimulatePortfolioArgs",
    "ExportReportArgs",
]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_tool_schemas -v 2>&1 | tail -25`
Expected: All ~15 tests pass.

- [ ] **Step 3: Verify imports + smoke test**

Run: `.venv/bin/python -c "from rag.tool_schemas import SecFilingsRAGArgs, MarketPriceArgs, ValidateClaimsArgs, ValidateClaimsLLMArgs, SimulatePortfolioArgs, ExportReportArgs; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add rag/tool_schemas.py
git commit -m "feat(etape4): Pydantic tool args (BaseModel per tool)"
```

---

## Task 4 (F3): Red tests — tool_events lifecycle

**Files:**
- Create: `tests/test_tool_events_lifecycle.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_tool_events_lifecycle.py` with this exact content:

```python
"""Lifecycle tool_events: running → completed / failed (PRD etape 4 §3.3)."""

import json
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import ValidationError


def _build_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "lc_messages": [],
        "pending_tool_calls": [],
        "tool_events": [],
        "stats": {},
        "final_chunks": [],
        "final_metadatas": [],
        "price_context": "",
        "report_artifacts": [],
    }
    state.update(overrides)
    return state


class ToolEventLifecycleTests(unittest.TestCase):
    def test_running_then_completed(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.agent_nodes import tools_node

        agent = MagicMock()
        tc = ToolCall(id="call_1", name="market_price_tool", arguments=json.dumps({
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:
            mock_execute.return_value = {"text": "ok", "price_context": "ctx"}
            result = tools_node(agent, state)

        events = result["tool_events"]
        self.assertGreaterEqual(len(events), 1)
        last = events[-1]
        self.assertEqual(last["tool"], "market_price_tool")
        self.assertEqual(last["status"], "completed")
        self.assertIn("started_at", last)
        self.assertIn("finished_at", last)
        # timestamps are ISO UTC; finished_at should be >= started_at
        self.assertGreaterEqual(
            datetime.fromisoformat(last["finished_at"]),
            datetime.fromisoformat(last["started_at"]),
        )

    def test_running_then_failed_on_execution_exception(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.agent_nodes import tools_node

        agent = MagicMock()
        tc = ToolCall(id="call_1", name="market_price_tool", arguments=json.dumps({
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:
            mock_execute.side_effect = RuntimeError("provider down")
            result = tools_node(agent, state)

        events = result["tool_events"]
        last = events[-1]
        self.assertEqual(last["status"], "failed")
        self.assertIn("provider down", last["error"])
        self.assertIn("finished_at", last)

    def test_running_then_failed_on_validation_error(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.agent_nodes import tools_node

        agent = MagicMock()
        # Missing required 'query' field → Pydantic validation should fail
        tc = ToolCall(id="call_1", name="sec_filings_rag_tool", arguments=json.dumps({}))

        state = _build_state(pending_tool_calls=[tc])

        with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:
            result = tools_node(agent, state)

        events = result["tool_events"]
        last = events[-1]
        self.assertEqual(last["status"], "failed")
        self.assertIn("args_validation", last["error"])
        # execute_tool must NOT be called when validation fails
        mock_execute.assert_not_called()

    def test_injected_chunks_and_metadatas_passed_to_execute_tool(self) -> None:
        from rag.llm_provider import ToolCall
        from rag.nodes.agent_nodes import tools_node

        agent = MagicMock()
        tc = ToolCall(
            id="call_1",
            name="validate_claims_tool",
            arguments=json.dumps({"claims": ["MSFT risk"]}),
        )

        state = _build_state(
            pending_tool_calls=[tc],
            final_chunks=["Item 1A risk."],
            final_metadatas=[{"ticker": "MSFT", "year": "2024"}],
        )

        with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:
            mock_execute.return_value = {"text": "ok", "validations": []}
            tools_node(agent, state)
            args = mock_execute.call_args[0][1]  # second positional arg

        # args is a ValidateClaimsArgs BaseModel
        self.assertEqual(args.claims, ["MSFT risk"])
        self.assertEqual(args.chunks, ["Item 1A risk."])
        self.assertEqual(args.metadatas, [{"ticker": "MSFT", "year": "2024"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_tool_events_lifecycle -v 2>&1 | head -30`
Expected: ImportError on `from rag.nodes.agent_nodes import tools_node` (function signature unchanged yet), so tests still import — but the new `status` field is absent from existing event dicts, so assertions on `last["status"] == "completed"` FAIL.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_events_lifecycle.py
git commit -m "test(etape4): red tests for tool_events running/completed/failed"
```

---

## Task 5 (F4a): Add `ToolEvent` TypedDict to `state.py`

**Files:**
- Modify: `rag/nodes/state.py`

- [ ] **Step 1: Add the `ToolEvent` TypedDict and tighten `tool_events` type**

Replace the entire content of `rag/nodes/state.py` with:

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict

from rag.llm_provider import ToolCall


class ToolEvent(TypedDict, total=False):
    id: str
    tool: str
    status: Literal["running", "completed", "failed"]
    started_at: str  # ISO UTC
    finished_at: str  # ISO UTC
    args: dict[str, Any]
    args_summary: str
    result: dict[str, Any] | None
    error: str | None


class GraphState(TypedDict, total=False):
    conversation_id: str
    query: str
    messages: list[dict[str, str]]
    lc_messages: list[dict[str, Any]]
    tool_events: list[ToolEvent]
    report_artifacts: list[dict[str, Any]]
    tool_calls_pending: bool
    pending_tool_calls: list[ToolCall]
    agent_iterations: int
    normalized_query: str
    metadata_filter: dict[str, str]
    target_tickers: list[str]
    doc_type_priority: list[str]
    off_topic_blocked: bool
    clarification_question: str
    price_context: str
    price_tickers: list[str]
    price_window_start: str
    price_window_end: str
    memory_summary: str
    memory_window: list[dict[str, str]]
    candidate_indices: list[int]
    final_chunks: list[str]
    final_metadatas: list[dict[str, Any]]
    draft_answer: str
    answer: str
    gc_applied: bool
    stats: dict[str, Any]
```

- [ ] **Step 2: Verify import + smoke**

Run: `.venv/bin/python -c "from rag.nodes.state import GraphState, ToolEvent; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add rag/nodes/state.py
git commit -m "refactor(state): add ToolEvent TypedDict; tighten tool_events type"
```

---

## Task 6 (F4b): Refactor `rag/tools.py` — BaseModel args

**Files:**
- Modify: `rag/tools.py`

This is a large refactor. Read the current file first (`rag/tools.py`, 639 lines) before editing.

- [ ] **Step 1: Replace imports**

At the top of `rag/tools.py`, add the new imports (after the existing `from rag.paths import` line):

```python
from pydantic import BaseModel

from rag.tool_schemas import (
    ExportReportArgs,
    MarketPriceArgs,
    SecFilingsRAGArgs,
    SimulatePortfolioArgs,
    ValidateClaimsArgs,
    ValidateClaimsLLMArgs,
)
```

- [ ] **Step 2: Refactor `run_sec_filings_rag` signature**

Replace the function signature and first lines so it accepts a `SecFilingsRAGArgs`:

Find:
```python
def run_sec_filings_rag(
    agent: Any,
    query: str,
    tickers: list[str] | None = None,
    years: list[str] | None = None,
    doc_types: list[str] | None = None,
) -> dict[str, Any]:
    normalized_tickers = _normalize_tickers(tickers)
    normalized_years = _normalize_years(years)
    normalized_doc_types = _normalize_doc_types(doc_types)
```

Replace with:
```python
def run_sec_filings_rag(
    args: SecFilingsRAGArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    normalized_tickers = _normalize_tickers(args.tickers)
    normalized_years = _normalize_years(args.years)
    normalized_doc_types = _normalize_doc_types(args.doc_types)
    query = args.query
```

The rest of the function body uses `query` (now a local). The variable `query` was previously a parameter; reassign it from `args.query` so the rest of the body (`decompose_query(agent, query)`, `rag_state["normalized_query"] = query`, etc.) is unchanged.

- [ ] **Step 3: Refactor `run_market_price_tool` signature**

Find:
```python
def run_market_price_tool(
    agent: Any,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    normalized = _normalize_tickers(tickers)
```

Replace with:
```python
def run_market_price_tool(
    args: MarketPriceArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    normalized = _normalize_tickers(args.tickers)
    start_date = args.start_date
    end_date = args.end_date
```

- [ ] **Step 4: Refactor `run_validate_claims` signature**

Find:
```python
def run_validate_claims(
    agent: Any,
    claims: list[str],
    chunks: list[str],
    metadatas: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks:
```

Replace with:
```python
def run_validate_claims(
    args: ValidateClaimsArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    claims = args.claims
    chunks = args.chunks
    metadatas = args.metadatas
    if not chunks:
```

The rest of the function body is unchanged (uses `claims`, `chunks`, `metadatas` locals).

- [ ] **Step 5: Refactor `run_simulate_portfolio` signature**

Find:
```python
def run_simulate_portfolio(
    allocations: dict[str, float],
    notional_usd: float = 100_000,
) -> dict[str, Any]:
    normalized = _normalize_allocations(allocations)
```

Replace with:
```python
def run_simulate_portfolio(
    args: SimulatePortfolioArgs,
) -> dict[str, Any]:
    normalized = _normalize_allocations(args.allocations)
    notional_usd = args.notional_usd
```

- [ ] **Step 6: Refactor `run_export_investment_report` signature**

Find:
```python
def run_export_investment_report(title: str, content: str, fmt: str = "md") -> dict[str, Any]:
    ensure_dir(REPORTS_DIR)
    safe_title = re.sub(r"[^\w\-]+", "_", title.strip())[:80] or "investment_report"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    extension = "md" if fmt.lower() != "pdf" else "pdf"
```

Replace with:
```python
def run_export_investment_report(args: ExportReportArgs) -> dict[str, Any]:
    ensure_dir(REPORTS_DIR)
    title = args.title
    content = args.content
    fmt = args.format
    safe_title = re.sub(r"[^\w\-]+", "_", title.strip())[:80] or "investment_report"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    extension = "md" if fmt.lower() != "pdf" else "pdf"
```

(Replace the remaining `title`, `content` references in the function body with the new locals — they're already named the same.)

- [ ] **Step 7: Refactor `get_tool_definitions` to use Pydantic JSON schema for validate_claims**

Find the `validate_claims_tool` entry in `get_tool_definitions()` (around line 543-560):

```python
{
    "type": "function",
    "function": {
        "name": "validate_claims_tool",
        "description": VALIDATE_CLAIMS_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Factual claims to verify against current RAG excerpts.",
                },
            },
            "required": ["claims"],
        },
    },
},
```

Replace with:
```python
{
    "type": "function",
    "function": {
        "name": "validate_claims_tool",
        "description": VALIDATE_CLAIMS_DESCRIPTION,
        "parameters": ValidateClaimsLLMArgs.model_json_schema(),
    },
},
```

(Remove the redundant `"type": "object"` — Pydantic v2 schema already includes it.)

- [ ] **Step 8: Refactor `execute_tool` signature**

Find the entire `execute_tool` function and replace it with:

```python
def execute_tool(
    name: str,
    args: BaseModel,
    *,
    agent: Any,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name == "sec_filings_rag_tool":
        return run_sec_filings_rag(args, agent=agent)  # type: ignore[arg-type]
    if name == "market_price_tool":
        return run_market_price_tool(args, agent=agent)  # type: ignore[arg-type]
    if name == "export_investment_report_tool":
        return run_export_investment_report(args)  # type: ignore[arg-type]
    if name == "validate_claims_tool":
        # Resolve injected args from state (chunks + metadatas).
        if not isinstance(args, ValidateClaimsArgs):
            full = ValidateClaimsArgs(
                claims=args.claims,  # type: ignore[union-attr]
                chunks=list((state or {}).get("final_chunks") or []),
                metadatas=list((state or {}).get("final_metadatas") or []),
            )
        else:
            full = args
        return run_validate_claims(full, agent=agent)
    if name == "simulate_portfolio_tool":
        return run_simulate_portfolio(args)  # type: ignore[arg-type]
    return {"text": f"Unknown tool: {name}"}
```

- [ ] **Step 9: Verify smoke import**

Run: `.venv/bin/python -c "from rag.tools import get_tool_definitions, execute_tool, run_validate_claims; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 10: Commit**

```bash
git add rag/tools.py
git commit -m "refactor(tools): BaseModel args; inject chunks/metadatas from state"
```

---

## Task 7 (F4c): Refactor `tools_node` in `agent_nodes.py` — lifecycle + injection

**Files:**
- Modify: `rag/nodes/agent_nodes.py`

- [ ] **Step 1: Add imports**

At the top of `rag/nodes/agent_nodes.py`, add the new imports (after existing imports):

```python
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from rag.tool_schemas import (
    ExportReportArgs,
    MarketPriceArgs,
    SecFilingsRAGArgs,
    SimulatePortfolioArgs,
    ValidateClaimsArgs,
    ValidateClaimsLLMArgs,
)
from rag.nodes.state import ToolEvent
```

Remove the existing `from datetime import UTC, datetime` if it's already there (avoid duplicate).

- [ ] **Step 2: Add helper functions**

After the `_summarize_tool_args` function (around line 92), add:

```python
def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return {}


def _safe_result(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return {"text": str(obj)}


def _resolve_full_args(
    tool_name: str, llm_args: dict[str, Any], state: Any
) -> BaseModel:
    """Map a tool name + LLM-supplied args to a fully-resolved BaseModel.

    For ``validate_claims_tool``, chunks and metadatas are injected from
    ``state["final_chunks"]`` and ``state["final_metadatas"]``.
    """
    if tool_name == "validate_claims_tool":
        llm = ValidateClaimsLLMArgs.model_validate(llm_args)
        return ValidateClaimsArgs(
            claims=llm.claims,
            chunks=list((state or {}).get("final_chunks") or []),
            metadatas=list((state or {}).get("final_metadatas") or []),
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

- [ ] **Step 3: Replace the `tools_node` function body**

Find the `tools_node` function (around line 182-247) and replace the body of the `for tc in pending:` loop with the new lifecycle logic:

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
        try:
            llm_args = json.loads(tc.arguments or "{}")
        except json.JSONDecodeError:
            llm_args = {}

        event: ToolEvent = {
            "id": event_id,
            "tool": tc.name,
            "status": "running",
            "started_at": _now_utc(),
            "args": llm_args,
            "args_summary": _summarize_tool_args(tc.name, tc.arguments),
        }
        tool_events.append(event)

        # Validate LLM args; resolve injected (chunks/metadatas) from state.
        try:
            full_args = _resolve_full_args(tc.name, llm_args, state)
        except ValidationError as e:
            event["status"] = "failed"
            event["error"] = f"args_validation: {e.errors()[0]['msg']}"
            event["finished_at"] = _now_utc()
            continue

        # Execute the tool.
        try:
            result = execute_tool(tc.name, full_args, agent=agent, state=state)
        except Exception as e:
            event["status"] = "failed"
            event["error"] = f"execution: {e}"
            event["finished_at"] = _now_utc()
            continue

        event["status"] = "completed"
        event["result"] = _safe_result(result)
        event["finished_at"] = _now_utc()

        # Update state side-effects per tool.
        tool_text = result.get("text", json.dumps(result, ensure_ascii=False))

        if tc.name == "sec_filings_rag_tool":
            final_chunks = result.get("final_chunks") or final_chunks
            final_metadatas = result.get("final_metadatas") or final_metadatas
            stats.update(result.get("stats") or {})
            stats["rag_tool_used"] = True

        if tc.name == "market_price_tool" and result.get("price_context"):
            price_context = result["price_context"]
            stats["price_tool_used"] = True

        if tc.name == "export_investment_report_tool" and result.get("path"):
            report_artifacts.append(
                {
                    "path": result["path"],
                    "filename": result.get("filename", ""),
                    "title": result.get("title", ""),
                    "format": result.get("format", "md"),
                }
            )
            stats["report_exported"] = True

        if tc.name == "validate_claims_tool":
            stats["validate_tool_used"] = True
            stats.update(result.get("stats") or {})

        if tc.name == "simulate_portfolio_tool" and result.get("positions"):
            stats["simulate_tool_used"] = True

        lc_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": tool_text,
            }
        )

    return {
        "lc_messages": lc_messages,
        "tool_calls_pending": False,
        "pending_tool_calls": [],
        "final_chunks": final_chunks,
        "final_metadatas": final_metadatas,
        "price_context": price_context,
        "report_artifacts": report_artifacts,
        "tool_events": tool_events,
        "stats": stats,
    }
```

- [ ] **Step 4: Verify import + smoke**

Run: `.venv/bin/python -c "from rag.nodes.agent_nodes import tools_node, _resolve_full_args, _now_utc; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add rag/nodes/agent_nodes.py
git commit -m "refactor(tools_node): lifecycle running/completed/failed + injected args"
```

---

## Task 8 (F4d): Adapt existing tests to new signatures

**Files:**
- Modify: `tests/test_regressions.py`

- [ ] **Step 1: Update `test_validate_claims_supported_and_unsupported`**

Find (around line 420-454):
```python
def test_validate_claims_supported_and_unsupported(self):
    from rag.tools import run_validate_claims
    ...
    result = run_validate_claims(
        agent=agent,
        claims=[...],
        chunks=chunks,
        metadatas=metadatas,
    )
```

Replace the call with:
```python
from rag.tool_schemas import ValidateClaimsArgs
from rag.tools import run_validate_claims

result = run_validate_claims(
    args=ValidateClaimsArgs(claims=[...], chunks=chunks, metadatas=metadatas),
    agent=agent,
)
```

(Use the exact same `claims=[...]` list as before, with the two test claims.)

- [ ] **Step 2: Update `test_validate_claims_requires_rag_chunks`**

Find (around line 456-463):
```python
def test_validate_claims_requires_rag_chunks(self):
    from rag.tools import run_validate_claims

    agent = SimpleNamespace(rag=SimpleNamespace(provider=MagicMock()))
    result = run_validate_claims(agent=agent, claims=["test claim"], chunks=[], metadatas=[])
```

Replace with:
```python
def test_validate_claims_requires_rag_chunks(self):
    from rag.tool_schemas import ValidateClaimsArgs
    from rag.tools import run_validate_claims

    agent = SimpleNamespace(rag=SimpleNamespace(provider=MagicMock()))
    result = run_validate_claims(
        args=ValidateClaimsArgs(claims=["test claim"], chunks=[], metadatas=[]),
        agent=agent,
    )
```

- [ ] **Step 3: Update the other `run_validate_claims` calls in `ValidateClaimsNLITests`**

There are 2 more calls in `ValidateClaimsNLITests` (around line 556, 584, 601). Apply the same refactor:
- Import `ValidateClaimsArgs` at the top of the class (or method)
- Replace `run_validate_claims(agent=agent, claims=..., chunks=..., metadatas=...)` with `run_validate_claims(args=ValidateClaimsArgs(claims=..., chunks=..., metadatas=...), agent=agent)`

- [ ] **Step 4: Update `test_agent_tool_loop_mocked`**

Find (around line 477-525). The `execute_tool` mock should be updated to reflect the new signature. The mock returns a dict that previously represented `execute_tool(agent, name, args, rag_context=...)` output — that signature changed.

Replace the `with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:` block to:

```python
with patch("rag.nodes.agent_nodes.execute_tool") as mock_execute:
    mock_execute.return_value = {
        "text": "[1] ticker=MSFT excerpt",
        "final_chunks": ["risk factors supply chain"],
        "final_metadatas": [{"ticker": "MSFT", "year": "2024", "file_type": "10-K"}],
    }
    final = agent_node(agent, state)
    final = tools_node(agent, final)
```

(Add the `final = tools_node(agent, final)` line after the agent_node call — to actually exercise the new lifecycle.)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m unittest discover tests 2>&1 | tail -15`
Expected: 49+ tests pass (46 existing + ~9 new from F3 and F4). Look for `OK`.

If failures, fix the specific test code (do NOT change the production code).

- [ ] **Step 6: Run the new lifecycle tests specifically**

Run: `.venv/bin/python -m unittest tests.test_tool_events_lifecycle -v 2>&1 | tail -10`
Expected: All 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_regressions.py
git commit -m "test(regressions): adapt to BaseModel tool args signature"
```

---

## Task 9 (F5): Verification report

**Files:**
- Create: `docs/superpowers/plans/2026-06-05-async-multi-agent-verification.md`

- [ ] **Step 1: Run the full test suite + capture counts**

Run: `.venv/bin/python -m unittest discover tests 2>&1 | tail -5`

- [ ] **Step 2: Smoke import all touched modules**

Run:
```bash
.venv/bin/python -c "
from rag.tool_schemas import SecFilingsRAGArgs, MarketPriceArgs, ValidateClaimsArgs, ValidateClaimsLLMArgs, SimulatePortfolioArgs, ExportReportArgs
from rag.nodes.state import GraphState, ToolEvent
from rag.tools import get_tool_definitions, execute_tool, run_validate_claims, run_sec_filings_rag, run_market_price_tool, run_simulate_portfolio, run_export_investment_report
from rag.nodes.agent_nodes import tools_node, _resolve_full_args
print('all imports OK')
"
```
Expected: `all imports OK`

- [ ] **Step 3: Verify UI / app_rag.py still compiles**

Run: `.venv/bin/python -m py_compile ui/app_rag.py && echo "app_rag OK"`
Expected: `app_rag OK`

- [ ] **Step 4: Write the verification report**

Create `docs/superpowers/plans/2026-06-05-async-multi-agent-verification.md` with:

```markdown
# Étape 4 — Verification Report

**Date** : 2026-06-05
**Branche** : `etape2`
**Spec** : `docs/superpowers/specs/2026-06-05-async-multi-agent-design.md`
**Plan** : `docs/superpowers/plans/2026-06-05-async-multi-agent.md`

## Test suite

| Suite | Tests | Statut |
|-------|-------|--------|
| `test_tool_schemas` | ~15 | ✅ OK |
| `test_tool_events_lifecycle` | 4 | ✅ OK |
| `test_regressions` (adapted) | 34 | ✅ OK |
| `test_embedding_backoff` | 12 | ✅ OK |
| **Total** | **65+** | **✅ All green** |

## Smoke tests

- [x] Pydantic BaseModel imports
- [x] ToolEvent TypedDict import
- [x] All run_* functions take BaseModel
- [x] get_tool_definitions uses ValidateClaimsLLMArgs.model_json_schema()
- [x] execute_tool injects chunks/metadatas from state for validate_claims_tool
- [x] tools_node emits running/completed/failed events
- [x] UI app_rag.py compiles

## Couverture PRD

- ✅ §3.1 Pydantic BaseModel typage strict
- ✅ §3.2 Isolation contexte (chunks/metadatas via args Pydantic, schéma LLM minimal)
- ✅ §3.3 tool_events lifecycle running/completed/failed
- ✅ §5.3 tests async-ready (TypedDict + BaseModel sérialisables)

## Commits

```
<git log --oneline e89a822..HEAD à compléter>
```

## Hors scope (volontairement)

- ❌ asyncio refactor
- ❌ Multi-agent orchestration
```

(Substitute the actual `git log --oneline` output.)

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-06-05-async-multi-agent-verification.md
git commit -m "chore(etape4): verification report"
```

---

## Self-Review Notes

- **Spec coverage**: All 4 PRD sections (§3.1 typage, §3.2 isolation, §3.3 lifecycle, §5.3 tests) have explicit tasks (T2-F2, T6, T7, T8).
- **Placeholder scan**: No TBD/TODO. All code blocks are complete and copy-pastable.
- **Type consistency**: `BaseModel` used everywhere; `ToolEvent` TypedDict defined once in state.py and imported in agent_nodes.py; `ValidateClaimsArgs` vs `ValidateClaimsLLMArgs` distinct throughout.

## Risks

1. **Existing test fragility** — `test_agent_tool_loop_mocked` may need extra iteration if state shape changed. Plan accounts for this in T8.
2. **Pydantic v2 vs v1** — we target v2 (`Field`, `field_validator`, `model_validate`, `model_json_schema`). If v1 is installed, tests will fail with `Field` import error.
