"""End-to-end smoke test: instantiate a real agent and call arun/astream.

Verifies that the async runtime works on a real (not mocked) graph. Uses
a tiny in-memory RAG + minimal agent config to keep the test fast and
deterministic (no network, no ChromaDB persistence).
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LANGSMITH_TRACING", "false")

from src.graph.flow import FinanceLangGraphAgent
from src.llm.provider import LLMToolResponse


async def _inline_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


def _make_fake_provider(content: str = "Réponse stub async.") -> object:
    """Build a fake LLMProvider that returns canned responses for testing."""
    from src.llm.provider import LLMStreamChunk

    provider = SimpleNamespace()
    provider.generate = lambda *a, **kw: content
    provider.invoke_with_tools = lambda *a, **kw: LLMToolResponse(
        content=content, tool_calls=[]
    )

    async def fake_astream(messages, tools=None, temperature=0.1, max_tokens=2000):
        # Yield a single chunk with the canned content
        yield LLMStreamChunk(delta=content, finish_reason="stop")

    provider.ainvoke_with_tools_stream = fake_astream
    provider.embed = lambda texts: [[0.0] * 8 for _ in texts]
    return provider


def _build_minimal_agent() -> FinanceLangGraphAgent:
    """Build a FinanceLangGraphAgent with a tiny in-memory RAG."""
    # Skip the real RAG entirely by mocking at the agent level. We only
    # care about the graph wiring (which is what G4 added). The fake
    # provider avoids all network calls.
    rag = SimpleNamespace()
    rag.provider = _make_fake_provider()
    rag.documents = []
    rag.doc_metadata = []
    rag._deduplicate_indices = lambda idxs: list(idxs)
    rag._rerank = lambda q, idxs, top_k: idxs[:top_k]
    rag.retrieve = lambda *a, **kw: SimpleNamespace(
        chunk_indices=[], documents=[], metadatas=[]
    )
    rag.count_context_tokens = lambda chunks: 0
    return FinanceLangGraphAgent(rag=rag)


@unittest.skipUnless(
    os.getenv("RUN_ASYNC_E2E") == "1",
    "Set RUN_ASYNC_E2E=1 to run LangGraph smoke tests.",
)
class AsyncEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_returns_state(self):
        agent = _build_minimal_agent()
        with patch("src.graph.memory_store", side_effect=_inline_to_thread):
            result = await agent.arun("Question test async")
        # The result must be a state dict with at least conversation_id
        self.assertIsInstance(result, dict)
        self.assertIn("conversation_id", result)
        self.assertEqual(result["query"], "Question test async")

    async def test_astream_yields_events(self):
        agent = _build_minimal_agent()
        events = []
        with patch("src.graph.memory_store", side_effect=_inline_to_thread):
            async for event in agent.astream("Question streaming"):
                events.append(event)
        # At least one event must be yielded (prepare_query_node start/end)
        self.assertGreaterEqual(
            len(events),
            1,
            f"astream must yield at least one event; got {len(events)}",
        )

    async def test_arun_preserves_conversation_id(self):
        agent = _build_minimal_agent()
        with patch("src.graph.memory_store", side_effect=_inline_to_thread):
            result = await agent.arun("Test", conversation_id="my-convo-123")
        self.assertEqual(result["conversation_id"], "my-convo-123")


if __name__ == "__main__":
    unittest.main()
