"""Tests for LLMProvider async streaming (PRD etape 6 §3.5).

The provider must expose:
- async_client attribute (AsyncOpenAI for openai/github_models)
- agenerate_stream(prompt) -> AsyncIterator[str]
- ainvoke_with_tools_stream(messages, tools) -> AsyncIterator[LLMStreamChunk]

These tests are RED at H1 (no async streaming yet). They turn GREEN after H2.
"""

import asyncio
import os
import unittest
import unittest.mock
from dataclasses import dataclass
from typing import Any


class _FakeAsyncIterator:
    """Async iterator that yields pre-set chunks in order."""

    def __init__(self, items: list[Any]):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


class _FakeDelta:
    def __init__(self, content: str = "", tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeFunctionCall:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str = "", arguments: str = ""):
        self.id = id
        self.function = _FakeFunctionCall(name=name, arguments=arguments)


class _FakeChoice:
    def __init__(self, delta: _FakeDelta, finish_reason: str | None = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChatCompletionChunk:
    def __init__(self, choices: list[_FakeChoice]):
        self.choices = choices


class _FakeChatCompletions:
    def __init__(self, chunks: list[_FakeChatCompletionChunk]):
        self._chunks = chunks

    async def create(self, **kwargs):
        return _FakeAsyncIterator(self._chunks)


class _FakeAsyncClient:
    def __init__(self, chunks: list[_FakeChatCompletionChunk] | None = None):
        self.chat = type(
            "_Chat", (), {"completions": _FakeChatCompletions(chunks or [])}
        )()


class LLMProviderAsyncInitTests(unittest.TestCase):
    def test_async_client_created_for_openai(self):
        from src.llm.types import LLMConfig
        from src.llm.provider import LLMProvider
        from src.llm.openai_client import OpenAIClient

        # AsyncOpenAI is now constructed inside the per-provider client, so we
        # patch it there (not on src.llm.provider).
        with unittest.mock.patch("src.llm.openai_client.AsyncOpenAI") as mock_async:
            mock_async.return_value = _FakeAsyncClient()
            provider = LLMProvider.__new__(LLMProvider)
            provider.config = LLMConfig(
                provider="openai",
                chat_model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                api_key="x",
                embedding_api_key="x",
            )
            provider._client = OpenAIClient(provider.config)
            provider._init_async_client()
            mock_async.assert_called_once()
            self.assertIsNotNone(provider._client.async_client)
            # The provider proxies the active client's async_client.
            self.assertIsNotNone(provider.async_client)

    def test_no_async_client_for_gemini(self):
        # Gemini uses genai.aio (different path); no OpenAI-style async client.
        from src.llm.types import LLMConfig
        from src.llm.provider import LLMProvider

        provider = LLMProvider.__new__(LLMProvider)
        provider.config = LLMConfig(
            provider="gemini",
            chat_model="gemini-2.0-flash",
            embedding_model="text-embedding-3-small",
            api_key="x",
            embedding_api_key="x",
        )
        provider._client = None
        provider._init_async_client()
        self.assertIsNone(provider.async_client)


class LLMProviderAsyncStreamTests(unittest.IsolatedAsyncioTestCase):
    def _make_provider(self, chunks: list[_FakeChatCompletionChunk]):
        from src.llm.types import LLMConfig
        from src.llm.provider import LLMProvider
        from src.llm.openai_client import OpenAIClient

        provider = LLMProvider.__new__(LLMProvider)
        provider.config = LLMConfig(
            provider="openai",
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            api_key="x",
            embedding_api_key="x",
        )
        provider._gemini_client = None
        # Streaming now lives in the per-provider client; inject a fake
        # AsyncOpenAI-shaped client there (the provider delegates to it).
        client = OpenAIClient(provider.config)
        client.async_client = _FakeAsyncClient(chunks)
        provider._client = client
        return provider

    async def test_agenerate_stream_yields_text_deltas(self):
        chunks = [
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="Hello "))]),
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="world"))]),
            _FakeChatCompletionChunk(
                [_FakeChoice(_FakeDelta(content="!"), finish_reason="stop")]
            ),
        ]
        provider = self._make_provider(chunks)
        tokens = []
        async for token in provider.agenerate_stream("Hi"):
            tokens.append(token)
        self.assertEqual("".join(tokens), "Hello world!")
        self.assertGreater(len(tokens), 0)

    async def test_ainvoke_with_tools_stream_yields_deltas(self):
        chunks = [
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="A"))]),
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="B"))]),
        ]
        provider = self._make_provider(chunks)
        from src.llm.provider import LLMStreamChunk

        stream_chunks: list = []
        async for chunk in provider.ainvoke_with_tools_stream(
            [{"role": "user", "content": "x"}],
            tools=None,
            temperature=0.1,
            max_tokens=100,
        ):
            stream_chunks.append(chunk)
        deltas = [c.delta for c in stream_chunks]
        self.assertEqual("".join(deltas), "AB")

    async def test_ainvoke_with_tools_stream_accumulates_tool_calls(self):
        # Simulate a streaming tool call: chunks of name/arguments deltas
        chunks = [
            _FakeChatCompletionChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            content="",
                            tool_calls=[
                                _FakeToolCall(id="c1", name="sec_filings_rag"),
                            ],
                        )
                    )
                ]
            ),
            _FakeChatCompletionChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            content="",
                            tool_calls=[
                                _FakeToolCall(id="c1", arguments='{"query":'),
                            ],
                        )
                    )
                ]
            ),
            _FakeChatCompletionChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            content="",
                            tool_calls=[
                                _FakeToolCall(id="c1", arguments='"risk"}'),
                            ],
                        )
                    )
                ]
            ),
            _FakeChatCompletionChunk(
                [_FakeChoice(_FakeDelta(content=""), finish_reason="tool_calls")]
            ),
        ]
        provider = self._make_provider(chunks)
        from src.llm.provider import LLMStreamChunk

        stream_chunks: list = []
        async for chunk in provider.ainvoke_with_tools_stream(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "sec_filings_rag_tool"}}],
            temperature=0.1,
            max_tokens=100,
        ):
            stream_chunks.append(chunk)
        # At least one chunk should carry a tool_call_delta
        tool_deltas = [c.tool_call_delta for c in stream_chunks if c.tool_call_delta]
        self.assertGreaterEqual(
            len(tool_deltas),
            1,
            f"ainvoke_with_tools_stream must yield tool_call_delta chunks; got {len(tool_deltas)}",
        )
        # At least one chunk should have finish_reason set
        finish_chunks = [c for c in stream_chunks if c.finish_reason]
        self.assertEqual(len(finish_chunks), 1)
        self.assertEqual(finish_chunks[0].finish_reason, "tool_calls")


class TokenSinkTests(unittest.IsolatedAsyncioTestCase):
    """Verify the contextvar-based token sink receives each text delta
    produced by ainvoke_with_tools_stream."""

    def _make_provider(self, chunks: list[_FakeChatCompletionChunk]):
        from src.llm.types import LLMConfig
        from src.llm.provider import LLMProvider
        from src.llm.openai_client import OpenAIClient

        provider = LLMProvider.__new__(LLMProvider)
        provider.config = LLMConfig(
            provider="openai",
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            api_key="x",
            embedding_api_key="x",
        )
        provider._gemini_client = None
        # Streaming now lives in the per-provider client; inject a fake
        # AsyncOpenAI-shaped client there (the provider delegates to it).
        client = OpenAIClient(provider.config)
        client.async_client = _FakeAsyncClient(chunks)
        provider._client = client
        return provider

    async def test_sink_invoked_for_each_text_delta(self):
        from src.llm.sinks import token_sink

        chunks = [
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="A"))]),
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="B"))]),
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="C"))]),
        ]
        provider = self._make_provider(chunks)
        captured: list[str] = []

        def sink(delta: str) -> None:
            captured.append(delta)

        with token_sink(sink):
            async for _ in provider.ainvoke_with_tools_stream(
                [{"role": "user", "content": "x"}],
                tools=None,
                temperature=0.1,
                max_tokens=100,
            ):
                pass

        self.assertEqual(captured, ["A", "B", "C"])

    async def test_sink_not_invoked_when_no_sink_registered(self):
        from src.llm.sinks import token_sink

        chunks = [
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="A"))]),
        ]
        provider = self._make_provider(chunks)
        # No sink registered — must not raise
        with token_sink(None):
            async for _ in provider.ainvoke_with_tools_stream(
                [{"role": "user", "content": "x"}],
                tools=None,
                temperature=0.1,
                max_tokens=100,
            ):
                pass

    async def test_async_sink_is_awaited(self):
        from src.llm.sinks import token_sink

        chunks = [
            _FakeChatCompletionChunk([_FakeChoice(_FakeDelta(content="A"))]),
        ]
        provider = self._make_provider(chunks)
        captured: list[str] = []

        async def async_sink(delta: str) -> None:
            captured.append(delta)

        with token_sink(async_sink):
            async for _ in provider.ainvoke_with_tools_stream(
                [{"role": "user", "content": "x"}],
                tools=None,
                temperature=0.1,
                max_tokens=100,
            ):
                pass

        self.assertEqual(captured, ["A"])


if __name__ == "__main__":
    unittest.main()
