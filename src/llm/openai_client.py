"""Client OpenAI pour les appels LLM."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI

from src.llm.types import LLMToolResponse, LLMStreamChunk
from src.llm.messages import _openai_messages_to_api
from src.llm.sinks import _call_token_sink
from src.llm.parser import _parse_openai_tool_calls
from src.llm.config_builder import parse_request_timeout, parse_max_retries


class OpenAIClient:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.async_client = None
        # Request deadline + retries: prevents the multi-minute hang seen when
        # a call stalls with no timeout (regression guard, see Fix #1).
        self.request_timeout = parse_request_timeout()
        self.max_retries = parse_max_retries()

    def _init_sync(self):
        if self.client is None:
            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self.client = OpenAI(**client_kwargs)

    def _init_async(self):
        if self.async_client is None:
            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self.async_client = AsyncOpenAI(**client_kwargs)

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMToolResponse:
        self._init_sync()
        kwargs: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": _openai_messages_to_api(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        return LLMToolResponse(
            content=message.content,
            tool_calls=_parse_openai_tool_calls(message.tool_calls),
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else None,
        )

    async def ainvoke_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> AsyncIterator[LLMStreamChunk]:
        self._init_async()
        kwargs: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": _openai_messages_to_api(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self.async_client.chat.completions.create(**kwargs)
        async for raw_chunk in response:
            usage_data = getattr(raw_chunk, "usage", None)
            if usage_data:
                yield LLMStreamChunk(
                    usage={
                        "prompt_tokens": usage_data.prompt_tokens,
                        "completion_tokens": usage_data.completion_tokens,
                        "total_tokens": usage_data.total_tokens,
                    }
                )
            if not getattr(raw_chunk, "choices", None):
                continue
            choice = raw_chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            delta_text = getattr(delta, "content", None) or ""
            tool_deltas: Optional[list[dict[str, Any]]] = None
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                tool_deltas = []
                for tc in delta_tool_calls:
                    fn = getattr(tc, "function", None)
                    tool_deltas.append(
                        {
                            "index": getattr(tc, "index", None),
                            "id": getattr(tc, "id", None),
                            "name": getattr(fn, "name", None) if fn else None,
                            "arguments": getattr(fn, "arguments", None) if fn else None,
                        }
                    )
            finish_reason = getattr(choice, "finish_reason", None)
            chunk = LLMStreamChunk(
                delta=delta_text,
                tool_call_delta=tool_deltas,
                finish_reason=finish_reason,
            )
            if delta_text:
                await _call_token_sink(delta_text)
            yield chunk
