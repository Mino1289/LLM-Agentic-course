"""Client Gemini pour les appels LLM et streaming."""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from src.llm.types import LLMToolResponse, LLMStreamChunk
from src.llm.messages import _gemini_contents_from_messages
from src.llm.parser import _tool_definitions_to_gemini
from src.llm.sinks import _call_token_sink
from src.llm.parser import _parse_gemini_tool_calls


class GeminiClient:
    def __init__(self, config):
        self.config = config
        self._gemini_client = None

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.config.api_key)
        return self._gemini_client

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMToolResponse:
        client = self._get_gemini_client()
        from google.genai import types
        system_instruction, contents = _gemini_contents_from_messages(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = _tool_definitions_to_gemini(tools)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )

        response = client.models.generate_content(
            model=self.config.chat_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text_parts = []
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(candidate.content, "parts", []) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        return LLMToolResponse(
            content="\n".join(text_parts).strip() or None,
            tool_calls=_parse_gemini_tool_calls(response),
        )

    async def ainvoke_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Gemini streaming via genai.Client.aio. Fallback non-streaming si indisponible."""
        try:
            from google.genai import types
        except ImportError:
            yield LLMStreamChunk(
                delta=self.invoke_with_tools(messages, tools, temperature, max_tokens).content or "",
                finish_reason="stop",
            )
            return

        client = self._get_gemini_client()
        system_instruction, contents = _gemini_contents_from_messages(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = _tool_definitions_to_gemini(tools)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )

        try:
            response = await client.aio.models.generate_content_stream(
                model=self.config.chat_model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except (AttributeError, RuntimeError, NotImplementedError):
            # Fallback non-streaming
            full = self.invoke_with_tools(messages, tools, temperature, max_tokens)
            yield LLMStreamChunk(
                delta=full.content or "",
                tool_call_delta=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "thought_signature": tc.thought_signature}
                    for tc in full.tool_calls
                ] or None,
                finish_reason="stop",
            )
            return

        index_counter = 0
        async for raw_chunk in response:
            text_parts: list[str] = []
            tool_deltas: list[dict[str, Any]] = []
            for candidate in getattr(raw_chunk, "candidates", []) or []:
                for part in getattr(candidate.content, "parts", []) or []:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    fc = getattr(part, "function_call", None)
                    if fc:
                        ts = getattr(part, "thought_signature", None)
                        tool_deltas.append(
                            {
                                "index": index_counter,
                                "id": str(__import__('uuid').uuid4()),
                                "name": getattr(fc, "name", "") or "",
                                "arguments": __import__('json').dumps(dict(getattr(fc, "args", None) or {})),
                                "thought_signature": ts,
                            }
                        )
                        index_counter += 1
            delta = "".join(text_parts)
            chunk = LLMStreamChunk(
                delta=delta,
                tool_call_delta=tool_deltas or None,
            )
            if delta:
                await _call_token_sink(delta)
            yield chunk
