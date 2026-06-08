from __future__ import annotations

import contextvars
import inspect
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Generator, Iterable, Optional

from openai import AsyncOpenAI, OpenAI

SUPPORTED_CHAT_PROVIDERS = {"openai", "github_models", "gemini"}
SUPPORTED_EMBEDDING_PROVIDERS = {"openai", "github_models"}
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
GITHUB_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Backward-compatible alias
SUPPORTED_PROVIDERS = SUPPORTED_CHAT_PROVIDERS


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMConfig:
    provider: str
    chat_model: str
    embedding_model: str
    api_key: str
    base_url: Optional[str] = None
    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_base_url: Optional[str] = None


@dataclass
class LLMToolResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LLMStreamChunk:
    """Chunk incrémental d'un stream LLM.
    Soit un delta texte (delta != ""), soit un delta de tool_call
    (tool_call_delta != None), soit les deux. Un chunk final peut
    contenir finish_reason set ("stop", "tool_calls", "length")."""
    delta: str = ""
    tool_call_delta: Optional[list[dict[str, Any]]] = None
    finish_reason: Optional[str] = None


# Module-level context variable for the token sink. The provider's async
# stream methods invoke this sink for each token, allowing the UI layer
# to receive tokens in real time without depending on LangChain's chat
# model callback machinery (which doesn't fire for raw provider calls).
_token_sink_var: contextvars.ContextVar[Optional[Callable[[str], Any]]] = (
    contextvars.ContextVar("rag_llm_token_sink", default=None)
)


@contextmanager
def token_sink(sink: Optional[Callable[[str], Any]]):
    """Context manager that registers a token sink for the duration of
    the block. The sink is called with each text delta as it streams.
    Supports both sync and async sinks.

    Example:
        with token_sink(lambda t: container.markdown(t)):
            async for event in agent.astream(...): ...
    """
    token = _token_sink_var.set(sink)
    try:
        yield
    finally:
        _token_sink_var.reset(token)


async def _call_token_sink(delta: str) -> None:
    """Await the registered token sink (if any) for a text delta.
    No-ops when no sink is registered or the delta is empty.
    """
    sink = _token_sink_var.get()
    if not sink or not delta:
        return
    result = sink(delta)
    if inspect.iscoroutine(result):
        await result



def _normalize_github_model_id(model: str) -> str:
    """GitHub Models inference uses IDs like gpt-4.1-mini, not openai/gpt-4.1-mini."""
    model = model.strip()
    for prefix in ("openai/", "azure-openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _build_openai_style_embedding_config(provider: str) -> tuple[str, str, Optional[str]]:
    if provider == "openai":
        return (
            os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            _require_env("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL", "").strip() or OPENAI_DEFAULT_BASE_URL,
        )
    return (
        _normalize_github_model_id(
            os.getenv("GITHUB_EMBEDDING_MODEL", GITHUB_DEFAULT_EMBEDDING_MODEL)
        ),
        _require_env("GITHUB_MODELS_API_KEY"),
        os.getenv("GITHUB_MODELS_BASE_URL", GITHUB_MODELS_BASE_URL).strip(),
    )


def build_llm_config_from_env() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_CHAT_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            f"Expected one of: {sorted(SUPPORTED_CHAT_PROVIDERS)}"
        )

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider != "gemini":
        embedding_provider = provider
    elif embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER='{embedding_provider}'. "
            f"Expected one of: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
        )

    embed_model, embed_key, embed_base = _build_openai_style_embedding_config(embedding_provider)

    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        raw_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        return LLMConfig(
            provider=provider,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=embed_model,
            api_key=api_key,
            base_url=raw_base_url or OPENAI_DEFAULT_BASE_URL,
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
        )

    if provider == "github_models":
        api_key = _require_env("GITHUB_MODELS_API_KEY")
        return LLMConfig(
            provider=provider,
            chat_model=_normalize_github_model_id(
                os.getenv("GITHUB_CHAT_MODEL", GITHUB_DEFAULT_CHAT_MODEL)
            ),
            embedding_model=embed_model,
            api_key=api_key,
            base_url=os.getenv("GITHUB_MODELS_BASE_URL", GITHUB_MODELS_BASE_URL).strip(),
            embedding_provider=embedding_provider,
            embedding_api_key=embed_key,
            embedding_base_url=embed_base,
        )

    api_key = _require_env("GEMINI_API_KEY")
    return LLMConfig(
        provider=provider,
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        embedding_model=embed_model,
        api_key=api_key,
        base_url=None,
        embedding_provider=embedding_provider,
        embedding_api_key=embed_key,
        embedding_base_url=embed_base,
    )


def _openai_messages_to_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
            )
            continue
        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                    )
                else:
                    tool_calls.append(tc)
            payload: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if msg.get("content"):
                payload["content"] = msg["content"]
            converted.append(payload)
            continue
        converted.append({"role": role, "content": msg.get("content", "")})
    return converted


def _parse_openai_tool_calls(raw_calls: Any) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    if not raw_calls:
        return parsed
    for item in raw_calls:
        fn = getattr(item, "function", None) or (item.get("function") if isinstance(item, dict) else None)
        if not fn:
            continue
        name = getattr(fn, "name", None) or fn.get("name", "")
        args = getattr(fn, "arguments", None) or fn.get("arguments", "{}")
        call_id = getattr(item, "id", None) or item.get("id", str(uuid.uuid4()))
        parsed.append(ToolCall(id=call_id, name=name, arguments=args or "{}"))
    return parsed


def _tool_definitions_to_gemini(tools: list[dict[str, Any]]) -> list[Any]:
    from google.genai import types

    declarations = []
    for tool in tools:
        fn = tool.get("function", {})
        declarations.append(
            types.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _gemini_contents_from_messages(messages: list[dict[str, Any]]) -> tuple[Optional[str], list[Any]]:
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_parts.append(str(msg.get("content", "")))
            continue
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=str(msg.get("content", "")))]))
            continue
        if role == "assistant":
            parts: list[Any] = []
            if msg.get("content"):
                parts.append(types.Part(text=str(msg.get("content"))))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, ToolCall):
                    args = json.loads(tc.arguments or "{}")
                    parts.append(types.Part.from_function_call(name=tc.name, args=args))
                elif isinstance(tc, dict):
                    fn = tc.get("function", {})
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                    parts.append(types.Part.from_function_call(name=fn.get("name", ""), args=args))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue
        if role == "tool":
            name = msg.get("name", "tool")
            response_payload = msg.get("content", "")
            try:
                response_data = json.loads(response_payload) if isinstance(response_payload, str) else response_payload
            except Exception:
                response_data = {"result": response_payload}
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=name, response={"result": response_data})],
                )
            )
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _parse_gemini_tool_calls(response: Any) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return parsed
    content = candidates[0].content
    for part in getattr(content, "parts", []) or []:
        fc = getattr(part, "function_call", None)
        if not fc:
            continue
        args = getattr(fc, "args", None) or {}
        parsed.append(
            ToolCall(
                id=str(uuid.uuid4()),
                name=getattr(fc, "name", "") or "",
                arguments=json.dumps(dict(args)),
            )
        )
    return parsed


class LLMProvider:
    """
    Unified provider wrapper for OpenAI, GitHub Models, and Gemini (chat).
    Embeddings always use an OpenAI-compatible endpoint (openai or github_models).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or build_llm_config_from_env()
        self._gemini_client = None
        self.async_client: Optional[AsyncOpenAI] = None

        # Request timeout + max retries for all OpenAI-compatible clients.
        # Default 30s prevents indefinite hangs when the upstream is slow
        # (e.g. GitHub Models rate-limited or cold-starting). Override via
        # OPENAI_REQUEST_TIMEOUT / OPENAI_MAX_RETRIES env vars.
        self.request_timeout = self._parse_request_timeout()
        self.max_retries = self._parse_max_retries()

        embed_kwargs = {"api_key": self.config.embedding_api_key or self.config.api_key}
        if self.config.embedding_base_url:
            embed_kwargs["base_url"] = self.config.embedding_base_url
        embed_kwargs["timeout"] = self.request_timeout
        embed_kwargs["max_retries"] = self.max_retries
        self.embedding_client = OpenAI(**embed_kwargs)

        if self.config.provider in {"openai", "github_models"}:
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            client_kwargs["timeout"] = self.request_timeout
            client_kwargs["max_retries"] = self.max_retries
            self.client = OpenAI(**client_kwargs)
        else:
            self.client = None

        self._init_async_client()

    @staticmethod
    def _parse_request_timeout() -> float:
        raw = os.getenv("OPENAI_REQUEST_TIMEOUT", "30.0")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 30.0
        if value <= 0:
            return 30.0
        return value

    @staticmethod
    def _parse_max_retries() -> int:
        raw = os.getenv("OPENAI_MAX_RETRIES", "2")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 2
        if value < 0:
            return 2
        return value

    def _init_async_client(self) -> None:
        """Initialize the async OpenAI client (openai/github_models only).
        For Gemini, async_client stays None — genai.aio is used directly
        via _get_gemini_client().aio.models.generate_content_stream(...).
        """
        if self.config.provider in {"openai", "github_models"}:
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            # request_timeout / max_retries are normally set in __init__,
            # but some tests construct a provider via __new__ and call
            # _init_async_client directly — fall back to defaults.
            client_kwargs["timeout"] = getattr(self, "request_timeout", 30.0)
            client_kwargs["max_retries"] = getattr(self, "max_retries", 2)
            self.async_client = AsyncOpenAI(**client_kwargs)
        else:
            self.async_client = None

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(api_key=self.config.api_key)
        return self._gemini_client

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        payload = [text for text in texts if text and text.strip()]
        if not payload:
            return []
        cache = self._get_embedding_cache()
        # Deduplicate while preserving order — duplicates in the same batch
        # would otherwise cost N API calls for the same vector.
        unique_payload = list(dict.fromkeys(payload))
        cached: dict[str, list[float]] = {}
        uncached: list[str] = []
        for text in unique_payload:
            hit = cache.get(text)
            if hit is not None:
                cached[text] = hit
            else:
                uncached.append(text)
        if uncached:
            response = self.embedding_client.embeddings.create(
                model=self.config.embedding_model,
                input=uncached,
            )
            new_vectors = [item.embedding for item in response.data]
            cache.put_many(list(zip(uncached, new_vectors)))
            for text, vec in zip(uncached, new_vectors):
                cached[text] = vec
        return [cached[text] for text in payload]

    def _get_embedding_cache(self):
        """Lazy-load the persistent query-embedding cache.

        Held as ``self._embedding_cache`` to keep the constructor signature
        stable. Falls back to a fresh in-memory cache if the default file
        cannot be read (e.g. on a read-only filesystem).
        """
        cache = getattr(self, "_embedding_cache", None)
        if cache is None:
            from rag.embedding_cache import build_embedding_cache_from_env

            try:
                cache = build_embedding_cache_from_env()
            except Exception:
                from rag.embedding_cache import EmbeddingCache
                from rag.paths import DATA_DIR

                cache = EmbeddingCache(DATA_DIR / "embedding_query_cache.json")
            self._embedding_cache = cache
        return cache

    def embedding_cache_stats(self) -> dict[str, int | str]:
        """Return hit/miss counters for the persistent query-embedding cache."""
        return self._get_embedding_cache().stats()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.invoke_with_tools(messages, tools=None, temperature=temperature, max_tokens=max_tokens)
        return response.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> Generator[str, None, None]:
        # Streaming not required for agent loop; fallback to single-shot.
        yield self.generate(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMToolResponse:
        if self.config.provider == "gemini":
            return self._invoke_gemini_with_tools(messages, tools, temperature, max_tokens)
        return self._invoke_openai_with_tools(messages, tools, temperature, max_tokens)

    async def agenerate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        """Stream token-par-token (texte pur). Pour NLI judge, summary, etc.
        Délègue à ainvoke_with_tools_stream(tools=None) et yield les deltas.
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async for chunk in self.ainvoke_with_tools_stream(
            messages, tools=None, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.delta:
                yield chunk.delta

    async def ainvoke_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream LLM token-par-token avec gestion des tool_calls.
        Pour OpenAI/GitHub: utilise async_client.chat.completions.create(stream=True).
        Pour Gemini: tente genai.aio.models.generate_content_stream; fallback
        non-streaming via _invoke_gemini_with_tools si aio indisponible.
        """
        if self.config.provider == "gemini":
            async for chunk in self._ainvoke_gemini_stream(messages, tools, temperature, max_tokens):
                yield chunk
            return
        async for chunk in self._ainvoke_openai_stream(messages, tools, temperature, max_tokens):
            yield chunk

    async def _ainvoke_openai_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[LLMStreamChunk]:
        kwargs: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": _openai_messages_to_api(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.async_client is None:
            raise RuntimeError(
                "async_client is None — openai/github_models provider required for streaming"
            )
        response = await self.async_client.chat.completions.create(**kwargs)
        async for raw_chunk in response:
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
            chunk = LLMStreamChunk(
                delta=delta_text,
                tool_call_delta=tool_deltas,
                finish_reason=getattr(choice, "finish_reason", None),
            )
            if delta_text:
                await _call_token_sink(delta_text)
            yield chunk

    async def _ainvoke_gemini_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Gemini streaming via genai.Client.aio. Fallback non-streaming
        si genai.aio indisponible (régression safe)."""
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
            full = self._invoke_gemini_with_tools(messages, tools, temperature, max_tokens)
            yield LLMStreamChunk(
                delta=full.content or "",
                tool_call_delta=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in full.tool_calls
                ] or None,
                finish_reason="stop",
            )
            return
        async for raw_chunk in response:
            text_parts: list[str] = []
            tool_deltas: list[dict[str, Any]] = []
            for candidate in getattr(raw_chunk, "candidates", []) or []:
                for part in getattr(candidate.content, "parts", []) or []:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    fc = getattr(part, "function_call", None)
                    if fc:
                        tool_deltas.append(
                            {
                                "id": str(uuid.uuid4()),
                                "name": getattr(fc, "name", "") or "",
                                "arguments": json.dumps(dict(getattr(fc, "args", None) or {})),
                            }
                        )
            delta = "".join(text_parts)
            chunk = LLMStreamChunk(
                delta=delta,
                tool_call_delta=tool_deltas or None,
            )
            if delta:
                await _call_token_sink(delta)
            yield chunk

    def _invoke_openai_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> LLMToolResponse:
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
        return LLMToolResponse(
            content=message.content,
            tool_calls=_parse_openai_tool_calls(message.tool_calls),
        )

    def _invoke_gemini_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> LLMToolResponse:
        client = self._get_gemini_client()
        system_instruction, contents = _gemini_contents_from_messages(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            from google.genai import types

            config_kwargs["tools"] = _tool_definitions_to_gemini(tools)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )

        from google.genai import types

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

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)
