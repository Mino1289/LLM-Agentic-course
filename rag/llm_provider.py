from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable, Optional

from openai import OpenAI

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

        embed_kwargs = {"api_key": self.config.embedding_api_key or self.config.api_key}
        if self.config.embedding_base_url:
            embed_kwargs["base_url"] = self.config.embedding_base_url
        self.embedding_client = OpenAI(**embed_kwargs)

        if self.config.provider in {"openai", "github_models"}:
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self.client = OpenAI(**client_kwargs)
        else:
            self.client = None

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(api_key=self.config.api_key)
        return self._gemini_client

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        payload = [text for text in texts if text and text.strip()]
        if not payload:
            return []
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding_model,
            input=payload,
        )
        return [item.embedding for item in response.data]

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
