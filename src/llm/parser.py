"""Parsing des tool calls depuis les réponses des différents providers."""
from __future__ import annotations

import uuid
from typing import Any

from src.llm.types import ToolCall


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
        ts = getattr(part, "thought_signature", None) or getattr(fc, "thought_signature", None)
        parsed.append(
            ToolCall(
                id=str(uuid.uuid4()),
                name=getattr(fc, "name", "") or "",
                arguments=__import__('json').dumps(dict(args)),
                thought_signature=ts,
            )
        )
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
