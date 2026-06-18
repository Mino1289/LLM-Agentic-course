"""Conversion des messages entre formats OpenAI et internes."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.llm.types import ToolCall


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


def _gemini_contents_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[Optional[str], list[Any]]:
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_parts.append(str(msg.get("content", "")))
            continue
        if role == "user":
            contents.append(
                types.Content(
                    role="user", parts=[types.Part(text=str(msg.get("content", "")))]
                )
            )
            continue
        if role == "assistant":
            parts: list[Any] = []
            if msg.get("content"):
                parts.append(types.Part(text=str(msg.get("content"))))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, ToolCall):
                    try:
                        args = json.loads(tc.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    part = types.Part.from_function_call(name=tc.name, args=args)
                    if getattr(tc, "thought_signature", None):
                        part.thought_signature = tc.thought_signature
                    parts.append(part)
                elif isinstance(tc, dict):
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", " {}") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    part = types.Part.from_function_call(
                        name=fn.get("name", ""), args=args
                    )
                    if tc.get("thought_signature"):
                        part.thought_signature = tc.thought_signature
                    parts.append(part)
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue
        if role == "tool":
            name = msg.get("name", "tool")
            response_payload = msg.get("content", "")
            try:
                response_data = (
                    json.loads(response_payload)
                    if isinstance(response_payload, str)
                    else response_payload
                )
            except Exception:
                response_data = {"result": response_payload}
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=name, response={"result": response_data}
                        )
                    ],
                )
            )
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents
