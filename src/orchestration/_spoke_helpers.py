from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from src.llm.types import ToolCall, LLMStreamChunk
from src.orchestration.progress import emit_agent_progress
from src.tools.execute import ToolExecutor

_LOGGER = logging.getLogger("src.orchestration._spoke_helpers")

LLM_TIMEOUT = 180
TOOL_TIMEOUT = 120


async def run_spoke_agent(
    agent: Any,
    system_prompt: str,
    task: str,
    tool_names: list[str],
    state: dict[str, Any],
    max_iterations: int = 2,
) -> tuple[str, dict[str, Any]]:
    from src.tools.definitions import get_tool_definitions

    all_tools = get_tool_definitions()
    allowed_tools = [
        t for t in all_tools if t.get("function", {}).get("name") in tool_names
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tool_call_count = 0
    accumulated_chunks: list[str] = []
    accumulated_metadatas: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        try:
            full_text, final_tool_calls = await asyncio.wait_for(
                _collect_llm_response(agent, messages, allowed_tools),
                timeout=LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _LOGGER.warning("Spoke LLM timed out after %ss", LLM_TIMEOUT)
            return f"Analyse interrompue (timeout LLM après {LLM_TIMEOUT}s).", {
                "spoke_llm_iterations": iteration + 1,
                "spoke_tool_calls": tool_call_count,
                "final_chunks": accumulated_chunks,
                "final_metadatas": accumulated_metadatas,
            }
        except Exception as e:
            _LOGGER.warning("Spoke LLM error: %s", e)
            return f"Erreur LLM: {e}", {
                "spoke_llm_iterations": iteration + 1,
                "spoke_tool_calls": tool_call_count,
                "final_chunks": accumulated_chunks,
                "final_metadatas": accumulated_metadatas,
            }

        if final_tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": full_text,
                    "tool_calls": final_tool_calls,
                }
            )
            executor = ToolExecutor(agent, state)
            for tc in final_tool_calls:
                tool_call_count += 1
                emit_agent_progress(
                    "Outils",
                    "running",
                    f"Exécution de {tc.name}...",
                )
                try:
                    outcome = await asyncio.wait_for(
                        executor.execute(tc), timeout=TOOL_TIMEOUT
                    )
                    messages.append(outcome.message)
                    if outcome.result and "final_chunks" in outcome.result:
                        accumulated_chunks.extend(outcome.result["final_chunks"] or [])
                        accumulated_metadatas.extend(
                            outcome.result["final_metadatas"] or []
                        )
                except asyncio.TimeoutError:
                    _LOGGER.warning("Tool %s timed out", tc.name)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": json.dumps(
                                {"error": f"Tool timed out after {TOOL_TIMEOUT}s"}
                            ),
                        }
                    )
                except Exception as e:
                    _LOGGER.warning("Spoke tool %s failed: %s", tc.name, e)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": json.dumps({"error": str(e)}),
                        }
                    )
        else:
            return full_text.strip(), {
                "spoke_llm_iterations": iteration + 1,
                "spoke_tool_calls": tool_call_count,
                "final_chunks": accumulated_chunks,
                "final_metadatas": accumulated_metadatas,
            }

    return "Max iterations atteint.", {
        "spoke_llm_iterations": max_iterations,
        "spoke_tool_calls": tool_call_count,
        "final_chunks": accumulated_chunks,
        "final_metadatas": accumulated_metadatas,
    }


async def run_spoke_agent_stream(
    agent: Any,
    system_prompt: str,
    task: str,
    tool_names: list[str],
    state: dict[str, Any],
    max_iterations: int = 2,
) -> AsyncIterator[str]:
    from src.tools.definitions import get_tool_definitions

    all_tools = get_tool_definitions()
    allowed_tools = [
        t for t in all_tools if t.get("function", {}).get("name") in tool_names
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    accumulated_chunks: list[str] = []
    accumulated_metadatas: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        try:
            full_text, final_tool_calls = await asyncio.wait_for(
                _collect_llm_response(agent, messages, allowed_tools),
                timeout=LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            yield f"\n[Spoke LLM timeout after {LLM_TIMEOUT}s]"
            return
        except Exception as e:
            yield f"\n[Spoke LLM error: {e}]"
            return

        if final_tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": full_text,
                    "tool_calls": final_tool_calls,
                }
            )
            executor = ToolExecutor(agent, state)
            for tc in final_tool_calls:
                try:
                    outcome = await asyncio.wait_for(
                        executor.execute(tc), timeout=TOOL_TIMEOUT
                    )
                    messages.append(outcome.message)
                    if outcome.result and "final_chunks" in outcome.result:
                        accumulated_chunks.extend(outcome.result["final_chunks"] or [])
                        accumulated_metadatas.extend(
                            outcome.result["final_metadatas"] or []
                        )
                    yield f"\n[Tool {tc.name} completed]\n"
                except asyncio.TimeoutError:
                    yield f"\n[Tool {tc.name} timed out]"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": json.dumps(
                                {"error": f"Timed out after {TOOL_TIMEOUT}s"}
                            ),
                        }
                    )
                except Exception as e:
                    yield f"\n[Tool {tc.name} failed: {e}]"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": json.dumps({"error": str(e)}),
                        }
                    )
        else:
            yield full_text.strip()
            return

    yield "Max iterations atteint."


async def _collect_llm_response(
    agent: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, list[ToolCall]]:
    full_text = ""
    tool_calls_dict: dict[str | int, dict[str, Any]] = {}

    async for stream_chunk in agent.rag.provider.ainvoke_with_tools_stream(
        messages,
        tools=tools,
        temperature=0.1,
        max_tokens=4096,
    ):
        if stream_chunk.delta:
            full_text += stream_chunk.delta
        if stream_chunk.tool_call_delta:
            for tc_delta in stream_chunk.tool_call_delta:
                tc_index = tc_delta.get("index")
                tc_key = tc_index if tc_index is not None else "tc_0"
                if tc_key not in tool_calls_dict:
                    tool_calls_dict[tc_key] = {
                        "id": tc_delta.get("id") or f"tc_{tc_key}",
                        "name": tc_delta.get("name") or "",
                        "arguments_parts": [],
                        "thought_signature": tc_delta.get("thought_signature"),
                    }
                if tc_delta.get("id"):
                    tool_calls_dict[tc_key]["id"] = tc_delta["id"]
                if tc_delta.get("name"):
                    tool_calls_dict[tc_key]["name"] = tc_delta["name"]
                if tc_delta.get("arguments"):
                    tool_calls_dict[tc_key]["arguments_parts"].append(
                        tc_delta["arguments"]
                    )

    final_tool_calls = [
        ToolCall(
            id=tc["id"],
            name=tc["name"],
            arguments="".join(tc["arguments_parts"]),
            thought_signature=tc.get("thought_signature"),
        )
        for tc in tool_calls_dict.values()
        if tc["name"]
    ]
    return full_text, final_tool_calls
