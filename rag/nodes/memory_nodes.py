from __future__ import annotations

import asyncio
from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="memory_read_node")
async def memory_read_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    summary, window = await asyncio.gather(
        asyncio.to_thread(agent.memory_store.get_summary, conversation_id),
        asyncio.to_thread(agent.memory_store.get_window, conversation_id),
    )
    return {
        "memory_summary": summary,
        "memory_window": window,
    }


@traceable(name="memory_write_node")
async def memory_write_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    answer = state.get("answer", "")
    if not answer:
        for msg in reversed(state.get("lc_messages") or []):
            if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
                answer = str(msg["content"])
                break
    await asyncio.gather(
        asyncio.to_thread(
            agent.memory_store.append_turn,
            conversation_id, "user", state.get("normalized_query", ""),
        ),
        asyncio.to_thread(
            agent.memory_store.append_turn,
            conversation_id, "assistant", answer,
        ),
    )
    return {}


@traceable(name="context_prune_node")
async def context_prune_node(agent: Any, state: GraphState) -> GraphState:
    chunks = state.get("final_chunks", [])
    metadatas = state.get("final_metadatas", [])
    filtered = await asyncio.gather(*[
        asyncio.to_thread(
            agent.memory_store.is_duplicate_chunk,
            state["conversation_id"], chunk,
        )
        for chunk in chunks
    ])
    kept_pairs = [
        (chunk, metadatas[index] if index < len(metadatas) else {})
        for index, chunk in enumerate(chunks)
        if not filtered[index]
    ]

    while kept_pairs and agent.rag.count_context_tokens([chunk for chunk, _ in kept_pairs]) > agent.max_context_tokens:
        kept_pairs = kept_pairs[:-1]

    final_chunks = [chunk for chunk, _ in kept_pairs]
    final_metadatas = [metadata for _, metadata in kept_pairs]
    stats = state.get("stats", {})
    stats.update(
        {
            "chunks_used": len(final_chunks),
            "estimated_context_tokens": agent.rag.count_context_tokens(final_chunks),
            "context_pruned": len(final_chunks) != len(chunks),
        }
    )
    return {
        "final_chunks": final_chunks,
        "final_metadatas": final_metadatas,
        "stats": stats,
    }


@traceable(name="gc_node")
async def gc_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    memory = await asyncio.to_thread(
        agent.memory_store.get_or_create, conversation_id
    )
    gc_applied = False

    if len(memory.turns) >= agent.summarize_every_n_turns:
        transcript = "\n".join(
            f"{t['role']}: {t['content']}" for t in memory.turns[:-agent.memory_store.window_size]
        )
        if transcript.strip():
            summary_prompt = (
                "Summarize the following conversation in at most 8 lines for long-term memory "
                "of a finance assistant.\n"
                f"Existing memory summary: {memory.summary}\n\n"
                f"Conversation to compress:\n{transcript}"
            )
            new_summary = await asyncio.to_thread(
                agent.rag.provider.generate,
                summary_prompt, temperature=0.0, max_tokens=300,
            )
            await asyncio.gather(
                asyncio.to_thread(
                    agent.memory_store.update_summary, conversation_id, new_summary,
                ),
                asyncio.to_thread(
                    agent.memory_store.trim_turns,
                    conversation_id, keep_last=agent.memory_store.window_size,
                ),
            )
            gc_applied = True

    final_chunks = state.get("final_chunks", [])
    if final_chunks:
        await asyncio.gather(*[
            asyncio.to_thread(
                agent.memory_store.remember_chunk, conversation_id, chunk,
            )
            for chunk in final_chunks
        ])

    stats = state.get("stats", {})
    price_tool_used = any(
        event.get("tool") == "market_price_tool"
        for event in (state.get("tool_events") or [])
    )
    stats.update(
        {
            "chunks_used": len(final_chunks),
            "gc_applied": gc_applied,
            "estimated_context_tokens": agent.rag.count_context_tokens(final_chunks),
            "price_tool_used": price_tool_used,
        }
    )
    return {"gc_applied": gc_applied, "stats": stats}
