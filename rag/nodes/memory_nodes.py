from __future__ import annotations

from typing import Any

from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="memory_read_node")
def memory_read_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    return {
        "memory_summary": agent.memory_store.get_summary(conversation_id),
        "memory_window": agent.memory_store.get_window(conversation_id),
    }


@traceable(name="memory_write_node")
def memory_write_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    agent.memory_store.append_turn(conversation_id, "user", state.get("normalized_query", ""))
    agent.memory_store.append_turn(conversation_id, "assistant", state.get("answer", ""))
    return {}


@traceable(name="context_prune_node")
def context_prune_node(agent: Any, state: GraphState) -> GraphState:
    chunks = state.get("final_chunks", [])
    metadatas = state.get("final_metadatas", [])
    kept_pairs = [
        (chunk, metadatas[index] if index < len(metadatas) else {})
        for index, chunk in enumerate(chunks)
        if not agent.memory_store.is_duplicate_chunk(state["conversation_id"], chunk)
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
def gc_node(agent: Any, state: GraphState) -> GraphState:
    conversation_id = state["conversation_id"]
    memory = agent.memory_store.get_or_create(conversation_id)
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
            new_summary = agent.rag.provider.generate(summary_prompt, temperature=0.0, max_tokens=300)
            agent.memory_store.update_summary(conversation_id, new_summary)
            agent.memory_store.trim_turns(conversation_id, keep_last=agent.memory_store.window_size)
            gc_applied = True

    for chunk in state.get("final_chunks", []):
        agent.memory_store.remember_chunk(conversation_id, chunk)

    stats = state.get("stats", {})
    stats.update(
        {
            "chunks_used": len(state.get("final_chunks", [])),
            "gc_applied": gc_applied,
            "estimated_context_tokens": agent.rag.count_context_tokens(state.get("final_chunks", [])),
            "price_tool_used": state.get("price_tool_used", False),
        }
    )
    return {"gc_applied": gc_applied, "stats": stats}
