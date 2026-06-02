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
                "Resume la conversation suivante en 8 lignes max pour memoire long-terme d'un assistant financier.\n"
                f"Memoire existante: {memory.summary}\n\n"
                f"Conversation a compresser:\n{transcript}"
            )
            new_summary = agent.rag.provider.generate(summary_prompt, temperature=0.0, max_tokens=300)
            agent.memory_store.update_summary(conversation_id, new_summary)
            agent.memory_store.trim_turns(conversation_id, keep_last=agent.memory_store.window_size)
            gc_applied = True

    deduped_chunks = []
    for chunk in state.get("final_chunks", []):
        if not agent.memory_store.is_duplicate_chunk(conversation_id, chunk):
            deduped_chunks.append(chunk)
    if deduped_chunks:
        state["final_chunks"] = deduped_chunks

    final_chunks = state.get("final_chunks", [])
    while final_chunks and agent.rag.count_context_tokens(final_chunks) > agent.max_context_tokens:
        final_chunks = final_chunks[:-1]
        gc_applied = True
    state["final_chunks"] = final_chunks

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
