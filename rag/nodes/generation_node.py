from __future__ import annotations

from typing import Any

from rag.nodes.memory_store import format_chat_context, format_memory_context
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="answer_generate_node")
def answer_generate_node(agent: Any, state: GraphState) -> GraphState:
    final_chunks = state.get("final_chunks", [])
    if not final_chunks:
        return {
            "draft_answer": (
                "Je ne trouve pas assez de sources fiables pour repondre precisement. "
                "Peux-tu preciser une entreprise, une periode, ou un angle (risques, "
                "catalyseurs, marges, guidance) ?"
            )
        }

    memory_context = format_memory_context(
        state.get("memory_summary", ""),
        state.get("memory_window", []),
    )
    message_context = format_chat_context(state.get("messages", []))
    chunks_context = "\n\n---\n\n".join(final_chunks)
    price_context = state.get("price_context", "")
    price_block = f"Contexte prix de marche:\n{price_context}\n\n" if price_context else ""
    prompt = (
        "Contexte conversationnel:\n"
        f"{memory_context}\n\n"
        "Historique de chat recent:\n"
        f"{message_context}\n\n"
        f"{price_block}"
        "Extraits financiers recuperes:\n"
        f"{chunks_context}\n\n"
        f"Question utilisateur: {state['normalized_query']}\n\n"
        "Reponds en francais avec une analyse financiere structuree."
    )
    draft = agent.rag.provider.generate(prompt, temperature=0.1, max_tokens=900)
    return {"draft_answer": draft}


@traceable(name="synthesis_node")
def synthesis_node(agent: Any, state: GraphState) -> GraphState:
    chunk_count = len(state.get("final_chunks", []))
    if chunk_count < 2:
        return {
            "answer": (
                "Le contexte source est limite pour une reponse ferme. "
                "Je peux donner une vue generale, mais elle reste incertaine. "
                "Si tu veux une reponse fiable, precise entreprise et periode."
            )
        }

    prompt = (
        "Synthetise et clarifie cette reponse d'analyse financiere.\n"
        "Conserve uniquement les elements actionnables et factuels.\n"
        "Format strict en 5 sections:\n"
        "1) Synthese\n2) Faits observes\n3) Interpretations\n4) Incertitudes\n5) Conclusion\n\n"
        f"Texte a synthetiser:\n{state.get('draft_answer', '')}"
    )
    final_answer = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=700)
    return {"answer": final_answer}
