from __future__ import annotations

from typing import Any

from rag.nodes.memory_store import format_chat_context, format_memory_context
from rag.nodes.prompt_context import format_universe_hint
from rag.nodes.state import GraphState
from rag.nodes.tracing import traceable


@traceable(name="answer_generate_node")
def answer_generate_node(agent: Any, state: GraphState) -> GraphState:
    final_chunks = state.get("final_chunks", [])
    if not final_chunks:
        universe_hint = format_universe_hint(agent, max_items=10)
        return {
            "draft_answer": (
                "Je ne trouve pas assez de sources fiables pour repondre precisement. "
                "Peux-tu preciser une entreprise, une periode, ou un angle (risques, "
                "catalyseurs, marges, guidance) ? "
                f"Exemples dans la base: {universe_hint}."
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
        "Tu es un assistant d'analyse financiere strictement ancre aux sources fournies.\n"
        "Regles obligatoires anti-hallucination:\n"
        "1) Utilise uniquement les informations presentes dans 'Extraits financiers recuperes' "
        "et dans 'Contexte prix de marche' (si present).\n"
        "2) N'ajoute aucun fait externe, aucune connaissance generale non visible dans ces extraits.\n"
        "3) Si une information demandee n'est pas dans les extraits, dis-le explicitement "
        "('information non disponible dans les sources recuperées').\n"
        "4) N'invente ni chiffres, ni dates, ni citations, ni evenements.\n"
        "5) Priorise precision et tracabilite sur eloquence.\n\n"
        "Contexte conversationnel:\n"
        f"{memory_context}\n\n"
        "Historique de chat recent:\n"
        f"{message_context}\n\n"
        f"{price_block}"
        "Extraits financiers recuperes:\n"
        f"{chunks_context}\n\n"
        f"Question utilisateur: {state['normalized_query']}\n\n"
        "Reponds en francais avec une analyse financiere concise et fidele aux extraits."
    )
    draft = agent.rag.provider.generate(prompt, temperature=0.1, max_tokens=900)
    return {"draft_answer": draft}


@traceable(name="general_chat_node")
def general_chat_node(agent: Any, state: GraphState) -> GraphState:
    message_context = format_chat_context(state.get("messages", []), keep_last=8)
    query = state.get("normalized_query", "")
    system_prompt = (
        "Tu es un assistant conversationnel utile et concis. "
        "Si la question est hors finance, reponds normalement. "
        "Si l'utilisateur bascule vers la finance, invite-le doucement a preciser entreprise/periode si necessaire."
    )
    prompt = (
        "Historique recent:\n"
        f"{message_context}\n\n"
        f"Message utilisateur: {query}\n\n"
        "Reponds en francais, ton naturel, sans inventer d'informations factuelles incertaines."
    )
    answer = agent.rag.provider.generate(
        prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=280,
    )
    stats = state.get("stats", {})
    stats.update(
        {
            "general_chat": True,
            "chunks_used": 0,
            "estimated_context_tokens": 0,
            "gc_applied": False,
        }
    )
    return {"answer": answer, "stats": stats}


@traceable(name="off_topic_block_node")
def off_topic_block_node(_agent: Any, state: GraphState) -> GraphState:
    query = state.get("normalized_query", "").strip()
    stats = state.get("stats", {})
    stats.update(
        {
            "off_topic_blocked": True,
            "general_chat": False,
            "chunks_used": 0,
            "estimated_context_tokens": 0,
            "gc_applied": False,
        }
    )
    answer = (
        "Je prefere rester focalise sur l'analyse financiere de l'univers couvert "
        "(entreprises, risques, catalyseurs, performance, rapports SEC, etc.). "
        "Je ne peux pas traiter cette demande telle quelle.\n\n"
        "Si tu veux, reformule en version finance (ex: 'compare NVDA et AMD sur les risques 2024')."
    )
    if query:
        answer += f"\n\nRequete recue: {query}"
    return {"answer": answer, "off_topic_blocked": True, "stats": stats}


@traceable(name="coverage_info_node")
def coverage_info_node(agent: Any, state: GraphState) -> GraphState:
    universe_hint = format_universe_hint(agent, max_items=20)
    stats = state.get("stats", {})
    stats.update(
        {
            "coverage_info": True,
            "chunks_used": 0,
            "estimated_context_tokens": 0,
            "gc_applied": False,
        }
    )
    answer = (
        "Je peux te repondre sur les entreprises couvertes dans la base actuelle:\n"
        f"{universe_hint}.\n\n"
        "Tu peux poser une question par ticker + periode (ex: 'NVDA 2024: risques et catalyseurs')."
    )
    return {"answer": answer, "stats": stats}


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
        "Tu es l'etape de synthese finale d'un assistant finance.\n"
        "Ta mission: repondre clairement a la question utilisateur en choisissant le format le plus adapte.\n\n"
        "Regles obligatoires:\n"
        "1) Priorite: reponds d'abord a la question (pas de blabla).\n"
        "2) Format adaptatif (ne force PAS 5 sections):\n"
        "   - question simple / suivi court -> reponse concise en 1-2 paragraphes ou bullets.\n"
        "   - question comparative / semi-complexe -> structure legere avec 2-3 parties max.\n"
        "   - question complexe explicite -> structure plus complete (jusqu'a 5 parties si utile).\n"
        "3) Reste factuel et actionnable; evite les repetitions.\n"
        "4) Si le contexte est incertain/incomplet, signale-le en 1 phrase.\n"
        "5) Longueur cible: 120-220 mots (peut depasser legerement si necessaire).\n\n"
        f"Question utilisateur: {state.get('normalized_query', '')}\n"
        f"Texte brouillon a synthetiser:\n{state.get('draft_answer', '')}"
    )
    final_answer = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=520)
    return {"answer": final_answer}
