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
                "I cannot find enough reliable evidence in the retrieved sources to answer precisely. "
                "Please specify a company, timeframe, or angle (risks, catalysts, margins, guidance). "
                f"Covered companies include: {universe_hint}."
            )
        }

    memory_context = format_memory_context(
        state.get("memory_summary", ""),
        state.get("memory_window", []),
    )
    message_context = format_chat_context(state.get("messages", []))
    chunks_context = "\n\n---\n\n".join(final_chunks)
    price_context = state.get("price_context", "")
    universe_hint = format_universe_hint(agent, max_items=20)
    price_block = f"Market price context:\n{price_context}\n\n" if price_context else ""
    prompt = (
        "You are a financial analysis assistant strictly grounded in provided evidence.\n"
        "Mandatory anti-hallucination rules:\n"
        "1) Use only information present in 'Retrieved financial excerpts' and 'Market price context' (if present).\n"
        "2) Do not add external facts or general knowledge not visible in those inputs.\n"
        "3) If requested information is missing, explicitly state: "
        "'information not available in retrieved sources'.\n"
        "4) Do not invent numbers, dates, quotes, or events.\n"
        "5) Prioritize precision and traceability over style.\n"
        "6) Keep the answer focused on covered companies.\n"
        "7) Distinguish filing year/date from reporting-period year.\n"
        "   Example: a 2026 filing can report FY2025 metrics.\n"
        "8) Never output a blanket 'no data' claim if excerpts exist; instead state exactly "
        "which period is evidenced and what is missing.\n\n"
        f"Covered companies (tickers): {universe_hint}\n\n"
        "Conversation memory context:\n"
        f"{memory_context}\n\n"
        "Recent chat history:\n"
        f"{message_context}\n\n"
        f"{price_block}"
        "Retrieved financial excerpts:\n"
        f"{chunks_context}\n\n"
        f"User question: {state['normalized_query']}\n\n"
        "Respond in French with a concise, evidence-faithful financial analysis."
    )
    draft = agent.rag.provider.generate(prompt, temperature=0.1, max_tokens=900)
    return {"draft_answer": draft}


@traceable(name="general_chat_node")
def general_chat_node(agent: Any, state: GraphState) -> GraphState:
    message_context = format_chat_context(state.get("messages", []), keep_last=8)
    query = state.get("normalized_query", "")
    system_prompt = (
        "You are a concise and helpful conversational assistant. "
        "If the question is non-finance small talk, respond naturally. "
        "If the user switches back to finance, gently ask for company/timeframe when needed."
    )
    prompt = (
        "Recent chat history:\n"
        f"{message_context}\n\n"
        f"User message: {query}\n\n"
        "Respond in French, naturally, without inventing uncertain factual claims."
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
    disclaimer = (
        "\n\n---\n"
        "Avertissement: cette reponse est informative et basee sur les sources disponibles du RAG. "
        "Elle ne constitue pas un conseil financier personnalise. Investir comporte un risque de perte en capital."
    )
    chunk_count = len(state.get("final_chunks", []))
    if chunk_count == 0:
        return {
            "answer": (
                "Aucun extrait source pertinent n'a ete retrouve pour repondre de facon fiable. "
                "Si tu veux une reponse fiable, precise entreprise et periode."
            )
            + disclaimer
        }

    if chunk_count == 1:
        draft = (state.get("draft_answer", "") or "").strip()
        if draft:
            return {
                "answer": (
                    draft
                    + "\n\nNote: le contexte source est limite a un seul extrait; "
                    "la conclusion doit donc rester prudente."
                    + disclaimer
                )
            }

    prompt = (
        "You are the final synthesis step of a finance assistant.\n"
        "Your task: answer the user clearly, choosing the most appropriate format.\n\n"
        "Mandatory rules:\n"
        "1) Lead with a direct answer to the question.\n"
        "2) Use adaptive formatting (do NOT force 5 sections):\n"
        "   - simple follow-up -> concise paragraphs or short bullets.\n"
        "   - comparative / medium complexity -> lightweight 2-3 part structure.\n"
        "   - explicitly complex request -> fuller structure only if useful.\n"
        "3) Stay factual, actionable, and avoid repetition.\n"
        "4) If evidence is incomplete, state uncertainty in one sentence.\n"
        "5) Target length: 120-220 words (slightly longer only if necessary).\n"
        "6) Preserve period accuracy: separate filing year from fiscal-year metrics.\n"
        "   If a filing dated 2026 reports FY2025 values, state that explicitly.\n"
        "7) If requested period is not directly evidenced, provide the closest evidenced period "
        "and explain the gap.\n\n"
        f"User question: {state.get('normalized_query', '')}\n"
        f"Draft answer to synthesize:\n{state.get('draft_answer', '')}"
    )
    final_answer = agent.rag.provider.generate(prompt, temperature=0.0, max_tokens=520)
    return {"answer": (final_answer or "").strip() + disclaimer}
