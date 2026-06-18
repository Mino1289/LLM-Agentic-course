from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from src.graph.tracing import traceable
from src.orchestration._spoke_helpers import run_spoke_agent
from src.orchestration.progress import emit_agent_progress
from src.orchestration.prompts import (
    FUNDAMENTAL_ANALYST_PROMPT,
    QUANTITATIVE_ANALYST_PROMPT,
)
from src.orchestration.state import HubSpokeState

_LOGGER = logging.getLogger("src.orchestration.spoke_agents")

FUNDAMENTAL_TOOLS = ["sec_filings_rag_tool", "get_news_tool"]
QUANTITATIVE_TOOLS = ["market_price_tool", "portfolio_history_tool"]


@traceable(name="fundamental_analyst")
async def fundamental_analyst_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    plan = state.get("analysis_plan", "")
    query = state.get("normalized_query") or state.get("query", "")
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Fundamental Analyst",
            "status": "running",
            "message": "Analyse des filings SEC et des actualités...",
            "tool_events": [],
        }
    )
    emit_agent_progress(
        "Fundamental Analyst",
        "running",
        "Analyse des filings SEC et des actualités...",
    )

    today = datetime.now(UTC).date().isoformat()
    task = f"""Date actuelle : {today}
Requête utilisateur : {query}

Plan du Portfolio Manager :
{plan}

Réalise une analyse fondamentale. Utilise sec_filings_rag_tool et get_news_tool pour collecter des données.
Fournis un rapport structuré couvrant : risques métier, santé financière, sentiment de marché, développements clés."""

    try:
        report, spoke_stats = await run_spoke_agent(
            agent,
            FUNDAMENTAL_ANALYST_PROMPT,
            task,
            FUNDAMENTAL_TOOLS,
            dict(state),
            max_iterations=3,
        )
        spoke_events.append(
            {
                "agent": "Fundamental Analyst",
                "status": "completed",
                "message": "Analyse fondamentale terminée.",
                "tool_events": [],
            }
        )
        emit_agent_progress(
            "Fundamental Analyst",
            "completed",
            "Analyse fondamentale terminée.",
        )
        return {
            "fundamental_report": report,
            "spoke_events": spoke_events,
            "stats": spoke_stats,
        }
    except Exception as e:
        _LOGGER.exception("Fundamental analyst failed")
        spoke_events.append(
            {
                "agent": "Fundamental Analyst",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {"fundamental_report": f"Error: {e}", "spoke_events": spoke_events}


@traceable(name="quantitative_analyst")
async def quantitative_analyst_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    plan = state.get("analysis_plan", "")
    query = state.get("normalized_query") or state.get("query", "")
    spoke_events = list(state.get("spoke_events") or [])

    spoke_events.append(
        {
            "agent": "Quantitative Analyst",
            "status": "running",
            "message": "Analyse des prix et de l'historique du portefeuille...",
            "tool_events": [],
        }
    )
    emit_agent_progress(
        "Quantitative Analyst",
        "running",
        "Analyse des prix et de l'historique du portefeuille...",
    )

    today = datetime.now(UTC).date().isoformat()
    task = f"""Date actuelle : {today}
Requête utilisateur : {query}

Plan du Portfolio Manager :
{plan}

Réalise une analyse quantitative. Utilise market_price_tool et portfolio_history_tool pour collecter des données.
Fournis un rapport structuré couvrant : tendances de prix, volatilité, adéquation au portefeuille, métriques de risque."""

    try:
        report, spoke_stats = await run_spoke_agent(
            agent,
            QUANTITATIVE_ANALYST_PROMPT,
            task,
            QUANTITATIVE_TOOLS,
            dict(state),
            max_iterations=3,
        )
        spoke_events.append(
            {
                "agent": "Quantitative Analyst",
                "status": "completed",
                "message": "Analyse quantitative terminée.",
                "tool_events": [],
            }
        )
        emit_agent_progress(
            "Quantitative Analyst",
            "completed",
            "Analyse quantitative terminée.",
        )
        return {
            "quantitative_report": report,
            "spoke_events": spoke_events,
            "stats": spoke_stats,
        }
    except Exception as e:
        _LOGGER.exception("Quantitative analyst failed")
        spoke_events.append(
            {
                "agent": "Quantitative Analyst",
                "status": "failed",
                "message": f"Erreur: {e}",
                "tool_events": [],
            }
        )
        return {"quantitative_report": f"Error: {e}", "spoke_events": spoke_events}


@traceable(name="analyze_parallel")
async def analyze_parallel_node(agent: Any, state: HubSpokeState) -> HubSpokeState:
    results = await asyncio.gather(
        fundamental_analyst_node(agent, state),
        quantitative_analyst_node(agent, state),
    )
    merged: dict[str, Any] = {}
    all_events: list[dict[str, Any]] = []
    merged_stats: dict[str, Any] = dict(state.get("stats") or {})
    merged_price_series: list[dict[str, Any]] = list(state.get("price_series") or [])
    all_chunks: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    for result in results:
        events = result.pop("spoke_events", [])
        if isinstance(events, list):
            all_events.extend(events)
        result_stats = result.pop("stats", {})
        if result_stats:
            # Extraire les données internes AVANT de fusionner, sinon elles
            # polluent les stats affichées (chunks RAG, séries de prix).
            chunks = result_stats.pop("final_chunks", [])
            metadatas = result_stats.pop("final_metadatas", [])
            spoke_series = result_stats.pop("price_series", None)
            if spoke_series:
                from src.graph.tool_execution_node import _merge_price_series

                merged_price_series = _merge_price_series(
                    merged_price_series, spoke_series
                )
            if chunks:
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
            # Sommer les compteurs par spoke au lieu de les écraser.
            for counter in ("spoke_tool_calls", "spoke_llm_iterations"):
                if counter in result_stats:
                    merged_stats[counter] = (
                        merged_stats.get(counter, 0) + result_stats.pop(counter)
                    )
            merged_stats.update(result_stats)
        merged.update(result)
    merged["spoke_events"] = all_events
    merged["stats"] = merged_stats
    if merged_price_series:
        merged["price_series"] = merged_price_series
    if all_chunks:
        merged["final_chunks"] = all_chunks
        merged["final_metadatas"] = all_metadatas
    return merged
