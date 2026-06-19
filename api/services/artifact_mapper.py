from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("api.services.artifact_mapper")

from api.schemas.artifacts import (
    AgentStep,
    MessageArtifacts,
    PricePoint,
    PriceSeriesArtifact,
    PriceSeriesStats,
    ReportArtifact,
    SourceItem,
    StatItem,
    TradeProposal,
)
from api.schemas.chat import ChatMessageDTO, ChatResponse, HumanReviewPayload
from src.orchestration.pm_decision import enrich_pm_decision
from src.orchestration.trade_intent import has_trade_proposal
from src.paths import REPORTS_DIR


def build_sources(
    chunks: list[str], metadatas: list[dict[str, Any]]
) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for i, chunk in enumerate(chunks):
        meta = metadatas[i] if i < len(metadatas) else {}
        ticker = str(meta.get("ticker", "UNKNOWN"))
        source = str(meta.get("source", "unknown"))
        section = str(meta.get("section", "unknown"))
        year = str(meta.get("year", "unknown"))
        chunk_index = meta.get("chunk_index", i)
        sources.append(
            SourceItem(
                id=f"s{i + 1}",
                title=f"{ticker} | {source} | {section}",
                excerpt=chunk[:500] + ("..." if len(chunk) > 500 else ""),
                meta=f"{source} · {year} · chunk {chunk_index}",
                ticker=ticker,
                section=section,
            )
        )
    return sources


def tool_events_to_steps(tool_events: list[dict[str, Any]]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for idx, event in enumerate(tool_events, start=1):
        tool_name = event.get("tool", "outil")
        summary = event.get("args_summary", "")
        steps.append(
            AgentStep(
                id=str(idx),
                text=f"L'agent utilise l'outil **{tool_name}** — {summary}",
            )
        )
    return steps


def stats_to_items(stats: dict[str, Any], locale: str = "fr") -> list[StatItem]:
    if not stats:
        return []
    labels_fr = {
        "tokens": "Tokens utilisés",
        "chunks": "Chunks RAG",
        "iterations": "Itérations LLM",
        "tools": "Outils appelés",
        "candidates": "Candidats",
        "final": "Final",
    }
    labels_en = {
        "tokens": "Tokens used",
        "chunks": "RAG chunks",
        "iterations": "LLM iterations",
        "tools": "Tools called",
        "candidates": "Candidates",
        "final": "Final",
    }
    labels = labels_fr if locale == "fr" else labels_en
    items: list[StatItem] = []

    total_tokens = stats.get("llm_total_tokens", 0) + stats.get("guard_total_tokens", 0)
    if not total_tokens:
        total_tokens = stats.get("estimated_context_tokens", 0)
    if total_tokens:
        items.append(
            StatItem(id="tokens", label=labels["tokens"], value=f"{total_tokens:,}")
        )

    chunks = stats.get("chunks_used") or stats.get("rerank_final_count")
    if chunks:
        items.append(StatItem(id="chunks", label=labels["chunks"], value=str(chunks)))

    spoke_llm = stats.get("spoke_llm_iterations") or stats.get("agent_iterations")
    if spoke_llm:
        items.append(
            StatItem(id="iterations", label=labels["iterations"], value=str(spoke_llm))
        )

    spoke_tc = stats.get("spoke_tool_calls")
    if spoke_tc:
        items.append(StatItem(id="tools", label=labels["tools"], value=str(spoke_tc)))

    if stats.get("retrieval_candidate_count"):
        items.append(
            StatItem(
                id="candidates",
                label=labels["candidates"],
                value=str(stats["retrieval_candidate_count"]),
            )
        )

    if len(items) < 6 and stats.get("intent_route"):
        route = str(stats.get("intent_route", ""))
        route_label = {
            "simple": "Agent simple" if locale == "fr" else "Simple agent",
            "complex": "Hub-and-Spoke (multi-agents)",
        }.get(route, route)
        items.append(StatItem(id="route", label="Mode", value=route_label))

    return items[:6]


def _format_size(path: Path) -> str:
    if not path.is_file():
        return "—"
    size = path.stat().st_size
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} Mo"
    if size >= 1000:
        return f"{size / 1000:.0f} Ko"
    return f"{size} o"


def reports_to_artifacts(
    report_artifacts: list[dict[str, Any]],
) -> list[ReportArtifact]:
    results: list[ReportArtifact] = []
    for idx, artifact in enumerate(report_artifacts, start=1):
        path = Path(str(artifact.get("path", "")))
        filename = artifact.get("filename") or path.name
        fmt = str(artifact.get("format") or path.suffix.lstrip(".") or "pdf").lower()
        report_type = "pdf" if "pdf" in fmt else "md"
        results.append(
            ReportArtifact(
                id=f"r{idx}",
                name=str(artifact.get("title") or filename),
                size=_format_size(path),
                type=report_type,
                download_url=f"/api/reports/{filename}",
            )
        )
    return results


def _normalize_price_point(raw: Any, fallback_date: str) -> PricePoint | None:
    if not isinstance(raw, dict):
        return None
    date_val = raw.get("date") or raw.get("Date") or fallback_date
    if not date_val:
        return None
    close_raw = raw.get("close", raw.get("Close"))
    try:
        close_val = float(close_raw)
    except (TypeError, ValueError):
        return None
    if not (close_val == close_val):  # NaN
        return None
    return PricePoint(date=str(date_val), close=close_val)


def price_series_to_artifacts(
    price_series: list[dict[str, Any]],
) -> list[PriceSeriesArtifact]:
    charts: list[PriceSeriesArtifact] = []
    for idx, series in enumerate(price_series, start=1):
        raw_stats = series.get("stats") or {}
        stats = PriceSeriesStats(
            perf_pct=raw_stats.get("perf_pct"),
            vol_ann_pct=raw_stats.get("vol_ann_pct"),
            max_drawdown_pct=raw_stats.get("max_drawdown_pct"),
            close_min=raw_stats.get("close_min"),
            close_max=raw_stats.get("close_max"),
            close_last=raw_stats.get("close_last"),
            high_date=raw_stats.get("high_date"),
            low_date=raw_stats.get("low_date"),
        )
        start_date = str(series.get("start_date", ""))
        points: list[PricePoint] = []
        for point_idx, raw_point in enumerate(series.get("points") or []):
            fallback_date = start_date or f"point-{point_idx + 1}"
            normalized = _normalize_price_point(raw_point, fallback_date)
            if normalized is not None:
                points.append(normalized)
        if raw_stats and not points:
            _LOGGER.warning(
                "Price series for %s has stats but no chart points after mapping",
                series.get("ticker", "N/A"),
            )
        charts.append(
            PriceSeriesArtifact(
                id=f"p{idx}",
                ticker=str(series.get("ticker", "N/A")),
                start_date=str(series.get("start_date", "")),
                end_date=str(series.get("end_date", "")),
                points=points,
                stats=stats,
            )
        )
    return charts


def _enrich_pm_decision(state: dict[str, Any]) -> dict[str, Any]:
    return enrich_pm_decision(state)


def state_to_trade(state: dict[str, Any]) -> TradeProposal | None:
    if not has_trade_proposal(state):
        return None
    decision = _enrich_pm_decision(state)
    verdict = str(state.get("compliance_verdict", ""))
    risk = "low" if verdict == "PASS" else "high" if verdict == "FAIL" else "medium"
    qty = decision.get("qty", decision.get("quantity", "N/A"))
    try:
        qty_val: int | float | str = int(qty)
    except (TypeError, ValueError):
        qty_val = str(qty)
    return TradeProposal(
        ticker=str(decision.get("ticker") or "Voir justification"),
        side=str(decision.get("side", "N/A")).upper(),
        quantity=qty_val,
        order_type=str(decision.get("order_type", "market")),
        limit_price=str(decision.get("limit_price", "")) or None,
        risk_level=risk,
        justification=str(decision.get("response") or state.get("answer", "")),
        compliance_verdict=verdict or None,
        compliance_detail=str(state.get("compliance_detail", "")) or None,
    )


def build_trade_for_review(state: dict[str, Any]) -> TradeProposal:
    trade = state_to_trade(state)
    if trade is not None:
        return trade
    raise ValueError("No trade proposal in state")


def state_to_artifacts(state: dict[str, Any], locale: str = "fr") -> MessageArtifacts:
    chunks = state.get("final_chunks") or []
    metadatas = state.get("final_metadatas") or []
    tool_events = state.get("tool_events") or []
    stats = state.get("stats") or {}
    report_artifacts = state.get("report_artifacts") or []
    price_series = state.get("price_series") or []

    trade = None
    if state.get("human_review_pending"):
        trade = build_trade_for_review(state)

    return MessageArtifacts(
        steps=tool_events_to_steps(tool_events),
        sources=build_sources(chunks, metadatas),
        reports=reports_to_artifacts(report_artifacts),
        stats=stats_to_items(stats, locale),
        price_charts=price_series_to_artifacts(price_series),
        trade=trade,
    )


def state_to_response(
    state: dict[str, Any],
    conversation_id: str,
    run_id: str | None = None,
    locale: str = "fr",
) -> ChatResponse:
    answer = str(state.get("answer", ""))
    artifacts = state_to_artifacts(state, locale)
    if not state.get("human_review_pending"):
        artifacts.trade = None

    return ChatResponse(
        conversation_id=conversation_id,
        run_id=run_id,
        answer=answer,
        human_review_pending=bool(state.get("human_review_pending")),
        artifacts=artifacts,
    )


def state_to_human_review(
    state: dict[str, Any],
    conversation_id: str,
    run_id: str,
    locale: str = "fr",
) -> HumanReviewPayload:
    trade = build_trade_for_review(state)
    artifacts = state_to_artifacts(state, locale)
    artifacts.trade = trade
    return HumanReviewPayload(
        run_id=run_id,
        conversation_id=conversation_id,
        answer=str(state.get("answer", "")),
        trade=trade,
        artifacts=artifacts,
    )


def message_dto_from_assistant(
    state: dict[str, Any],
    message_id: str,
    timestamp: str,
    locale: str = "fr",
) -> ChatMessageDTO:
    return ChatMessageDTO(
        id=message_id,
        role="assistant",
        content=str(state.get("answer", "")),
        timestamp=timestamp,
        artifacts=state_to_artifacts(state, locale),
    )
