from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.graph.state import GraphState, ToolEvent


class SpokeEvent(TypedDict, total=False):
    agent: str
    status: Literal["running", "completed", "failed"]
    message: str
    tool_events: list[ToolEvent]


class HubSpokeState(GraphState, total=False):
    intent_route: Literal["simple", "complex"]
    trade_requested: bool
    analysis_plan: str
    fundamental_report: str
    quantitative_report: str
    pm_decision: dict[str, Any]
    compliance_verdict: str
    compliance_reasons: list[str]
    compliance_detail: str
    human_approved: bool
    human_review_pending: bool
    trade_result: dict[str, Any]
    spoke_events: list[SpokeEvent]
    spoke_iterations: int
    compliance_retries: int
