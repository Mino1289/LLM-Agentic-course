from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from src.orchestration.state import HubSpokeState
from src.orchestration.intent_router import intent_router_node, route_after_intent_router
from src.orchestration.pm_node import pm_plan_node, pm_synthesis_node
from src.orchestration.spoke_agents import analyze_parallel_node
from src.orchestration.compliance_node import compliance_validator_node, route_after_compliance
from src.orchestration.human_review_node import human_review_node, human_approve_node, human_reject_node
from src.orchestration.executor_node import executor_trader_node
from src.orchestration.simple_agent_node import simple_agent_node

_LOGGER = logging.getLogger("src.orchestration.hub_graph")


class HubAndSpokeGraph:
    def __init__(self, agent: Any, max_spoke_iterations: int = 3):
        self.agent = agent
        self.max_spoke_iterations = max_spoke_iterations
        self.graph = self._build_graph()

    def _bind(self, fn):
        return partial(fn, self.agent)

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(HubSpokeState)

        builder.add_node("intent_router", self._bind(intent_router_node))
        builder.add_node("simple_agent", self._bind(simple_agent_node))
        builder.add_node("pm_plan", self._bind(pm_plan_node))
        builder.add_node("analyze_parallel", self._bind(analyze_parallel_node))
        builder.add_node("pm_synthesis", self._bind(pm_synthesis_node))
        builder.add_node("compliance_validator", self._bind(compliance_validator_node))
        builder.add_node("human_review", self._bind(human_review_node))
        builder.add_node("human_approve", self._bind(human_approve_node))
        builder.add_node("human_reject", self._bind(human_reject_node))
        builder.add_node("executor_trader", self._bind(executor_trader_node))

        builder.set_entry_point("intent_router")
        builder.add_conditional_edges("intent_router", route_after_intent_router, {
            "simple_agent": "simple_agent",
            "pm_plan": "pm_plan",
        })
        builder.add_edge("simple_agent", END)
        builder.add_edge("pm_plan", "analyze_parallel")
        builder.add_edge("analyze_parallel", "pm_synthesis")
        builder.add_edge("pm_synthesis", "compliance_validator")
        builder.add_conditional_edges("compliance_validator", route_after_compliance, {
            "pm_plan": "pm_plan",
            "human_review": "human_review",
        })
        builder.add_edge("human_review", END)
        builder.add_edge("human_approve", "executor_trader")
        builder.add_edge("human_reject", END)
        builder.add_edge("executor_trader", END)

        return builder.compile()

    def _initial_state(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> HubSpokeState:
        return {
            "query": query,
            "messages": messages or [],
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "intent_route": "complex",
            "spoke_events": [],
            "spoke_iterations": 0,
            "compliance_retries": 0,
            "tool_events": [],
            "stats": {},
        }

    async def ainvoke(self, state: HubSpokeState) -> HubSpokeState:
        bound = {k: v for k, v in state.items() if v is not None}
        return await self.graph.ainvoke(bound)

    async def arun(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> HubSpokeState:
        return await self.ainvoke(self._initial_state(query, conversation_id, messages))

    def invoke(self, state: HubSpokeState) -> HubSpokeState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError("Cannot call invoke() from inside a running event loop. Use await arun() instead.")
        return asyncio.run(self.ainvoke(state))

    async def astream(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        initial_state = self._initial_state(query, conversation_id, messages)
        last_state: HubSpokeState | None = None

        spoke_event_sinks: dict[str, list[str]] = {}

        async for event in self.graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in (
                "intent_router", "pm_plan", "pm_synthesis",
                "analyze_parallel",
                "compliance_validator", "executor_trader",
                "human_review", "simple_agent",
            ):
                agent_labels = {
                    "intent_router": "Intent Router",
                    "pm_plan": "Portfolio Manager",
                    "pm_synthesis": "Portfolio Manager",
                    "analyze_parallel": "Analystes (Fundamental + Quant)",
                    "compliance_validator": "Compliance Validator",
                    "executor_trader": "Executor Trader",
                    "human_review": "Human Review",
                    "simple_agent": "Simple Agent",
                }
                label = agent_labels.get(name, name)
                spoke_event_sinks[name] = []
                yield {"event": "on_spoke_event", "agent": label, "status": "running", "message": f"{label} en cours...", "tool_events": []}

            elif kind == "on_chain_end" and name in spoke_event_sinks:
                agent_labels = {
                    "intent_router": "Intent Router",
                    "pm_plan": "Portfolio Manager",
                    "pm_synthesis": "Portfolio Manager",
                    "analyze_parallel": "Analystes (Fundamental + Quant)",
                    "compliance_validator": "Compliance Validator",
                    "executor_trader": "Executor Trader",
                    "human_review": "Human Review",
                    "simple_agent": "Simple Agent",
                }
                label = agent_labels.get(name, name)
                yield {"event": "on_spoke_event", "agent": label, "status": "completed", "message": f"{label} terminé.", "tool_events": []}

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                delta = ""
                if chunk is not None:
                    delta = getattr(chunk, "content", None) or ""
                    if not delta and isinstance(chunk, dict):
                        delta = chunk.get("content", "") or ""
                if delta:
                    yield {"event": "on_llm_token", "token": delta}
                continue

            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    last_state = output

            yield event

        if last_state is not None:
            yield {"event": "on_graph_end", "state": last_state}

    async def resume_after_human(
        self,
        state: HubSpokeState,
        approved: bool,
    ) -> HubSpokeState:
        state["human_approved"] = approved
        state["human_review_pending"] = False

        if approved:
            result = await executor_trader_node(self.agent, state)
            state.update(result)
        else:
            state["answer"] = "Ordre annulé par l'utilisateur."

        return state
