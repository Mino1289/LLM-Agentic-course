from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

from src.graph.flow import FinanceLangGraphAgent
from src.orchestration.hub_graph import HubAndSpokeGraph
from src.rag.core import HybridRAG

from api.schemas.settings import AgentSettings


class AgentFactory:
    def __init__(self) -> None:
        self._agent_cache: dict[str, FinanceLangGraphAgent] = {}
        self._rag_ready = False

    def ensure_rag_ready(self) -> bool:
        if self._rag_ready:
            return True
        try:
            rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
            rag.load_and_index_data(max_new_embeddings=0)
            self._rag_ready = True
            return True
        except Exception:
            return False

    def get_hub_graph(self, settings: AgentSettings) -> HubAndSpokeGraph:
        key = settings.cache_key()
        if key not in self._agent_cache:
            rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
            rag.load_and_index_data(max_new_embeddings=0)
            self._rag_ready = True
            self._agent_cache[key] = FinanceLangGraphAgent(rag=rag, **settings.to_agent_kwargs())
        agent = self._agent_cache[key]
        return HubAndSpokeGraph(agent, max_spoke_iterations=settings.max_spoke_iterations)


agent_factory = AgentFactory()
