from __future__ import annotations

from fastapi import APIRouter

from api.services.agent_factory import agent_factory

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict:
    rag_ready = agent_factory.ensure_rag_ready()
    return {"status": "ready" if rag_ready else "degraded", "rag_indexed": rag_ready}
