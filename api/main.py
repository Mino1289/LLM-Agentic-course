from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, config, health, reports, tools
from api.services.agent_factory import agent_factory
from src.graph.tracing import ensure_langsmith_env
from src.paths import ENV_FILE

load_dotenv(ENV_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_langsmith_env()
    agent_factory.ensure_rag_ready()
    yield


app = FastAPI(title="Finance RAG Hub-and-Spoke API", lifespan=lifespan)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(config.router)
app.include_router(tools.router)
app.include_router(chat.router)
app.include_router(reports.router)
