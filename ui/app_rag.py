import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

# Reduce noisy transformers warnings triggered by Streamlit module introspection.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.langsmith_env import ensure_langsmith_env

ensure_langsmith_env()

from rag.hybrid_rag import HybridRAG
from rag.langgraph_flow import FinanceLangGraphAgent
from rag.llm_provider import build_llm_config_from_env
from rag.tools import get_tool_definitions
from ui.streaming import run_stream

st.set_page_config(page_title="Finance RAG LangGraph", page_icon="📈", layout="wide")
st.title("📈 Finance RAG LangGraph")
st.markdown(
    "Assistant d'analyse financière (agent à outils, style MCP) : SEC filings, earnings calls, "
    "prix de marché et export de rapports. **Pas de serveur MCP séparé** — les outils sont exposés "
    "au LLM via function calling (`get_tool_definitions` / `tools_node`)."
)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je peux comparer NVDA, AMD et MSFT, interroger les 10-K/10-Q/8-K "
                "et les transcripts earnings, valider des affirmations, simuler une allocation fictive, "
                "récupérer les prix et générer un rapport téléchargeable."
            ),
        }
    ]


@st.cache_resource
def build_agent(
    memory_window_size: int,
    summarize_every_n_turns: int,
    max_context_chunks: int,
    max_context_tokens: int,
    decompose_query_count: int,
    price_max_days: int,
    price_max_points: int,
    price_max_tickers: int,
    price_default_days: int,
    price_max_attempts: int,
    max_tool_iterations: int,
):
    rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
    rag.load_and_index_data(max_new_embeddings=0)
    return FinanceLangGraphAgent(
        rag=rag,
        memory_window_size=memory_window_size,
        summarize_every_n_turns=summarize_every_n_turns,
        max_context_chunks=max_context_chunks,
        max_context_tokens=max_context_tokens,
        decompose_query_count=decompose_query_count,
        price_max_days=price_max_days,
        price_max_points=price_max_points,
        price_max_tickers=price_max_tickers,
        price_default_days=price_default_days,
        price_max_attempts=price_max_attempts,
        max_tool_iterations=max_tool_iterations,
    )


def format_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return "aucun"
    return " | ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def build_sources(chunks: list[str], metadatas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for i, chunk in enumerate(chunks):
        meta = metadatas[i] if i < len(metadatas) else {}
        sources.append(
            {
                "ticker": meta.get("ticker", "UNKNOWN"),
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "unknown"),
                "year": meta.get("year", "unknown"),
                "file_type": meta.get("file_type", "unknown"),
                "chunk_index": meta.get("chunk_index", i),
                "source_index": i + 1,
                "chunk": chunk,
            }
        )
    return sources


def render_tool_thoughts(tool_events: list[dict[str, Any]], key_prefix: str) -> None:
    if not tool_events:
        return
    with st.expander("Réflexion de l'agent (outils utilisés)", expanded=True):
        for idx, event in enumerate(tool_events, start=1):
            tool_name = event.get("tool", "outil")
            summary = event.get("args_summary", "")
            st.markdown(f"{idx}. L'agent utilise l'outil **{tool_name}** — {summary}")


def render_report_downloads(report_artifacts: list[dict[str, Any]], key_prefix: str) -> None:
    if not report_artifacts:
        return
    st.markdown("**Rapports générés**")
    for idx, artifact in enumerate(report_artifacts):
        path = Path(artifact.get("path", ""))
        if not path.is_file():
            st.warning(f"Fichier introuvable: {path}")
            continue
        data = path.read_bytes()
        st.download_button(
            label=f"Télécharger {artifact.get('filename', path.name)}",
            data=data,
            file_name=artifact.get("filename", path.name),
            mime="text/markdown" if path.suffix == ".md" else "application/octet-stream",
            key=f"{key_prefix}_report_{idx}",
        )


def render_stats(stats: dict[str, Any]) -> None:
    if not stats:
        return

    cols = st.columns(6)
    cols[0].metric("Chunks", stats.get("chunks_used", 0))
    cols[1].metric("Tokens", stats.get("estimated_context_tokens", 0))
    cols[2].metric("Itérations agent", stats.get("agent_iterations", 0))
    cols[3].metric("Candidats", stats.get("retrieval_candidate_count", 0))
    cols[4].metric("Final", stats.get("rerank_final_count", stats.get("chunks_used", 0)))
    cols[5].metric("Prix", "oui" if stats.get("price_tool_used") else "non")

    retrieval_counts = stats.get("retrieval_candidate_ticker_counts", {})
    rerank_counts = stats.get("rerank_final_ticker_counts", {})
    if retrieval_counts or rerank_counts:
        st.caption(
            "Répartition tickers - "
            f"retrieval: {format_counts(retrieval_counts)} | "
            f"rerank final: {format_counts(rerank_counts)}"
        )
    else:
        st.caption(
            f"GC: {'oui' if stats.get('gc_applied') else 'non'} | "
            f"RAG tool: {'oui' if stats.get('rag_tool_used') else 'non'}"
        )


def render_sources(sources: list[dict[str, Any]], key_prefix: str) -> None:
    if not sources:
        return

    counts = Counter(str(source.get("ticker", "UNKNOWN")) for source in sources)
    summary = ", ".join(f"{ticker}: {count}" for ticker, count in sorted(counts.items()))
    with st.expander(f"Sources consultées ({len(sources)} chunks | {summary})"):
        tickers = ["Tous"] + sorted({str(source.get("ticker", "UNKNOWN")) for source in sources})
        sections = ["Toutes"] + sorted({str(source.get("section", "unknown")) for source in sources})
        col1, col2 = st.columns(2)
        selected_ticker = col1.selectbox("Ticker", tickers, key=f"{key_prefix}_source_ticker")
        selected_section = col2.selectbox("Section", sections, key=f"{key_prefix}_source_section")

        visible_sources = []
        for source in sources:
            if selected_ticker != "Tous" and source.get("ticker") != selected_ticker:
                continue
            if selected_section != "Toutes" and source.get("section") != selected_section:
                continue
            visible_sources.append(source)

        for source_idx, source in enumerate(visible_sources, start=1):
            stable_idx = source.get("source_index", source_idx)
            st.markdown(
                f"**Source {source_idx}** - "
                f"`{source.get('ticker', 'UNKNOWN')}` | "
                f"`{source.get('source', 'unknown')}` | "
                f"`{source.get('section', 'unknown')}` | "
                f"`{source.get('year', 'unknown')}` | "
                f"chunk `{source.get('chunk_index', '?')}`"
            )
            st.text_area(
                "Extrait",
                source.get("chunk", ""),
                height=180,
                disabled=True,
                key=f"{key_prefix}_source_{stable_idx}",
                label_visibility="collapsed",
            )
            st.divider()


def render_debug(message: dict[str, Any], key_prefix: str) -> None:
    stats = message.get("stats", {})
    debug_payload = message.get("debug", {})
    if not stats and not debug_payload:
        return

    with st.expander("Debug RAG"):
        tab_pipeline, tab_tools, tab_filters, tab_stats = st.tabs(
            ["Pipeline", "Outils", "Filtres", "Stats brutes"]
        )

        with tab_pipeline:
            pipeline_rows = {
                "pipeline": stats.get("pipeline"),
                "intent_route": stats.get("intent_route"),
                "agent_iterations": stats.get("agent_iterations"),
                "rag_tool_used": stats.get("rag_tool_used"),
                "price_tool_used": stats.get("price_tool_used"),
                "validate_tool_used": stats.get("validate_tool_used"),
                "simulate_tool_used": stats.get("simulate_tool_used"),
                "report_exported": stats.get("report_exported"),
            }
            st.json({k: v for k, v in pipeline_rows.items() if v not in (None, "", [])})

        with tab_tools:
            events = message.get("tool_events", [])
            if events:
                st.json(events)
            else:
                st.caption("Aucun appel d'outil enregistré.")

        with tab_filters:
            st.json(
                {
                    "normalized_query": debug_payload.get("normalized_query", ""),
                    "metadata_filter": debug_payload.get("metadata_filter", {}),
                    "target_tickers": debug_payload.get("target_tickers", []),
                }
            )

        with tab_stats:
            st.json(stats)


def render_sidebar_models(
    provider_name: str,
    chat_model: str,
    embed_provider: str,
    embed_model: str,
    config_error: str | None = None,
) -> None:
    st.sidebar.subheader("Modèles")
    if config_error:
        st.sidebar.error(f"Config LLM : {config_error}")
        return
    st.sidebar.markdown(f"**Chat** : `{provider_name}` · `{chat_model}`")
    st.sidebar.markdown(f"**Embeddings** : `{embed_provider}` · `{embed_model}`")
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}:
        region = os.getenv("LANGSMITH_REGION", "us").upper()
        project = os.getenv("LANGSMITH_PROJECT", "")
        st.sidebar.caption(f"LangSmith ({region}) · projet `{project}`")


def render_sidebar_tools() -> None:
    st.sidebar.subheader("Outils disponibles")
    st.sidebar.caption(
        "5 outils (contrat type MCP). L'agent les appelle dynamiquement ; "
        "il n'y a pas de serveur MCP stdio/HTTP dans ce projet."
    )
    for tool in get_tool_definitions():
        fn = tool.get("function", {})
        name = fn.get("name", "outil")
        description = fn.get("description", "")
        with st.sidebar.expander(name, expanded=False):
            st.markdown(description)
            params = fn.get("parameters", {}) or {}
            properties = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])
            if properties:
                st.markdown("**Paramètres**")
                for param_name, param_info in properties.items():
                    req = " *(requis)*" if param_name in required else ""
                    param_desc = param_info.get("description", "")
                    st.markdown(f"- `{param_name}`{req} — {param_desc}")


def render_assistant_artifacts(message: dict[str, Any], key_prefix: str) -> None:
    render_tool_thoughts(message.get("tool_events", []), key_prefix)
    render_report_downloads(message.get("report_artifacts", []), key_prefix)
    render_stats(message.get("stats", {}))
    render_sources(message.get("sources", []), key_prefix)
    price_context = message.get("price_context", "")
    if price_context:
        with st.expander("Contexte prix utilisé"):
            st.text(price_context)
    render_debug(message, key_prefix)


st.sidebar.title("Finance RAG Agent")
st.sidebar.caption("LangGraph v2 · semantic chunking · boucle agent ↔ outils")

try:
    llm_config = build_llm_config_from_env()
    render_sidebar_models(
        llm_config.provider,
        llm_config.chat_model,
        llm_config.embedding_provider,
        llm_config.embedding_model,
    )
except Exception as config_exc:
    render_sidebar_models("?", "?", "?", "?", str(config_exc))

render_sidebar_tools()
st.sidebar.divider()

st.sidebar.subheader("Configuration")

memory_window_default = int(os.getenv("MEMORY_WINDOW_SIZE", "6"))
summarize_every_n_turns_default = int(os.getenv("SUMMARIZE_EVERY_N_TURNS", "6"))
max_context_chunks_default = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
max_context_tokens_default = int(os.getenv("MAX_CONTEXT_TOKENS", "3500"))
decompose_query_count_default = int(os.getenv("QUERY_DECOMPOSE_COUNT", "2"))
price_max_days_default = int(os.getenv("PRICE_MAX_DAYS", "180"))
price_max_points_default = int(os.getenv("PRICE_MAX_POINTS", "40"))
price_max_tickers_default = int(os.getenv("PRICE_MAX_TICKERS", "3"))
price_default_days_default = int(os.getenv("PRICE_DEFAULT_DAYS", "90"))
price_max_attempts_default = int(os.getenv("PRICE_MAX_ATTEMPTS", "2"))
max_tool_iterations_default = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))

memory_window_size = st.sidebar.slider("Fenêtre mémoire", min_value=4, max_value=12, value=memory_window_default)
summarize_every_n_turns = st.sidebar.slider(
    "GC après N tours", min_value=4, max_value=16, value=summarize_every_n_turns_default
)
max_context_chunks = st.sidebar.slider("Chunks max", min_value=4, max_value=12, value=max_context_chunks_default)
max_context_tokens = st.sidebar.slider(
    "Tokens contexte max", min_value=1200, max_value=8000, step=100, value=max_context_tokens_default
)
decompose_query_count = st.sidebar.slider(
    "Nb sous-requêtes", min_value=1, max_value=8, value=decompose_query_count_default
)
price_max_days = st.sidebar.slider("Prix max jours", min_value=30, max_value=365, value=price_max_days_default)
price_max_points = st.sidebar.slider("Prix max points", min_value=10, max_value=120, value=price_max_points_default)
price_max_tickers = st.sidebar.slider("Prix max tickers", min_value=1, max_value=5, value=price_max_tickers_default)
price_default_days = st.sidebar.slider(
    "Prix fenêtre défaut (jours)", min_value=15, max_value=180, value=price_default_days_default
)
price_max_attempts = st.sidebar.slider(
    "Prix max tentatives outil", min_value=1, max_value=4, value=price_max_attempts_default
)
max_tool_iterations = st.sidebar.slider(
    "Max itérations agent/outils", min_value=2, max_value=10, value=max_tool_iterations_default
)

if st.sidebar.button("Nouvelle conversation", use_container_width=True):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je peux comparer NVDA, AMD et MSFT, interroger les filings SEC "
                "et earnings calls, valider des affirmations, simuler une allocation fictive "
                "et exporter un rapport."
            ),
        }
    ]
    st.rerun()

st.sidebar.divider()
try:
    agent = build_agent(
        memory_window_size,
        summarize_every_n_turns,
        max_context_chunks,
        max_context_tokens,
        decompose_query_count,
        price_max_days,
        price_max_points,
        price_max_tickers,
        price_default_days,
        price_max_attempts,
        max_tool_iterations,
    )
    st.sidebar.success("Système prêt")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation: {e}")
    st.stop()

for message_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(message.get("content", ""))
        if message.get("role") == "assistant":
            render_assistant_artifacts(message, key_prefix=f"msg_{message_idx}")

query = st.chat_input(
    "Question finance (ex: Compare MSFT et NVDA — risques SEC 2024, perf 6 mois, puis sauvegarde le rapport)"
)
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        text_container = st.empty()
        status_container = st.status("⏳ Analyse en cours...", expanded=True)
        try:
            result = run_stream(
                agent,
                query,
                conversation_id=st.session_state.conversation_id,
                messages=st.session_state.messages,
                text_container=text_container,
                status_container=status_container,
            )

            assistant_text = result.get("clarification_question") or result.get(
                "answer", "Aucune réponse générée."
            )
            # Replace the streamed text with the final answer (no cursor)
            text_container.markdown(assistant_text)

            chunks = result.get("final_chunks", [])
            metadatas = result.get("final_metadatas", [])
            price_context = result.get("price_context", "")
            sources = build_sources(chunks, metadatas)
            debug = {
                "normalized_query": result.get("normalized_query", ""),
                "metadata_filter": result.get("metadata_filter", {}),
                "target_tickers": result.get("target_tickers", []),
            }

            assistant_message = {
                "role": "assistant",
                "content": assistant_text,
                "stats": result.get("stats", {}),
                "sources": sources,
                "price_context": price_context,
                "tool_events": result.get("tool_events", []),
                "report_artifacts": result.get("report_artifacts", []),
                "debug": debug,
            }
            render_assistant_artifacts(
                assistant_message, key_prefix=f"live_{len(st.session_state.messages)}"
            )

            st.session_state.messages.append(assistant_message)
        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("OpenAI / GitHub Models / Gemini (chat) + ChromaDB + LangGraph Agent v2")
