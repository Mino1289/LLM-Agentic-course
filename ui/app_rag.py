import asyncio
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.tracing import ensure_langsmith_env
ensure_langsmith_env()

from src.rag.core import HybridRAG
from src.graph.flow import FinanceLangGraphAgent
from src.orchestration.hub_graph import HubAndSpokeGraph
from src.llm import build_llm_config_from_env
from src.tools.definitions import get_tool_definitions
from ui.streaming import run_phase3_stream

st.set_page_config(page_title="Finance RAG Hub-and-Spoke", page_icon="📈", layout="wide")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je peux analyser NVDA, AMD, MSFT, ARM, ASML — "
                "interroger les filings SEC, les transcripts earnings, valider des affirmations, "
                "simuler une allocation, récupérer les prix et générer des rapports."
            ),
        }
    ]
if "pending_trade_state" not in st.session_state:
    st.session_state.pending_trade_state = None


@st.cache_resource
def build_agent(
    max_context_chunks: int,
    decompose_query_count: int,
    price_max_days: int,
    price_max_points: int,
    price_max_tickers: int,
    price_default_days: int,
    max_tool_iterations: int,
):
    rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
    rag.load_and_index_data(max_new_embeddings=0)
    return FinanceLangGraphAgent(
        rag=rag,
        max_context_chunks=max_context_chunks,
        decompose_query_count=decompose_query_count,
        price_max_days=price_max_days,
        price_max_points=price_max_points,
        price_max_tickers=price_max_tickers,
        price_default_days=price_default_days,
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
        if path.suffix == ".md":
            mime = "text/markdown"
        elif path.suffix == ".pdf":
            mime = "application/pdf"
        else:
            mime = "application/octet-stream"
        st.download_button(
            label=f"Télécharger {artifact.get('filename', path.name)}",
            data=data,
            file_name=artifact.get("filename", path.name),
            mime=mime,
            key=f"{key_prefix}_report_{idx}",
        )


def render_stats(stats: dict[str, Any]) -> None:
    if not stats:
        return
    items = []
    spoke_tc = stats.get("spoke_tool_calls", 0)
    spoke_llm = stats.get("spoke_llm_iterations", 0)
    if spoke_tc:
        items.append(("Appels outils", str(spoke_tc)))
    if spoke_llm:
        items.append(("Itérations LLM", str(spoke_llm)))
    total_tokens = stats.get("llm_total_tokens", 0) + stats.get("guard_total_tokens", 0) or stats.get("estimated_context_tokens", 0)
    if total_tokens:
        items.append(("Tokens", str(total_tokens)))
    if stats.get("chunks_used"):
        items.append(("Chunks", str(stats["chunks_used"])))
    if stats.get("retrieval_candidate_count"):
        items.append(("Candidats", str(stats["retrieval_candidate_count"])))
    if stats.get("rerank_final_count", stats.get("chunks_used")):
        items.append(("Final", str(stats.get("rerank_final_count") or stats.get("chunks_used", ""))))
    if stats.get("price_tool_used"):
        items.append(("Prix", "oui"))
    if stats.get("rag_tool_used"):
        items.append(("RAG", "oui"))
    if stats.get("gc_applied"):
        items.append(("GC", "oui"))
    for stat_name in ("pm_plan_done", "pm_synthesis_done"):
        if stats.get(stat_name):
            items.append((stat_name.replace("_", " ").title(), "✓"))
    if not items:
        return
    cols = st.columns(min(len(items), 6))
    for i, (label, value) in enumerate(items[:6]):
        cols[i].metric(label, value)


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
                "alpaca_tool_used": stats.get("alpaca_tool_used"),
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


def render_human_review(state: dict[str, Any]) -> None:
    decision = state.get("pm_decision", {})
    detail = state.get("compliance_detail", "")

    st.markdown("---")
    st.subheader("🔐 Approbation Humaine Requise")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Ticker:** {decision.get('ticker', 'N/A')}")
        st.markdown(f"**Side:** {decision.get('side', 'N/A')}")
        st.markdown(f"**Quantité:** {decision.get('qty', 'N/A')}")
        st.markdown(f"**Type d'ordre:** {decision.get('order_type', 'market')}")
        if decision.get("limit_price"):
            st.markdown(f"**Prix limite:** {decision['limit_price']}")
    with col2:
        st.metric("Buying Power", "À vérifier")
        st.metric("Risque", "Faible" if state.get("compliance_verdict") == "PASS" else "Élevé")

    with st.expander("Justification complète"):
        st.markdown(decision.get("response", "N/A"))
    with st.expander("Détail de la vérification Compliance"):
        st.markdown(detail)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Approuver le Trade", type="primary", use_container_width=True):
            st.session_state.pending_trade_approved = True
            st.rerun()
    with col_b:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.pending_trade_approved = False
            st.rerun()


st.sidebar.title("Finance RAG Hub-and-Spoke")
st.sidebar.caption("Routeur d'intention → agents spécialisés Hub-and-Spoke")

try:
    llm_config = build_llm_config_from_env()
    st.sidebar.subheader("Modèles")
    st.sidebar.markdown(f"**Chat** : `{llm_config.provider}` · `{llm_config.chat_model}`")
    st.sidebar.markdown(f"**Embeddings** : `{llm_config.embedding_provider}` · `{llm_config.embedding_model}`")
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}:
        region = os.getenv("LANGSMITH_REGION", "us").upper()
        project = os.getenv("LANGSMITH_PROJECT", "")
        st.sidebar.caption(f"LangSmith ({region}) · projet `{project}`")
except Exception as config_exc:
    st.sidebar.error(f"Config LLM : {config_exc}")

st.sidebar.subheader("Outils disponibles")
st.sidebar.caption("10 outils (contrat type MCP). Les agents les appellent dynamiquement.")
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
                st.markdown(f"- `{param_name}`{req} — {param_info.get('description', '')}")

st.sidebar.divider()
st.sidebar.subheader("Configuration")
max_context_chunks_default = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
decompose_query_count_default = int(os.getenv("QUERY_DECOMPOSE_COUNT", "2"))
price_max_days_default = int(os.getenv("PRICE_MAX_DAYS", "180"))
price_max_points_default = int(os.getenv("PRICE_MAX_POINTS", "40"))
price_max_tickers_default = int(os.getenv("PRICE_MAX_TICKERS", "3"))
price_default_days_default = int(os.getenv("PRICE_DEFAULT_DAYS", "90"))
max_tool_iterations_default = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))

max_context_chunks = st.sidebar.slider("Chunks max", min_value=4, max_value=12, value=max_context_chunks_default)
decompose_query_count = st.sidebar.slider("Nb sous-requêtes", min_value=1, max_value=8, value=decompose_query_count_default)
price_max_days = st.sidebar.slider("Prix max jours", min_value=30, max_value=365, value=price_max_days_default)
price_max_points = st.sidebar.slider("Prix max points", min_value=10, max_value=120, value=price_max_points_default)
price_max_tickers = st.sidebar.slider("Prix max tickers", min_value=1, max_value=5, value=price_max_tickers_default)
price_default_days = st.sidebar.slider("Prix fenêtre défaut (jours)", min_value=15, max_value=180, value=price_default_days_default)
max_tool_iterations = st.sidebar.slider("Max itérations agent/outils", min_value=2, max_value=10, value=max_tool_iterations_default)

if st.sidebar.button("Nouvelle conversation", use_container_width=True):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je peux analyser NVDA, AMD, ARM, ASML et MSFT, "
                "interroger les filings SEC et earnings calls, valider des affirmations, "
                "simuler une allocation fictive et exporter un rapport."
            ),
        }
    ]
    st.session_state.pending_trade_state = None
    st.rerun()

st.sidebar.divider()

st.title("📈 Orchestration Hub-and-Spoke")
st.markdown(
    "Système multi-agents : **Routeur d'Intention** 🚦 décide automatiquement : "
    "requête simple → **Agent Phase 2** ; requête complexe → **Portfolio Manager** 👔 → "
    "**Analystes** 📚📈 → **Compliance** 🛡️ → **Executor** ⚡."
)

try:
    agent = build_agent(
        max_context_chunks,
        decompose_query_count,
        price_max_days,
        price_max_points,
        price_max_tickers,
        price_default_days,
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
        hub_graph = HubAndSpokeGraph(agent, max_spoke_iterations=3)

        console_container = st.empty()
        text_container = st.empty()
        status_container = st.status("⏳ Analyse en cours...", expanded=True)

        def on_human_review_cb(state: dict[str, Any]) -> None:
            st.session_state.pending_trade_state = state
            st.session_state.pending_trade_approved = None

        try:
            result = run_phase3_stream(
                hub_graph,
                query,
                conversation_id=st.session_state.conversation_id,
                messages=st.session_state.messages,
                console_container=console_container,
                text_container=text_container,
                status_container=status_container,
                on_human_review=on_human_review_cb,
            )

            if result.get("human_review_pending"):
                st.session_state.pending_trade_state = result

            assistant_text = result.get("answer", "Aucune réponse générée.")
            text_container.markdown(assistant_text)

            spoke_events = result.get("spoke_events", [])
            tool_events = result.get("tool_events", [])
            stats = result.get("stats", {})

            message = {
                "role": "assistant",
                "content": assistant_text,
                "stats": stats,
                "sources": build_sources([], []),
                "tool_events": tool_events if tool_events else [],
                "report_artifacts": [],
            }
            render_assistant_artifacts(message, key_prefix=f"live_{len(st.session_state.messages)}")
            st.session_state.messages.append(message)

        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Traitement approve/reject : doit être HORS du bloc if query: pour survivre aux reruns
if st.session_state.get("pending_trade_state") and st.session_state.get("pending_trade_approved") is None:
    render_human_review(st.session_state.pending_trade_state)
elif st.session_state.get("pending_trade_approved") is True:
    st.info("✅ Trade approuvé. Exécution en cours...")
    hub_graph = HubAndSpokeGraph(agent, max_spoke_iterations=3)
    final_state = asyncio.run(hub_graph.resume_after_human(
        st.session_state.pending_trade_state, approved=True,
    ))
    st.success(final_state.get("answer", "Trade exécuté."))
    st.session_state.pending_trade_state = None
    st.session_state.pending_trade_approved = None
elif st.session_state.get("pending_trade_approved") is False:
    st.warning("❌ Ordre annulé par l'utilisateur.")
    st.session_state.pending_trade_state = None
    st.session_state.pending_trade_approved = None

st.divider()
st.caption("OpenAI/GitHub Models/Gemini + ChromaDB + LangGraph · Orchestration Hub-and-Spoke multi-agents")
