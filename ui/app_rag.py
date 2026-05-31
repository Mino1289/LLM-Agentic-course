import os
import sys
import uuid

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid_rag import HybridRAG
from rag.langgraph_flow import FinanceLangGraphAgent

st.set_page_config(page_title="Finance RAG LangGraph", page_icon="📈", layout="wide")
st.title("📈 Finance RAG LangGraph")
st.markdown(
    "Assistant d'analyse financière basé sur 10-K, avec pipeline LangGraph, "
    "mémoire conversationnelle et garbage collector de contexte."
)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())


@st.cache_resource
def build_agent(
    memory_window_size: int,
    summarize_every_n_turns: int,
    max_context_chunks: int,
    max_context_tokens: int,
):
    rag = HybridRAG(chunk_strategy="semantic", search_mode="vector", use_reranking=True)
    rag.load_and_index_data(max_new_embeddings=0)
    return FinanceLangGraphAgent(
        rag=rag,
        memory_window_size=memory_window_size,
        summarize_every_n_turns=summarize_every_n_turns,
        max_context_chunks=max_context_chunks,
        max_context_tokens=max_context_tokens,
    )


st.sidebar.title("Configuration")
st.sidebar.caption("Stratégie unique: semantic chunking + vector retrieval + reranking")
memory_window_default = int(os.getenv("MEMORY_WINDOW_SIZE", "6"))
summarize_every_n_turns_default = int(os.getenv("SUMMARIZE_EVERY_N_TURNS", "6"))
max_context_chunks_default = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
max_context_tokens_default = int(os.getenv("MAX_CONTEXT_TOKENS", "3500"))

memory_window_size = st.sidebar.slider("Fenêtre mémoire", min_value=4, max_value=12, value=memory_window_default)
summarize_every_n_turns = st.sidebar.slider(
    "GC après N tours", min_value=4, max_value=16, value=summarize_every_n_turns_default
)
max_context_chunks = st.sidebar.slider("Chunks max", min_value=4, max_value=12, value=max_context_chunks_default)
max_context_tokens = st.sidebar.slider(
    "Tokens contexte max", min_value=1200, max_value=8000, step=100, value=max_context_tokens_default
)

provider_name = os.getenv("LLM_PROVIDER", "openai")
chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
if provider_name == "github_models":
    chat_model = os.getenv("GITHUB_CHAT_MODEL", "openai/gpt-4o-mini")

st.sidebar.info(f"Provider: {provider_name}\n\nModel: {chat_model}")

try:
    agent = build_agent(
        memory_window_size,
        summarize_every_n_turns,
        max_context_chunks,
        max_context_tokens,
    )
    st.sidebar.success("Système prêt")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation: {e}")
    st.stop()

query = st.text_input(
    "Question finance (ticker + année conseillé, ex: NVDA 2024):",
    placeholder="Quels sont les principaux risques et catalyseurs pour NVDA en 2024 ?",
)

if query:
    with st.spinner("Analyse en cours..."):
        try:
            result = agent.run(query, conversation_id=st.session_state.conversation_id)
            st.subheader("Réponse")
            st.markdown(result.get("answer", "Aucune réponse générée."))

            stats = result.get("stats", {})
            st.caption(
                f"Chunks utilisés: {stats.get('chunks_used', 0)} | "
                f"Tokens contexte estimés: {stats.get('estimated_context_tokens', 0)} | "
                f"GC appliqué: {'oui' if stats.get('gc_applied') else 'non'}"
            )

            with st.expander("Sources consultées"):
                chunks = result.get("final_chunks", [])
                metadatas = result.get("final_metadatas", [])
                for i, chunk in enumerate(chunks):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    source = meta.get("source", "unknown")
                    section = meta.get("section", "unknown")
                    st.markdown(f"**Source {i + 1}** - `{source}` / `{section}`")
                    st.text(chunk)
                    st.divider()
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")

st.divider()
st.caption("OpenAI/GitHub Models + ChromaDB + LangGraph")
