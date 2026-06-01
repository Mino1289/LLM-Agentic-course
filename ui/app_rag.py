import os
import sys
import uuid

import streamlit as st

# Reduce noisy transformers warnings triggered by Streamlit module introspection.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    # transformers is an optional transitive dependency in this UI path.
    pass

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
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour, je peux analyser les filings financiers comme un chatbot conversationnel. "
                "Pose ta question librement (même sans ticker/année)."
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
    )


st.sidebar.title("Configuration")
st.sidebar.caption("Stratégie unique: semantic chunking + vector retrieval + reranking")
memory_window_default = int(os.getenv("MEMORY_WINDOW_SIZE", "6"))
summarize_every_n_turns_default = int(os.getenv("SUMMARIZE_EVERY_N_TURNS", "6"))
max_context_chunks_default = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
max_context_tokens_default = int(os.getenv("MAX_CONTEXT_TOKENS", "3500"))
decompose_query_count_default = int(os.getenv("QUERY_DECOMPOSE_COUNT", "4"))

memory_window_size = st.sidebar.slider("Fenêtre mémoire", min_value=4, max_value=12, value=memory_window_default)
summarize_every_n_turns = st.sidebar.slider(
    "GC après N tours", min_value=4, max_value=16, value=summarize_every_n_turns_default
)
max_context_chunks = st.sidebar.slider("Chunks max", min_value=4, max_value=12, value=max_context_chunks_default)
max_context_tokens = st.sidebar.slider(
    "Tokens contexte max", min_value=1200, max_value=8000, step=100, value=max_context_tokens_default
)
decompose_query_count = st.sidebar.slider(
    "Nb sous-requêtes", min_value=3, max_value=8, value=decompose_query_count_default
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
        decompose_query_count,
    )
    st.sidebar.success("Système prêt")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation: {e}")
    st.stop()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(message.get("content", ""))
        if message.get("role") == "assistant":
            stats = message.get("stats", {})
            if stats:
                st.caption(
                    f"Chunks utilisés: {stats.get('chunks_used', 0)} | "
                    f"Tokens contexte estimés: {stats.get('estimated_context_tokens', 0)} | "
                    f"GC appliqué: {'oui' if stats.get('gc_applied') else 'non'} | "
                    f"Sous-requêtes: {stats.get('decomposed_query_count', 0)}"
                )
            sources = message.get("sources", [])
            if sources:
                with st.expander("Sources consultées"):
                    for source_idx, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**Source {source_idx}** - "
                            f"`{source.get('source', 'unknown')}` / "
                            f"`{source.get('section', 'unknown')}`"
                        )
                        st.text(source.get("chunk", ""))
                        st.divider()

query = st.chat_input("Question finance (ex: Quels risques majeurs sur les semiconducteurs en 2024 ?)")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                result = agent.run(
                    query,
                    conversation_id=st.session_state.conversation_id,
                    messages=st.session_state.messages,
                )

                assistant_text = result.get("clarification_question") or result.get(
                    "answer", "Aucune réponse générée."
                )
                st.markdown(assistant_text)

                stats = result.get("stats", {})
                st.caption(
                    f"Chunks utilisés: {stats.get('chunks_used', 0)} | "
                    f"Tokens contexte estimés: {stats.get('estimated_context_tokens', 0)} | "
                    f"GC appliqué: {'oui' if stats.get('gc_applied') else 'non'} | "
                    f"Sous-requêtes: {stats.get('decomposed_query_count', 0)}"
                )

                chunks = result.get("final_chunks", [])
                metadatas = result.get("final_metadatas", [])
                sources = []
                for i, chunk in enumerate(chunks):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    sources.append(
                        {
                            "source": meta.get("source", "unknown"),
                            "section": meta.get("section", "unknown"),
                            "chunk": chunk,
                        }
                    )

                if sources:
                    with st.expander("Sources consultées"):
                        for source_idx, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**Source {source_idx}** - "
                                f"`{source.get('source', 'unknown')}` / "
                                f"`{source.get('section', 'unknown')}`"
                            )
                            st.text(source.get("chunk", ""))
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "stats": stats,
                        "sources": sources,
                    }
                )
            except Exception as e:
                error_msg = f"Erreur lors de l'analyse: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("OpenAI/GitHub Models + ChromaDB + LangGraph")
