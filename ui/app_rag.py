import streamlit as st
import sys
import os

# Add the root directory to sys.path to allow importing from RAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid_rag import HybridRAG

st.set_page_config(page_title="Hybrid RAG Finan cier", page_icon="📈", layout="wide")

st.title("📈 Hybrid RAG : Analyse de Rapports 10-K")
st.markdown("""
Cette interface vous permet d'interroger les rapports financiers (10-K) de plusieurs entreprises technologiques.
Le système utilise une recherche hybride (BM25 + Vecteurs) pour rester sous le quota de 1000 embeddings par jour.
""")


@st.cache_resource
def load_rag():
    rag = HybridRAG()
    # On charge et on indexe au démarrage (si pas déjà fait)
    with st.spinner(
        "Chargement de l'index BM25 et vérification de la base vectorielle..."
    ):
        rag.load_and_index_data(
            max_new_embeddings=0
        )  # On ne fait pas de nouveaux embeddings ici par sécurité
    return rag


try:
    rag_system = load_rag()
    st.sidebar.success("Système RAG prêt !")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation : {e}")
    st.stop()

# Sidebar info
st.sidebar.title("Informations")
st.sidebar.info(f"Nombre de morceaux indexés : {len(rag_system.documents)}")

# Question input
query = st.text_input(
    "Posez votre question sur les finances d'une entreprise (ex: NVIDIA, Google, Apple...)",
    placeholder="Quels sont les risques majeurs mentionnés par Apple en 2024 ?",
)

if query:
    with st.spinner("Recherche et analyse en cours..."):
        # On pourrait ajouter un bouton mais l'input suffit pour streamlit
        try:
            # Récupération des chunks pour affichage (optionnel)
            retrieved_chunks = rag_system.retrieve(query)

            # Génération de la réponse
            answer = rag_system.answer_question(query)

            st.subheader("Réponse")
            st.markdown(answer)

            with st.expander("Voir les sources consultées (Chunks)"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i + 1}**")
                    st.text(chunk)
                    st.divider()

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération : {e}")

st.divider()
st.caption("Développé avec Gemini 2.0 Flash & ChromaDB")
