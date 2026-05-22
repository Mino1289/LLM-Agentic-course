# LLM-Agentic-course

Ce projet implémente un système RAG (Retrieval-Augmented Generation) hybride pour analyser les rapports annuels SEC 10-K, ainsi que des agents d'investissement boursier intelligents.

## Étapes et Lancement

### Étape 1 : RAG et Interface Utilisateur (UI)
- Lancer le pré-traitement : `python3 RAG/preprocess.py`
- Lancer l'interface RAG simple :
  `streamlit run ui/app_rag.py`

## Architecture
- `RAG/` : Traitement des données et moteur de recherche hybride (BM25 + ChromaDB Vectoriel).
- `ui/` : Interfaces Streamlit (RAG, MCP, LangGraph).

- `data/` : Espace contenant vos rapports financiers au format `.html`.