import os
import json
import glob
import time
from rank_bm25 import BM25Okapi
import chromadb
from google import genai
from google.genai import types

from dotenv import load_dotenv

# Charge les variables d'environnement du fichier .env
load_dotenv("../.env")

# Ensure your GEMINI_API_KEY is set in environment variables
# export GEMINI_API_KEY="your-api-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class HybridRAG:
    def __init__(self, collection_name="10k_reports"):
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []

        # Initialize Google GenAI client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def load_and_index_data(self, max_files=None, max_new_embeddings=1000):
        print("Loading and indexing data...")
        files = glob.glob(os.path.join(DATA_DIR, "*.json"))

        if max_files:
            files = files[:max_files]

        all_chunks = []
        all_metadata = []
        all_ids = []

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            filename = os.path.basename(file_path).replace(".json", "")

            for section_name, text in data.items():
                if not text:
                    continue
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{section_name}_{i}"
                    all_chunks.append(chunk)
                    all_metadata.append(
                        {"source": filename, "section": section_name, "chunk_index": i}
                    )
                    all_ids.append(chunk_id)

        self.documents = all_chunks
        self.doc_metadata = all_metadata

        print(f"Total chunks found in files: {len(self.documents)}")

        # 1. Index with BM25 (Always everything for local search)
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Indexing complete.")

        # 2. Vectorize ONLY what's missing in ChromaDB
        print("Checking for missing embeddings in ChromaDB...")

        # Get all existing IDs in the collection
        existing_result = self.collection.get()
        existing_ids = set(existing_result["ids"])

        missing_indices = [
            i for i, cid in enumerate(all_ids) if cid not in existing_ids
        ]

        if not missing_indices:
            print("All chunks are already indexed in ChromaDB.")
            return

        print(f"{len(missing_indices)} chunks are missing from ChromaDB.")

        # Limit to max_new_embeddings for today's quota
        indices_to_embed = missing_indices[:max_new_embeddings]
        print(
            f"Embedding {len(indices_to_embed)} new chunks (Quota limit: {max_new_embeddings})..."
        )

        for idx in indices_to_embed:
            chunk = self.documents[idx]
            metadata = self.doc_metadata[idx]
            chunk_id = all_ids[idx]

            try:
                # Use Gemini to generate individual embedding
                response = self.client.models.embed_content(
                    model="gemini-embedding-2", contents=chunk
                )
                embedding = response.embeddings[0].values

                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[chunk_id],
                )
                # Respecter le quota RPM (Requests Per Minute)
                time.sleep(0.6)
            except Exception as e:
                print(f"Error embedding chunk {chunk_id}: {e}")
                break  # Stop if we hit a quota error or other issue

        print(f"Vector Indexing complete. Total in DB: {self.collection.count()}")

    def _bm25_search(self, query, top_k=30):
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:top_k]
        return [self.documents[i] for i in top_indices]

    def _vector_search(self, query, top_k=5):
        try:
            query_embedding = (
                self.client.models.embed_content(
                    model="gemini-embedding-2", contents=query
                )
                .embeddings[0]
                .values
            )

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )
            if results["documents"]:
                return results["documents"][0]
            return []
        except Exception as e:
            print(f"Vector search warning/error: {e}")
            return []

    def retrieve(self, query):
        bm25_results = self._bm25_search(query, top_k=30)
        vector_results = self._vector_search(query, top_k=5)

        # Deduplicate
        all_results = list(set(bm25_results + vector_results))
        return all_results

    def answer_question(self, query):
        retrieved_chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(retrieved_chunks)

        prompt = f"""En te basant uniquement sur les extraits du rapport 10-K suivants, quels sont les signaux positifs ou négatifs concernant la trajectoire financière de l'entreprise ?
        
        Extraits:
        {context}
        
        Question: {query}
        """

        response = self.client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
        )

        return response.text


if __name__ == "__main__":
    rag = HybridRAG()
    # On indexe tous les fichiers, avec une limite de 800 nouveaux embeddings aujourd'hui
    rag.load_and_index_data(max_files=None, max_new_embeddings=800)

    query = "Quels sont les facteurs de risques principaux ou changements financiers mentionnés récemment chez NVIDIA ?"
    print(f"\nQuestion: {query}")
    answer = rag.answer_question(query)
    print(f"\nRéponse:\n{answer}")
