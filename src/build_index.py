import argparse
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chunking import CHUNKING_METHODS, validate_chunking_args
from embedding_models import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    collection_name_for_embedding_model,
    resolve_embedding_model,
)
from ingest import build_documents


CHROMA_DIR = "chroma_db"
ADD_BATCH_SIZE = 512


def collection_exists(client, name: str):
    collections = client.list_collections()
    collection_names = [
        collection.name if hasattr(collection, "name") else collection
        for collection in collections
    ]

    return name in collection_names


def recreate_collection(client, name: str, metadata: dict):
    if collection_exists(client, name):
        print(f"[INFO] Deleting existing collection: {name}")
        client.delete_collection(name)

    return client.create_collection(name=name, metadata=metadata)


def get_collection(client, name: str, rebuild: bool, metadata: dict):
    if rebuild:
        return recreate_collection(client, name, metadata)

    return client.get_or_create_collection(name=name, metadata=metadata)


def make_chunk_id(doc: dict):
    metadata = doc["metadata"]
    source_file = metadata.get("source_file", "unknown_source")
    chunking_method = metadata.get("chunking_method", "unknown_method")
    chunk_size = metadata.get("chunk_size", "unknown_size")
    chunk_overlap = metadata.get("chunk_overlap", "unknown_overlap")
    content_type = metadata.get("content_type", "unknown_content")
    table_id = metadata.get("table_id", "not_table")
    page = metadata.get("page", "unknown_page")
    chunk_id = metadata.get("chunk_id", "unknown_chunk")
    raw_id = (
        f"{source_file}:{chunking_method}:{chunk_size}:"
        f"{chunk_overlap}:{content_type}:{table_id}:{page}:{chunk_id}"
    )
    stable_hash = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]

    return f"{source_file}:{content_type}:{chunking_method}:{page}:{chunk_id}:{stable_hash}"


def add_to_collection_in_batches(
    collection,
    ids,
    texts,
    metadatas,
    embeddings,
    batch_size: int = ADD_BATCH_SIZE,
):
    for start in tqdm(range(0, len(ids), batch_size), desc="Adding batches"):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build or update the ChromaDB vector index."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before indexing.",
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_METHODS,
        default="simple",
        help="Chunking strategy to use before embedding.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Extract and index tables from PDF files.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=EMBEDDING_MODELS.keys(),
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use for the vector index.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional Chroma collection name. Defaults to one collection per embedding model.",
    )
    args = parser.parse_args()
    try:
        validate_chunking_args(args.chunk_size, args.overlap)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main():
    args = parse_args()
    embedding_model_name = resolve_embedding_model(args.embedding_model)
    collection_name = collection_name_for_embedding_model(
        args.embedding_model,
        args.collection_name,
    )
    collection_metadata = {
        "embedding_model_key": args.embedding_model,
        "embedding_model_name": embedding_model_name,
        "chunking_method": args.chunking,
        "chunk_size": str(args.chunk_size),
        "chunk_overlap": str(args.overlap),
    }

    print(f"[INFO] Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    table_status = "with PDF tables" if args.include_tables else "without PDF tables"
    print(f"[INFO] Building documents with {args.chunking} chunking ({table_status})...")
    docs = build_documents(
        chunk_method=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        include_tables=args.include_tables,
    )
    print(f"[INFO] Number of chunks: {len(docs)}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(
        client,
        collection_name,
        rebuild=args.rebuild,
        metadata=collection_metadata,
    )
    print(f"[INFO] Using Chroma collection: {collection_name}")

    ids = []
    texts = []
    metadatas = []

    for doc in docs:
        ids.append(make_chunk_id(doc))
        texts.append(doc["text"])
        metadatas.append({
            **doc["metadata"],
            "embedding_model": args.embedding_model,
            "embedding_model_name": embedding_model_name,
        })

    print("[INFO] Computing embeddings...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("[INFO] Adding to ChromaDB...")
    add_to_collection_in_batches(
        collection,
        ids,
        texts,
        metadatas,
        embeddings,
    )

    print("[DONE] Vector database created.")


if __name__ == "__main__":
    main()
