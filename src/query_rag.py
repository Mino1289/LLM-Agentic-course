import argparse
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from env_config import load_env_file
from llm_provider import LLMError, PROVIDERS as LLM_PROVIDERS, query_llm
from embedding_models import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    collection_name_for_embedding_model,
    resolve_embedding_model,
)
from query_planner import QueryPlanningError, plan_queries_with_openai
from retrieval import (
    BM25Index,
    SEARCH_MODES,
    load_corpus,
    reciprocal_rank_fusion,
    rerank_results,
    vector_search,
)


CHROMA_DIR = "chroma_db"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_QUESTION = (
    "What did NVIDIA management say about demand for Blackwell"
    "during the Q4 2025 earnings call?"
)
DEFAULT_QUERY_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the semiconductor RAG index."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to ask the RAG retriever.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=EMBEDDING_MODELS.keys(),
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used to build the target vector index.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional Chroma collection name. Defaults to one collection per embedding model.",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default="vector",
        help="Retrieval strategy before optional reranking.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=30,
        help="Number of candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of final results to print.",
    )
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=True,
        help="Rerank retrieved candidates with a cross-encoder.",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable cross-encoder reranking.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model used when reranking is enabled.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant for hybrid search.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare vector, BM25, hybrid, and hybrid+reranking for the same question.",
    )
    parser.add_argument(
        "--comparison-output",
        default="comparison.txt",
        help="Output file written when --compare is enabled.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Use an LLM to rewrite the question into focused retrieval queries.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="Maximum number of focused retrieval queries generated with --decompose.",
    )
    parser.add_argument(
        "--query-planner-model",
        default=None,
        help="OpenAI model used for query decomposition. Defaults to .env.",
    )
    parser.add_argument(
        "--query-planner-temperature",
        type=float,
        default=None,
        help="Temperature used for query decomposition. Defaults to .env.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate an LLM-synthesized answer from retrieved passages.",
    )
    parser.add_argument(
        "--answer-provider",
        choices=LLM_PROVIDERS,
        default="gemini",
        help="LLM provider for answer generation.",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="LLM model for answer generation (provider-specific default if omitted).",
    )
    args = parser.parse_args()

    if args.n_results <= 0:
        parser.error("--n-results must be greater than 0")

    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")

    if args.query_count <= 0:
        parser.error("--query-count must be greater than 0")

    return args


def retrieve(
    question: str,
    collection,
    search_mode: str,
    n_results: int,
    embedding_model=None,
    bm25_index=None,
    rrf_k: int = 60,
):
    if search_mode == "vector":
        return vector_search(
            collection=collection,
            question=question,
            embedding_model=embedding_model,
            n_results=n_results,
        )

    if search_mode == "bm25":
        return bm25_index.search(question, n_results=n_results)

    if search_mode == "hybrid":
        vector_results = vector_search(
            collection=collection,
            question=question,
            embedding_model=embedding_model,
            n_results=n_results,
        )
        bm25_results = bm25_index.search(question, n_results=n_results)

        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            n_results=n_results,
            rrf_k=rrf_k,
        )

    raise ValueError(f"Unsupported search mode: {search_mode}")


def build_query_plan(args):
    if not args.decompose:
        return [{
            "query": args.question,
            "purpose": "Original user question",
        }]

    print("[INFO] Planning focused retrieval queries with LLM...", flush=True)

    try:
        query_plan = plan_queries_with_openai(
            question=args.question,
            max_queries=args.query_count,
            model=args.query_planner_model,
            temperature=args.query_planner_temperature,
        )
    except QueryPlanningError as exc:
        raise SystemExit(f"[ERROR] Query decomposition failed: {exc}") from None

    print("[INFO] Focused retrieval queries:", flush=True)
    for index, item in enumerate(query_plan, start=1):
        purpose = f" ({item['purpose']})" if item.get("purpose") else ""
        print(f"  {index}. {item['query']}{purpose}", flush=True)

    return query_plan


def annotate_results_with_query(results: list[dict], query_item: dict, query_index: int):
    annotated = []

    for result in results:
        annotated.append({
            **result,
            "retrieval_query": query_item["query"],
            "retrieval_query_purpose": query_item.get("purpose", ""),
            "retrieval_query_index": query_index,
            "matched_queries": [query_item["query"]],
        })

    return annotated


def retrieve_with_query_plan(
    query_plan: list[dict],
    collection,
    search_mode: str,
    n_results: int,
    embedding_model=None,
    bm25_index=None,
    rrf_k: int = 60,
):
    if len(query_plan) == 1:
        return retrieve(
            question=query_plan[0]["query"],
            collection=collection,
            search_mode=search_mode,
            n_results=n_results,
            embedding_model=embedding_model,
            bm25_index=bm25_index,
            rrf_k=rrf_k,
        )

    result_lists = []

    for query_index, query_item in enumerate(query_plan, start=1):
        results = retrieve(
            question=query_item["query"],
            collection=collection,
            search_mode=search_mode,
            n_results=n_results,
            embedding_model=embedding_model,
            bm25_index=bm25_index,
            rrf_k=rrf_k,
        )
        result_lists.append(
            annotate_results_with_query(results, query_item, query_index)
        )

    return reciprocal_rank_fusion(
        result_lists,
        n_results=n_results,
        rrf_k=rrf_k,
        score_key="multi_query_score",
    )


def score_summary(result: dict):
    parts = []

    if result.get("vector_distance") is not None:
        parts.append(f"vector_distance={result['vector_distance']:.4f}")

    if result.get("bm25_score") is not None:
        parts.append(f"bm25_score={result['bm25_score']:.4f}")

    if result.get("hybrid_score") is not None:
        parts.append(f"hybrid_score={result['hybrid_score']:.4f}")

    if result.get("multi_query_score") is not None:
        parts.append(f"multi_query_score={result['multi_query_score']:.4f}")

    if result.get("rerank_score") is not None:
        parts.append(f"rerank_score={result['rerank_score']:.4f}")

    return ", ".join(parts)


def print_compact_results(title: str, results: list[dict], top_k: int):
    print("=" * 80)
    print(title)

    for rank, result in enumerate(results[:top_k], start=1):
        metadata = result["metadata"]
        print(
            f"{rank}. {metadata.get('company')} | "
            f"{metadata.get('source_file')} | "
            f"page={metadata.get('page')} | "
            f"type={metadata.get('content_type')} | "
            f"{score_summary(result)}"
        )


def print_detailed_results(results: list[dict], top_k: int):
    for rank, result in enumerate(results[:top_k], start=1):
        metadata = result["metadata"]
        print("=" * 80)
        print(f"Result {rank}")
        print(f"Company: {metadata.get('company')}")
        print(f"Ticker: {metadata.get('ticker')}")
        print(f"File: {metadata.get('source_file')}")
        print(f"Source type: {metadata.get('source_type')}")
        print(f"Content type: {metadata.get('content_type')}")

        if metadata.get("table_id") is not None:
            print(f"Table ID: {metadata.get('table_id')}")

        print(f"Page: {metadata.get('page')}")
        print(f"Chunking: {metadata.get('chunking_method')}")
        print(f"Embedding model: {metadata.get('embedding_model')}")

        if result.get("vector_distance") is not None:
            print(f"Vector distance: {result['vector_distance']:.4f}")

        if result.get("bm25_score") is not None:
            print(f"BM25 score: {result['bm25_score']:.4f}")

        if result.get("hybrid_score") is not None:
            print(f"Hybrid score: {result['hybrid_score']:.4f}")

        if result.get("rerank_score") is not None:
            print(f"Reranker score: {result['rerank_score']:.4f}")

        if result.get("matched_queries"):
            print("Matched retrieval queries:")
            for query in result["matched_queries"]:
                print(f"- {query}")

        print("-" * 80)
        print(result["document"][:1200])


def metadata_summary(metadata: dict):
    fields = [
        ("Company", metadata.get("company")),
        ("Ticker", metadata.get("ticker")),
        ("File", metadata.get("source_file")),
        ("Source type", metadata.get("source_type")),
        ("Content type", metadata.get("content_type")),
        ("Table ID", metadata.get("table_id")),
        ("Page", metadata.get("page")),
        ("Chunking", metadata.get("chunking_method")),
        ("Embedding model", metadata.get("embedding_model")),
    ]

    return [
        f"{label}: {value}"
        for label, value in fields
        if value is not None
    ]


def retrieval_query_summary(result: dict):
    queries = result.get("matched_queries") or []

    if not queries and result.get("retrieval_query"):
        queries = [result["retrieval_query"]]

    return [
        f"Matched query {index}: {query}"
        for index, query in enumerate(queries, start=1)
    ]


def write_comparison_file(
    output_path: str,
    sections: list[tuple[str, list[dict]]],
    args,
    collection_name: str,
    embedding_model_name: str,
    query_plan: list[dict],
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "RAG retrieval comparison",
        "=" * 80,
        f"Question: {args.question}",
        f"Collection: {collection_name}",
        f"Embedding model key: {args.embedding_model}",
        f"Embedding model name: {embedding_model_name}",
        f"Candidates per strategy: {args.n_results}",
        f"Top passages written per strategy: {args.top_k}",
        f"RRF k: {args.rrf_k}",
        f"Query decomposition: {'enabled' if args.decompose else 'disabled'}",
        "",
    ]

    if args.decompose:
        lines.extend([
            "Focused retrieval queries",
            "-" * 80,
        ])

        for index, item in enumerate(query_plan, start=1):
            purpose = f" | purpose: {item['purpose']}" if item.get("purpose") else ""
            lines.append(f"{index}. {item['query']}{purpose}")

        lines.append("")

    for title, results in sections:
        lines.extend([
            "=" * 80,
            title,
            "=" * 80,
            "",
        ])

        for rank, result in enumerate(results[:args.top_k], start=1):
            lines.extend([
                "-" * 80,
                f"Result {rank}",
                "-" * 80,
                *metadata_summary(result["metadata"]),
                *retrieval_query_summary(result),
                f"Scores: {score_summary(result)}",
                "",
                "Passage:",
                result["document"],
                "",
            ])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote comparison passages to {path}")


def build_bm25_index(collection):
    print("[INFO] Loading corpus for BM25...")
    corpus = load_corpus(collection)
    print(f"[INFO] Loaded {len(corpus)} chunks for BM25")

    return BM25Index(corpus)


def load_embedding_model(model_name: str):
    print(f"[INFO] Loading embedding model: {model_name}")

    return SentenceTransformer(model_name)


def load_reranker(model_name: str):
    print(f"[INFO] Loading reranker: {model_name}")

    return CrossEncoder(model_name)


ANSWER_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question based ONLY on "
    "the provided passages from semiconductor company documents (annual reports, "
    "earnings calls, financial data). "
    "Cite sources inline using [Source: filename, page X]. "
    "If the passages do not contain enough information to answer, say so clearly. "
    "Do not invent facts or use external knowledge."
)


def synthesize_answer(question: str, results: list[dict], provider: str, model: str | None):
    context_parts = []
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        header = f"[{rank}] Source: {meta.get('source_file', 'unknown')}"
        if meta.get("page"):
            header += f", page {meta['page']}"
        if meta.get("company"):
            header += f" | {meta['company']} ({meta.get('ticker', '')})"
        context_parts.append(f"{header}\n{result['document']}")

    context = "\n\n".join(context_parts)
    prompt = (
        f"Question: {question}\n\n"
        f"Passages:\n{context}\n\n"
        "Provide a concise answer citing the relevant sources."
    )

    print("\n" + "=" * 80)
    print(f"[ANSWER] Generating answer with {provider}...")
    print("=" * 80)

    try:
        answer = query_llm(
            prompt=prompt,
            provider=provider,
            model=model,
            system_prompt=ANSWER_SYSTEM_PROMPT,
        )
    except LLMError as exc:
        print(f"[ERROR] LLM answer failed: {exc}")
        return

    print("\n" + answer + "\n")


def run_single_mode(args, collection, embedding_model_name: str):
    needs_vector = args.search_mode in {"vector", "hybrid"}
    needs_bm25 = args.search_mode in {"bm25", "hybrid"}
    query_plan = build_query_plan(args)

    embedding_model = load_embedding_model(embedding_model_name) if needs_vector else None
    bm25_index = build_bm25_index(collection) if needs_bm25 else None

    results = retrieve_with_query_plan(
        query_plan=query_plan,
        collection=collection,
        search_mode=args.search_mode,
        n_results=args.n_results,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        rrf_k=args.rrf_k,
    )

    if args.rerank:
        reranker = load_reranker(args.reranker_model)
        results = rerank_results(args.question, results, reranker)

    print_detailed_results(results, top_k=args.top_k)

    if args.answer:
        synthesize_answer(
            question=args.question,
            results=results[:args.top_k],
            provider=args.answer_provider,
            model=args.answer_model,
        )


def run_comparison(args, collection, embedding_model_name: str, collection_name: str):
    query_plan = build_query_plan(args)
    embedding_model = load_embedding_model(embedding_model_name)
    bm25_index = build_bm25_index(collection)

    vector_results = retrieve_with_query_plan(
        query_plan=query_plan,
        collection=collection,
        search_mode="vector",
        n_results=args.n_results,
        embedding_model=embedding_model,
    )
    bm25_results = retrieve_with_query_plan(
        query_plan=query_plan,
        collection=collection,
        search_mode="bm25",
        n_results=args.n_results,
        bm25_index=bm25_index,
    )
    hybrid_results = retrieve_with_query_plan(
        query_plan=query_plan,
        collection=collection,
        search_mode="hybrid",
        n_results=args.n_results,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        rrf_k=args.rrf_k,
    )

    reranker = load_reranker(args.reranker_model)
    hybrid_reranked_results = rerank_results(
        args.question,
        hybrid_results,
        reranker,
    )

    sections = [
        ("Vector only", vector_results),
        ("BM25 only", bm25_results),
        ("Hybrid vector + BM25", hybrid_results),
        ("Hybrid + reranking", hybrid_reranked_results),
    ]

    for title, results in sections:
        print_compact_results(title, results, args.top_k)

    write_comparison_file(
        output_path=args.comparison_output,
        sections=sections,
        args=args,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        query_plan=query_plan,
    )


def main():
    load_env_file()
    args = parse_args()
    embedding_model_name = resolve_embedding_model(args.embedding_model)
    collection_name = collection_name_for_embedding_model(
        args.embedding_model,
        args.collection_name,
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(collection_name)
    print(f"[INFO] Using Chroma collection: {collection_name}", flush=True)
    print(f"[INFO] Question: {args.question}", flush=True)

    if args.compare:
        run_comparison(args, collection, embedding_model_name, collection_name)
        return

    run_single_mode(args, collection, embedding_model_name)


if __name__ == "__main__":
    main()
