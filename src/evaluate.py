import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import math

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunking import CHUNKING_METHODS
from embedding_models import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    resolve_embedding_model,
)
from retrieval import (
    BM25Index,
    SEARCH_MODES,
    load_corpus,
    reciprocal_rank_fusion,
    rerank_results,
    vector_search,
)


CHROMA_DIR = "chroma_db"
SCRIPTS_DIR = Path(__file__).parent
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEST_QUESTIONS = [
    {
        "question": "What did NVIDIA management say about demand for Blackwell during the Q4 2025 earnings call?",
        "expected": {
            "company": "NVIDIA",
            "source_files": ["NVDA_2025_Q4_earnings_call.txt"],
            "keywords": ["Blackwell", "demand"],
        },
    },
    {
        "question": "What is AMD's strategy for AI GPU market share?",
        "expected": {
            "company": "Advanced Micro Devices",
            "source_files": ["AMD_2024_Q4_earnings_call.txt", "AMD_2024_10K.pdf"],
            "keywords": ["AI", "GPU", "market"],
        },
    },
    {
        "question": "What were Intel's foundry revenue trends in 2024?",
        "expected": {
            "company": "Intel",
            "source_files": ["INTC_2024_annual_report.pdf"],
            "keywords": ["foundry", "revenue"],
        },
    },
    {
        "question": "What is TSMC's 2024 capital expenditure?",
        "expected": {
            "company": "Taiwan Semiconductor Manufacturing Company",
            "source_files": ["TSMC_2024_annual_report.pdf", "TSMC_financial_facts_sec.csv"],
            "keywords": ["capital expenditure", "capex"],
        },
    },
    {
        "question": "What are ASML's lithography system sales in 2024?",
        "expected": {
            "company": "ASML",
            "source_files": ["ASML_2024_annual_report.pdf"],
            "keywords": ["lithography", "system", "sales"],
        },
    },
    {
        "question": "What supply chain risks does NVIDIA mention in its 10-K?",
        "expected": {
            "company": "NVIDIA",
            "source_files": ["NVDA_2025_10K.pdf"],
            "keywords": ["supply chain", "risk"],
        },
    },
    {
        "question": "What is AMD's R&D spending trend?",
        "expected": {
            "company": "Advanced Micro Devices",
            "source_files": ["AMD_financial_facts_sec.csv", "AMD_2024_10K.pdf"],
            "keywords": ["research", "development", "R&D"],
        },
    },
]


def collection_name_for(embedding_key: str, chunking: str):
    return f"semiconductor__{embedding_key}__{chunking}"


LEGACY_COLLECTION_EMBEDDING = {
    "semiconductor_reports": "bge-small",
    "semiconductor_reports_bge_base": "bge-base",
    "semiconductor_reports_bge_large": "bge-large",
}


def detect_collection_config(name: str):
    if name in LEGACY_COLLECTION_EMBEDDING:
        return LEGACY_COLLECTION_EMBEDDING[name], "unknown"
    parts = name.split("__", 2)
    if len(parts) == 3 and parts[0] == "semiconductor":
        return parts[1], parts[2]
    return None, None


def compute_relevance(result: dict, expected: dict) -> int:
    meta = result["metadata"]
    company = meta.get("company", "")
    if company != expected["company"]:
        return 0
    source_file = meta.get("source_file", "")
    expected_files = expected.get("source_files", [])
    if any(ef in source_file for ef in expected_files):
        return 2
    return 1


def dcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return relevances[0] + sum(
        rel / math.log2(i + 1) for i, rel in enumerate(relevances[1:], start=2)
    )


def ndcg_at_k(results: list[dict], expected: dict, k: int) -> float:
    actual = [compute_relevance(r, expected) for r in results[:k]]
    ideal = sorted(actual, reverse=True)
    dcg = dcg_at_k(actual, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(results: list[dict], expected: dict, k: int = 5):
    top_k = results[:k]
    relevances = [compute_relevance(r, expected) for r in top_k]
    graded = [i + 1 for i, rel in enumerate(relevances) if rel >= 1]
    exact = [i + 1 for i, rel in enumerate(relevances) if rel >= 2]
    return {
        "recall@k": 1.0 if graded else 0.0,
        "mrr": 1.0 / graded[0] if graded else 0.0,
        "precision@k": len(graded) / k,
        "ndcg@k": ndcg_at_k(results, expected, k),
    }


def run_retrieval(question, collection, search_mode, n_results, embedding_model=None,
                  bm25_index=None, rrf_k=60):
    if search_mode == "vector":
        return vector_search(collection, question, embedding_model, n_results)
    if search_mode == "bm25":
        return bm25_index.search(question, n_results=n_results)
    if search_mode == "hybrid":
        vr = vector_search(collection, question, embedding_model, n_results)
        br = bm25_index.search(question, n_results=n_results)
        return reciprocal_rank_fusion([vr, br], n_results=n_results, rrf_k=rrf_k)
    raise ValueError(f"Unsupported search mode: {search_mode}")


def evaluate_strategy(questions, collection, search_mode, n_results, k,
                      embedding_model=None, bm25_index=None, rrf_k=60, reranker=None):
    agg = defaultdict(list)
    for item in questions:
        results = run_retrieval(
            item["question"], collection, search_mode, n_results,
            embedding_model, bm25_index, rrf_k,
        )
        if reranker:
            results = rerank_results(item["question"], results, reranker)
        metrics = compute_metrics(results, item["expected"], k=k)
        for key, val in metrics.items():
            agg[key].append(val)
    return {key: sum(vals) / len(vals) for key, vals in agg.items()}


def build_collection(embedding_key: str, chunking: str, include_tables: bool,
                     rebuild: bool):
    coll_name = collection_name_for(embedding_key, chunking)
    print(f"[BUILD] {coll_name} (embedding={embedding_key}, chunking={chunking})")
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_index.py"),
        f"--embedding-model={embedding_key}",
        f"--chunking={chunking}",
        f"--collection-name={coll_name}",
    ]
    if rebuild:
        cmd.append("--rebuild")
    if include_tables:
        cmd.append("--include-tables")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] build_index.py failed for {coll_name}")
    return coll_name


def get_or_build_collections(embedding_keys: list[str], chunking_methods: list[str],
                              include_tables: bool, rebuild: bool):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = set(c.name for c in client.list_collections())
    ready = []

    for emb in embedding_keys:
        for chunk in chunking_methods:
            name = collection_name_for(emb, chunk)
            if name in existing and not rebuild:
                ready.append(name)
            else:
                build_collection(emb, chunk, include_tables, rebuild)
                ready.append(name)

    client.clear_system_cache()
    return ready


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG strategy evaluation — compares all combinations."
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        choices=list(EMBEDDING_MODELS.keys()),
        default=list(EMBEDDING_MODELS.keys()),
        help="Embedding models to evaluate (default: all).",
    )
    parser.add_argument(
        "--chunking-methods",
        nargs="+",
        choices=CHUNKING_METHODS,
        default=list(CHUNKING_METHODS),
        help="Chunking methods to evaluate (default: all).",
    )
    parser.add_argument(
        "--search-modes",
        nargs="+",
        choices=SEARCH_MODES,
        default=list(SEARCH_MODES),
        help="Search modes to evaluate (default: all).",
    )
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        default=True,
        help="Include reranking variant.",
    )
    parser.add_argument(
        "--no-rerank",
        dest="with_rerank",
        action="store_false",
        help="Exclude reranking variant.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=30,
        help="Number of candidates to retrieve.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for metrics.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild all collections even if they exist.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Include PDF table extraction in builds.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build collections, skip evaluation.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.txt",
        help="Output file for the comparison results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("COMPREHENSIVE RAG EVALUATION")
    print("=" * 80)
    print(f"Embedding models: {', '.join(args.embedding_models)}")
    print(f"Chunking methods: {', '.join(args.chunking_methods)}")
    print(f"Search modes: {', '.join(args.search_modes)}")
    print(f"Rerank: {'yes' if args.with_rerank else 'no'}")
    print(f"Test questions: {len(TEST_QUESTIONS)}")
    print(f"Metrics top-k: {args.k}")
    print()

    collection_names = get_or_build_collections(
        args.embedding_models, args.chunking_methods,
        args.include_tables, args.rebuild,
    )
    print()

    if args.build_only:
        print("[DONE] Collections built. Skipping evaluation.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL) if args.with_rerank else None

    all_results = []

    for coll_name in collection_names:
        emb_key, chunking = detect_collection_config(coll_name)
        if not emb_key:
            print(f"[SKIP] Cannot detect config for collection: {coll_name}")
            continue

        print(f"[EVAL] {coll_name}")
        collection = client.get_collection(coll_name)

        emb_model_name = resolve_embedding_model(emb_key)
        print(f"  Loading embedding model {emb_model_name}...")
        embedding_model = SentenceTransformer(emb_model_name)

        print(f"  Building BM25 index...")
        corpus = load_corpus(collection)
        bm25_index = BM25Index(corpus)

        for mode in args.search_modes:
            metrics = evaluate_strategy(
                questions=TEST_QUESTIONS,
                collection=collection,
                search_mode=mode,
                n_results=args.n_results,
                k=args.k,
                embedding_model=embedding_model if mode in ("vector", "hybrid") else None,
                bm25_index=bm25_index if mode in ("bm25", "hybrid") else None,
                reranker=None,
            )
            tag = f"{coll_name} | {mode}"
            all_results.append((tag, emb_key, chunking, mode, False, metrics))

            if args.with_rerank:
                metrics_r = evaluate_strategy(
                    questions=TEST_QUESTIONS,
                    collection=collection,
                    search_mode=mode,
                    n_results=args.n_results,
                    k=args.k,
                    embedding_model=embedding_model if mode in ("vector", "hybrid") else None,
                    bm25_index=bm25_index if mode in ("bm25", "hybrid") else None,
                    reranker=reranker,
                )
                all_results.append((tag, emb_key, chunking, mode, True, metrics_r))

    all_results.sort(key=lambda r: r[5]["ndcg@k"], reverse=True)

    lines = []
    lines.append("=" * 130)
    lines.append("RAG STRATEGY LEADERBOARD (sorted by NDCG@k)")
    lines.append("=" * 130)
    lines.append(f"{'Rank':<5} {'Embedding':<12} {'Chunking':<12} {'Search':<10} {'Rerank':<8} {'Recall@k':<10} {'MRR':<10} {'Precision@k':<12} {'NDCG@k':<10}")
    lines.append("-" * 130)

    for rank, (tag, emb, chunk, mode, do_rerank, m) in enumerate(all_results, start=1):
        rerank_str = "yes" if do_rerank else "no"
        lines.append(
            f"{rank:<5} {emb:<12} {chunk:<12} {mode:<10} {rerank_str:<8} "
            f"{m['recall@k']:<10.3f} {m['mrr']:<10.3f} {m['precision@k']:<12.3f} {m['ndcg@k']:<10.3f}"
        )

    lines.append("-" * 130)

    best = all_results[0]
    lines.append(f"\n🏆 Best strategy: {best[1]} | chunking={best[2]} | search={best[3]} | rerank={'yes' if best[4] else 'no'}")
    lines.append(f"   NDCG@k={best[5]['ndcg@k']:.3f}, MRR={best[5]['mrr']:.3f}, Recall@k={best[5]['recall@k']:.3f}")

    output = "\n".join(lines)
    print("\n" + output + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"[DONE] Results written to {output_path}")


if __name__ == "__main__":
    main()
