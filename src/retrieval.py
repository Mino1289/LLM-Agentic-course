import math
import re
from collections import Counter, defaultdict


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_./%-]+")
SEARCH_MODES = ("vector", "bm25", "hybrid")


def tokenize(text: str):
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    ]


def load_corpus(collection):
    records = collection.get(include=["documents", "metadatas"])

    return [
        {
            "id": doc_id,
            "document": document,
            "metadata": metadata,
        }
        for doc_id, document, metadata in zip(
            records["ids"],
            records["documents"],
            records["metadatas"],
        )
    ]


class BM25Index:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_tokens = []
        self.doc_lengths = []
        self.term_frequencies = []
        self.document_frequencies = defaultdict(int)

        for record in corpus:
            tokens = tokenize(record["document"])
            frequencies = Counter(tokens)

            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.term_frequencies.append(frequencies)

            for token in frequencies:
                self.document_frequencies[token] += 1

        self.doc_count = len(corpus)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count
            if self.doc_count
            else 0
        )

    def idf(self, token: str):
        frequency = self.document_frequencies.get(token, 0)

        return math.log(1 + (self.doc_count - frequency + 0.5) / (frequency + 0.5))

    def score_document(self, query_tokens: list[str], doc_index: int):
        score = 0.0
        doc_length = self.doc_lengths[doc_index]
        frequencies = self.term_frequencies[doc_index]

        if doc_length == 0 or self.avg_doc_length == 0:
            return score

        for token in query_tokens:
            term_frequency = frequencies.get(token, 0)

            if term_frequency == 0:
                continue

            denominator = (
                term_frequency
                + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            )
            score += self.idf(token) * (term_frequency * (self.k1 + 1)) / denominator

        return score

    def search(self, question: str, n_results: int):
        query_tokens = tokenize(question)
        scores = []

        for doc_index, record in enumerate(self.corpus):
            score = self.score_document(query_tokens, doc_index)

            if score > 0:
                scores.append((score, record))

        ranked = sorted(scores, key=lambda item: item[0], reverse=True)

        return [
            {
                "id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
                "bm25_score": score,
                "bm25_rank": rank,
            }
            for rank, (score, record) in enumerate(ranked[:n_results], start=1)
        ]


def vector_search(collection, question: str, embedding_model, n_results: int):
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id": doc_id,
            "document": document,
            "metadata": metadata,
            "vector_distance": distance,
            "vector_rank": rank,
        }
        for rank, (doc_id, document, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        )
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    n_results: int,
    rrf_k: int = 60,
    score_key: str = "hybrid_score",
):
    fused = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]

            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    score_key: 0.0,
                }
            else:
                fused[doc_id].update({
                    key: value
                    for key, value in result.items()
                    if key not in {"document", "metadata", score_key, "matched_queries"}
                })

            matched_queries = fused[doc_id].setdefault("matched_queries", [])
            candidate_queries = result.get("matched_queries")

            if not candidate_queries and result.get("retrieval_query"):
                candidate_queries = [result["retrieval_query"]]

            for query in candidate_queries or []:
                if query not in matched_queries:
                    matched_queries.append(query)

            fused[doc_id][score_key] += 1 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda result: result[score_key],
        reverse=True,
    )[:n_results]


def rerank_results(question: str, results: list[dict], reranker):
    if not results:
        return []

    pairs = [
        [question, result["document"]]
        for result in results
    ]
    rerank_scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, rerank_scores):
        reranked.append({
            **result,
            "rerank_score": float(score),
        })

    return sorted(
        reranked,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )
