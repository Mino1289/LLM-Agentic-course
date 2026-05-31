import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import time
from dataclasses import dataclass, field
from typing import Generator, Literal, Optional

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.llm_provider import LLMProvider
from rag.paths import CHROMA_DB_DIR, ENV_FILE, PROCESSED_DATA_DIR

load_dotenv(ENV_FILE)

ChunkStrategy = Literal["semantic"]

DEFAULT_MAX_CHUNK_SIZE = 1500
DEFAULT_MIN_CHUNK_SIZE = 80

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DAILY_EMBEDDING_LIMIT = 1000
DEFAULT_EMBEDDING_RPM = 100
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_RETRIES = 3


@dataclass
class EmbeddingPlan:
    chunk_strategy: str
    collection_name: str
    section_files: int
    total_chunks: int
    already_indexed: int
    missing_chunks: int
    daily_limit: int
    daily_used: int
    daily_remaining: int
    embeddable_now: int
    deferred_chunks: int
    estimated_minutes: float
    rpm_limit: int

    def summary(self) -> str:
        lines = [
            f"=== Plan d'embedding ({self.chunk_strategy}) ===",
            f"Collection ChromaDB : {self.collection_name}",
            f"Fichiers .txt source  : {self.section_files}",
            f"Chunks totaux         : {self.total_chunks:,}",
            f"Déjà indexés          : {self.already_indexed:,}",
            f"Manquants             : {self.missing_chunks:,}",
            "",
            f"Quota journalier      : {self.daily_used:,} / {self.daily_limit:,} utilisés",
            f"Quota restant aujourd'hui : {self.daily_remaining:,}",
            f"Embeddings prévus now : {self.embeddable_now:,}",
            f"Reportés (quota)      : {self.deferred_chunks:,}",
            f"Durée estimée         : ~{self.estimated_minutes:.1f} min "
            f"({self.rpm_limit} req/min)",
        ]
        if self.missing_chunks > self.daily_remaining:
            days = (self.deferred_chunks // max(self.daily_remaining, 1)) + (
                1 if self.deferred_chunks % max(self.daily_remaining, 1) else 0
            )
            lines.append(
                f"\n⚠️  Indexation complète : ~{days} jour(s) supplémentaire(s) au rythme actuel."
            )
        elif self.missing_chunks == 0:
            lines.append("\n✅ Rien à embedder — index vectoriel à jour.")
        else:
            lines.append("\n✅ Tout peut être indexé aujourd'hui avec le quota restant.")
        return "\n".join(lines)


def embedding_sleep_seconds(rpm_limit: int = DEFAULT_EMBEDDING_RPM) -> float:
    return 60.0 / max(rpm_limit, 1)


def iter_batches(items: list[int], batch_size: int) -> Generator[list[int], None, None]:
    size = max(1, batch_size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass
class RetrievalResult:
    chunks: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    search_mode: str = "vector_rerank"
    reranking_enabled: bool = False


def chunk_text_fixed(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks


def chunk_text_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n| ", "\n\n", "\n| ", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def split_semantic_blocks(text: str) -> list[str]:
    """Split into paragraphs and markdown tables without breaking table rows."""
    blocks: list[str] = []
    current_lines: list[str] = []
    in_table = False

    for line in text.split("\n"):
        stripped = line.strip()
        is_table_line = _is_markdown_table_line(stripped)

        if is_table_line:
            if current_lines and not in_table:
                block = "\n".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
            in_table = True
            current_lines.append(line)
            continue

        if in_table:
            block = "\n".join(current_lines).strip()
            if block:
                blocks.append(block)
            current_lines = []
            in_table = False

        if not stripped:
            if current_lines:
                block = "\n".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        block = "\n".join(current_lines).strip()
        if block:
            blocks.append(block)

    if not blocks and text.strip():
        blocks = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]

    return blocks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])', text)
    return [part.strip() for part in parts if part.strip()]


def split_markdown_table(table: str, max_size: int) -> list[str]:
    lines = [line for line in table.split("\n") if line.strip()]
    if len(lines) <= 2:
        return [table]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]
    prefix = f"{header}\n{separator}"
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = prefix + "\n" + "\n".join(candidate_rows)
        if len(candidate) > max_size and current_rows:
            chunks.append(prefix + "\n" + "\n".join(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append(prefix + "\n" + "\n".join(current_rows))

    return chunks or [table]


def split_block_by_sentences(block: str, max_size: int) -> list[str]:
    if _is_markdown_table_line(block.split("\n", 1)[0].strip()):
        return split_markdown_table(block, max_size)

    sentences = split_sentences(block)
    if len(sentences) <= 1:
        return [block]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence if len(sentence) <= max_size else sentence[:max_size]
    if current:
        chunks.append(current)
    return chunks


def merge_semantic_blocks(
    blocks: list[str],
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > max_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_block_by_sentences(block, max_size))
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    return [chunk for chunk in merged if chunk.strip()]


def chunk_text_semantic(
    text: str,
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[str]:
    """Chunk by paragraphs/tables first, then sentences for oversized blocks."""
    if not text.strip():
        return []
    blocks = split_semantic_blocks(text)
    return merge_semantic_blocks(blocks, max_size=max_size, min_size=min_size)


def parse_processed_filename(file_path: str) -> tuple[str, str]:
    basename = os.path.basename(file_path)
    stem = basename[:-4] if basename.endswith(".txt") else basename
    if "__" in stem:
        source, section = stem.rsplit("__", 1)
        return source, section
    return stem, "unknown"


def extract_year_from_source(source: str) -> str:
    patterns = [
        r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?(0[1-9]|[12]\d|3[01])",
        r"(20\d{2})",
        r"(19\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return ""


def extract_ticker_from_source(source: str) -> str:
    match = re.match(r"^([a-zA-Z]{1,5})[-_]", source)
    if match:
        return match.group(1).upper()
    return ""


def extract_file_type_from_source(source: str) -> str:
    ext = os.path.splitext(source)[1].lower().lstrip(".")
    return ext or "txt"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class HybridRAG:
    def __init__(
        self,
        chunk_strategy: ChunkStrategy = "semantic",
        search_mode: str = "vector",
        use_reranking: bool = True,
        reranker_model: str = DEFAULT_RERANKER,
        collection_name: Optional[str] = None,
    ):
        self.chunk_strategy = chunk_strategy
        self.search_mode = search_mode
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model

        self.documents: list[str] = []
        self.doc_metadata: list[dict] = []
        self.chunk_ids: list[str] = []
        self._reranker = None

        self.provider = LLMProvider()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        resolved_collection = collection_name or "finance_rag_semantic_vector"
        self.collection = self.chroma_client.get_or_create_collection(
            name=resolved_collection
        )

    def _get_chunker(self):
        # Single strategy for this project: semantic chunking.
        return chunk_text_semantic

    def _build_metadata(self, source: str, section: str, chunk_index: int) -> dict:
        return {
            "source": source,
            "section": section,
            "chunk_index": chunk_index,
            "file_type": extract_file_type_from_source(source),
            "year": extract_year_from_source(source),
            "ticker": extract_ticker_from_source(source),
            "chunk_strategy": self.chunk_strategy,
        }

    def _build_corpus(
        self, max_files: Optional[int] = None
    ) -> tuple[list[str], list[dict], list[str], int]:
        files = sorted(PROCESSED_DATA_DIR.glob("*.txt"))
        if max_files:
            files = files[:max_files]

        chunker = self._get_chunker()
        all_chunks: list[str] = []
        all_metadata: list[dict] = []
        all_ids: list[str] = []

        for file_path in files:
            source, section = parse_processed_filename(str(file_path))
            text = file_path.read_text(encoding="utf-8")

            if not text.strip():
                continue

            for i, chunk in enumerate(chunker(text)):
                chunk_id = f"{source}__{section}__{self.chunk_strategy}__{i}"
                all_chunks.append(chunk)
                all_metadata.append(self._build_metadata(source, section, i))
                all_ids.append(chunk_id)

        return all_chunks, all_metadata, all_ids, len(files)

    def get_embedding_plan(
        self,
        max_files: Optional[int] = None,
        daily_quota_used: int = 0,
        daily_quota_limit: int = DEFAULT_DAILY_EMBEDDING_LIMIT,
        max_new_embeddings: Optional[int] = None,
        rpm_limit: int = DEFAULT_EMBEDDING_RPM,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> EmbeddingPlan:
        all_chunks, _, all_ids, section_files = self._build_corpus(max_files=max_files)
        total_chunks = len(all_chunks)

        existing_ids = set(self.collection.get()["ids"])
        missing_count = sum(1 for cid in all_ids if cid not in existing_ids)
        already_indexed = total_chunks - missing_count

        daily_remaining = max(0, daily_quota_limit - daily_quota_used)
        budget = daily_remaining if max_new_embeddings is None else max_new_embeddings
        budget = max(0, min(budget, daily_remaining))
        embeddable_now = min(missing_count, budget)
        deferred = max(0, missing_count - embeddable_now)

        estimated_requests = embeddable_now / max(1, embedding_batch_size)
        return EmbeddingPlan(
            chunk_strategy=self.chunk_strategy,
            collection_name=self.collection.name,
            section_files=section_files,
            total_chunks=total_chunks,
            already_indexed=already_indexed,
            missing_chunks=missing_count,
            daily_limit=daily_quota_limit,
            daily_used=daily_quota_used,
            daily_remaining=daily_remaining,
            embeddable_now=embeddable_now,
            deferred_chunks=deferred,
            estimated_minutes=estimated_requests / max(rpm_limit, 1),
            rpm_limit=rpm_limit,
        )

    def load_and_index_data(
        self,
        max_files: Optional[int] = None,
        max_new_embeddings: Optional[int] = None,
        daily_quota_used: int = 0,
        daily_quota_limit: int = DEFAULT_DAILY_EMBEDDING_LIMIT,
        rpm_limit: int = DEFAULT_EMBEDDING_RPM,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_embedding_retries: int = DEFAULT_EMBEDDING_RETRIES,
        dry_run: bool = False,
    ) -> EmbeddingPlan:
        print(f"Loading and indexing data (chunk_strategy={self.chunk_strategy})...")

        all_chunks, all_metadata, all_ids, _ = self._build_corpus(max_files=max_files)
        self.documents = all_chunks
        self.doc_metadata = all_metadata
        self.chunk_ids = all_ids

        plan = self.get_embedding_plan(
            max_files=max_files,
            daily_quota_used=daily_quota_used,
            daily_quota_limit=daily_quota_limit,
            max_new_embeddings=max_new_embeddings,
            rpm_limit=rpm_limit,
            embedding_batch_size=embedding_batch_size,
        )
        print(plan.summary())

        if not self.documents:
            print("No documents to index.")
            return plan

        if dry_run:
            print("\nMode --plan : aucun embedding lancé.")
            return plan

        if plan.missing_chunks == 0:
            print("All chunks are already indexed in ChromaDB.")
            return plan

        if plan.embeddable_now == 0:
            print(
                "\n⛔ Quota journalier épuisé — relance demain ou augmentez le quota restant."
            )
            return plan

        existing_ids = set(self.collection.get()["ids"])
        missing_indices = [i for i, cid in enumerate(all_ids) if cid not in existing_ids]
        indices_to_embed = missing_indices[: plan.embeddable_now]

        print(f"\nEmbedding {len(indices_to_embed)} / {plan.missing_chunks} chunks manquants...")
        print(
            f"Batch size={embedding_batch_size}, target rate={rpm_limit} req/min, "
            f"retries={max_embedding_retries}"
        )

        sleep_sec = embedding_sleep_seconds(rpm_limit)
        embedded = 0
        for batch_num, batch_indices in enumerate(
            iter_batches(indices_to_embed, embedding_batch_size), start=1
        ):
            texts = [self.documents[idx] for idx in batch_indices]
            metadatas = [self.doc_metadata[idx] for idx in batch_indices]
            ids = [all_ids[idx] for idx in batch_indices]

            embeddings: list[list[float]] | None = None
            for attempt in range(1, max_embedding_retries + 1):
                try:
                    embeddings = self.provider.embed(texts)
                    if len(embeddings) != len(texts):
                        raise ValueError(
                            f"Embedding response size mismatch: {len(embeddings)} vs {len(texts)}"
                        )
                    break
                except Exception as e:
                    if attempt >= max_embedding_retries:
                        print(f"Error embedding batch {batch_num} after {attempt} tries: {e}")
                        embeddings = None
                        break
                    backoff = min(8.0, 2 ** (attempt - 1))
                    print(
                        f"Embedding batch {batch_num} failed (attempt {attempt}/{max_embedding_retries}): {e}. "
                        f"Retry in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)

            if embeddings is None:
                break

            try:
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                embedded += len(batch_indices)
                if batch_num % 5 == 0 or embedded == len(indices_to_embed):
                    print(f"  Progression : {embedded}/{len(indices_to_embed)}")
                time.sleep(sleep_sec)
            except Exception as e:
                print(f"Error writing batch {batch_num} to ChromaDB: {e}")
                break

        print(f"Vector indexing complete. Total in DB: {self.collection.count()}")
        if plan.deferred_chunks > 0:
            print(
                f"⏳ {plan.deferred_chunks} chunks restants — "
                f"relancez demain avec --quota-used mis à jour."
            )
        return plan

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model, device="cpu")
        return self._reranker

    def _vector_search(self, query: str, top_k: int = 10) -> list[int]:
        try:
            query_embedding = self.provider.embed([query])[0]

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            returned_ids = results["ids"][0]
            id_to_index = {cid: i for i, cid in enumerate(self.chunk_ids)}
            indices = [id_to_index[cid] for cid in returned_ids if cid in id_to_index]
            return indices
        except Exception as e:
            print(f"Vector search warning/error: {e}")
            return []

    def _apply_metadata_filter(self, indices: list[int], metadata_filter: Optional[dict]) -> list[int]:
        if not metadata_filter:
            return indices
        filtered = []
        for idx in indices:
            metadata = self.doc_metadata[idx]
            if all(metadata.get(k) == v for k, v in metadata_filter.items()):
                filtered.append(idx)
        return filtered

    def _deduplicate_indices(self, indices: list[int]) -> list[int]:
        seen = set()
        deduped = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        return deduped

    def _rerank(self, query: str, indices: list[int], top_k: int = 10) -> list[int]:
        if not indices:
            return []
        try:
            pairs = [(query, self.documents[idx]) for idx in indices]
            scores = self._get_reranker().predict(pairs)
            ranked = sorted(
                zip(indices, scores), key=lambda item: item[1], reverse=True
            )
            return [idx for idx, _ in ranked[:top_k]]
        except Exception as e:
            print(f"Reranker warning/error: {e}")
            return indices[:top_k]

    def retrieve(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
        top_k: int = 10,
        candidate_pool: int = 50,
    ) -> RetrievalResult:
        mode = "vector_rerank"
        rerank_enabled = self.use_reranking if use_reranking is None else use_reranking

        start = time.perf_counter()
        candidate_indices: list[int] = []

        candidate_indices.extend(self._vector_search(query, top_k=min(candidate_pool, 30)))

        candidate_indices = self._deduplicate_indices(candidate_indices)
        candidate_indices = self._apply_metadata_filter(candidate_indices, metadata_filter)

        retrieval_latency_ms = (time.perf_counter() - start) * 1000

        rerank_start = time.perf_counter()
        if rerank_enabled and candidate_indices:
            final_indices = self._rerank(query, candidate_indices, top_k=top_k)
        else:
            final_indices = candidate_indices[:top_k]

        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

        return RetrievalResult(
            chunks=[self.documents[i] for i in final_indices],
            metadatas=[self.doc_metadata[i] for i in final_indices],
            chunk_indices=final_indices,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            search_mode=mode,
            reranking_enabled=rerank_enabled,
        )

    def count_context_tokens(self, chunks: list[str]) -> int:
        context = "\n\n---\n\n".join(chunks)
        return self.provider.estimate_tokens(context)

    def build_prompt(self, query: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return f"""Tu es un analyste buy-side spécialisé actions.
En te basant uniquement sur les extraits suivants, fais une synthèse financière structurée.

Extraits:
{context}

Question: {query}

Format attendu:
1) Résumé exécutif (3-5 lignes)
2) Signaux positifs
3) Signaux de risque
4) Conclusion orientée décision (surveiller/renforcer/réduire) avec justification
"""

    def answer_question(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[str, RetrievalResult]:
        retrieval = self.retrieve(
            query,
            search_mode=search_mode,
            use_reranking=use_reranking,
            metadata_filter=metadata_filter,
        )
        prompt = self.build_prompt(query, retrieval.chunks)

        response = self.provider.generate(prompt)
        return response, retrieval

    def answer_question_stream(
        self,
        query: str,
        search_mode: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        metadata_filter: Optional[dict] = None,
    ) -> Generator[str, None, RetrievalResult]:
        retrieval = self.retrieve(
            query,
            search_mode=search_mode,
            use_reranking=use_reranking,
            metadata_filter=metadata_filter,
        )
        prompt = self.build_prompt(query, retrieval.chunks)

        for chunk in self.provider.generate_stream(prompt):
            yield chunk

        return retrieval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid RAG — planifier ou lancer l'indexation vectorielle."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Afficher le nombre de chunks et le quota sans embedder (défaut).",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Lancer les embeddings (respecte le quota restant).",
    )
    parser.add_argument(
        "--strategy",
        choices=["semantic"],
        default="semantic",
    )
    parser.add_argument(
        "--quota-used",
        type=int,
        default=int(os.getenv("EMBEDDING_DAILY_USED", "0")),
        help="Embeddings déjà consommés aujourd'hui (ex: 269).",
    )
    parser.add_argument(
        "--quota-limit",
        type=int,
        default=int(os.getenv("EMBEDDING_DAILY_LIMIT", str(DEFAULT_DAILY_EMBEDDING_LIMIT))),
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Plafond explicite d'embeddings pour cette session (≤ quota restant).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("EMBEDDING_RPM", str(DEFAULT_EMBEDDING_RPM))),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
        help="Nombre de chunks embeddes par requete API.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("EMBEDDING_MAX_RETRIES", str(DEFAULT_EMBEDDING_RETRIES))),
        help="Nombre max de tentatives par batch en cas d'erreur API.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if not args.embed:
        args.plan = True

    rag = HybridRAG(chunk_strategy=args.strategy)
    rag.load_and_index_data(
        max_files=args.max_files,
        max_new_embeddings=args.max_new,
        daily_quota_used=args.quota_used,
        daily_quota_limit=args.quota_limit,
        rpm_limit=args.rpm,
        embedding_batch_size=args.batch_size,
        max_embedding_retries=args.retries,
        dry_run=args.plan and not args.embed,
    )

    if args.plan and not args.embed:
        print(
            "\nPour lancer l'embedding : "
            f"python3 rag/hybrid_rag.py --embed --strategy {args.strategy} "
            f"--quota-used {args.quota_used}"
        )
