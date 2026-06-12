"""TOON audit Option 1 (P1+P2+P3) on real indexed volume.

This script:
- P1-A: measures token savings on real RAG retrieval across the indexed tickers
- P1-B: confirms the OpenAI API blocker (theoretical only)
- P2-C: measures memory/chat context formatting (data-independent)
- P3: scans remaining serializers in src/ for further TOON candidates

Usage: python -m scripts.audit_toon_real_volume
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

import toon_format
from src.paths import CHROMA_DB_DIR
from src.tools.descriptions import format_rag_excerpts

# Avoid loading dotenv since this is a one-off audit
import chromadb
from src.graph.memory_store import format_chat_context, format_memory_context


_ENC = tiktoken.get_encoding("cl100k_base")
_TICKERS_20 = [
    "NVDA", "ASML", "TSM", "AMD", "AVGO", "ARM", "MSFT", "AAPL", "INTC", "QCOM",
    "MC.PA", "RMS.PA", "KER.PA", "AIR.PA", "TTE.PA", "BRK-B", "JPM", "CAT", "NKE", "XOM",
]
_TICKERS_INDEXED = ["AMD", "ARM", "MSFT", "NVDA", "AAPL", "AVGO"]


def _count(text: str) -> int:
    return len(_ENC.encode(text))


def _measure(name: str, json_payload: str, toon_payload: str) -> dict:
    json_t = _count(json_payload)
    toon_t = _count(toon_payload)
    delta = json_t - toon_t
    pct = (delta / json_t * 100) if json_t else 0
    return {
        "name": name,
        "json_tokens": json_t,
        "toon_tokens": toon_t,
        "saved_tokens": delta,
        "saved_percent": round(pct, 1),
    }


def _legacy_rag_excerpts(chunks, metadatas) -> str:
    """Reconstruct the legacy 'key=value' format from before TOON integration."""
    lines = ["Sources SEC :"]
    for idx, (text, meta) in enumerate(zip(chunks, metadatas), start=1):
        meta = meta or {}
        ticker = str(meta.get("ticker", "")).upper() or "?"
        year = meta.get("year") or "?"
        file_type = meta.get("file_type") or "?"
        section = meta.get("section") or "?"
        source = meta.get("source") or "?"
        lines.append(
            f"[{idx}] {ticker}/{year}/{file_type} | section={section} | source={source}\n"
            f"    {text}\n"
        )
    return "\n".join(lines)


def audit_p1a_rag_excerpts() -> list[dict]:
    """Measure P1-A (format_rag_excerpts) on real chunks pulled from the vectorstore.

    Note: we cannot use live ChromaDB .query() because its default embedding
    function (MiniLM 384d) doesn't match the indexed dimension (OpenAI 1536d).
    Instead we read the chunks + metadatas directly and format them.
    """
    print("\n=== P1-A : RAG excerpts format_rag_excerpts (real volume) ===")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_collection("finance_rag_semantic_vector")

    results = []
    # Group chunks by ticker (sample 5 per ticker)
    all_data = col.get(include=["documents", "metadatas"], limit=4458)
    by_ticker: dict[str, list[tuple[str, dict]]] = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        ticker = meta.get("ticker", "UNKNOWN")
        by_ticker.setdefault(ticker, []).append((doc, meta))

    for ticker in sorted(by_ticker.keys()):
        chunks_with_meta = by_ticker[ticker][:5]  # 5 chunks per ticker
        chunks = [c for c, _ in chunks_with_meta]
        metadatas = [m for _, m in chunks_with_meta]
        if not chunks:
            print(f"  ⚠️  {ticker}: no chunks found")
            continue

        toon_payload = format_rag_excerpts(chunks, metadatas)
        legacy_payload = _legacy_rag_excerpts(chunks, metadatas)

        m = _measure(f"P1-A/{ticker}", legacy_payload, toon_payload)
        results.append(m)
        print(f"  {ticker:6s}: legacy={m['json_tokens']:4d} tok → TOON={m['toon_tokens']:4d} tok "
              f"(saved {m['saved_tokens']:+d} = {m['saved_percent']:+.1f}%)")

    return results


def audit_p1b_tool_schema() -> list[dict]:
    """P1-B theoretical gain on tool JSON schemas (blocked by OpenAI API)."""
    print("\n=== P1-B : Tool schema JSON → TOON (BLOCKED) ===")

    tool_schema_json = json.dumps({
        "type": "function",
        "function": {
            "name": "sec_filings_rag_tool",
            "description": "Recherche dans les rapports SEC 10-K/10-Q/8-K indexés.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question utilisateur"},
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "year_min": {"type": "integer"},
                    "year_max": {"type": "integer"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }, indent=2)

    try:
        toon_payload = toon_format.encode(json.loads(tool_schema_json))
    except Exception as e:
        return [{
            "name": "P1-B/tool_schema",
            "json_tokens": _count(tool_schema_json),
            "toon_tokens": 0,
            "saved_tokens": 0,
            "saved_percent": 0,
            "note": f"encode failed: {e}",
        }]

    m = _measure("P1-B/tool_schema", tool_schema_json, toon_payload)
    m["note"] = "BLOCKED: OpenAI API requires tools: [{type: function, function: {...}}] JSON"
    m["theoretical"] = True
    print(f"  Theoretical: legacy={m['json_tokens']} tok → TOON={m['toon_tokens']} tok "
          f"(saved {m['saved_percent']:+.1f}%)")
    print(f"  Status: BLOCKED by OpenAI API contract — needs upstream validation")

    return [m]


def audit_p2c_memory_context() -> list[dict]:
    """Measure P2-C (memory + chat context) on simulated multi-turn conversation."""
    print("\n=== P2-C : Memory + chat context (data-independent) ===")

    results = []

    # Multi-turn memory context (10 turns)
    summary = (
        "L'utilisateur a demandé l'analyse de NVDA sur 2024-2026. "
        "Il s'intéresse aux data centers, à la concurrence avec AMD/INTC, "
        "et au pricing des GPU H100/B200."
    )
    turns = [
        {"role": "user", "content": "Donne-moi les risques NVDA 2024"},
        {"role": "assistant", "content": "Voici les 5 principaux risques..."},
        {"role": "user", "content": "Et pour AMD ?"},
        {"role": "assistant", "content": "AMD a des risques similaires mais aussi..."},
        {"role": "user", "content": "Compare les data centers"},
        {"role": "assistant", "content": "Les data centers représentent 78% du CA NVDA..."},
        {"role": "user", "content": "Et le pricing H100 ?"},
        {"role": "assistant", "content": "Le H100 se vendait entre 25-40K$ début 2024..."},
        {"role": "user", "content": "Quel est l'impact sur la marge ?"},
        {"role": "assistant", "content": "La marge brute NVDA est de 75.7%..."},
    ]
    toon_payload = format_memory_context(summary, turns)
    legacy_json = json.dumps({"summary": summary, "turns": turns}, ensure_ascii=False, indent=2)

    m = _measure("P2-C/memory_context", legacy_json, toon_payload)
    results.append(m)
    print(f"  memory_context: legacy={m['json_tokens']:4d} tok → TOON={m['toon_tokens']:4d} tok "
          f"(saved {m['saved_percent']:+.1f}%)")

    # Chat context (5 turns)
    chat = [
        {"role": "user", "content": "Analyse NVDA 2025"},
        {"role": "assistant", "content": "Le rapport 10-K de NVDA pour FY2025 montre..."},
        {"role": "user", "content": "Et le risk factors ?"},
        {"role": "assistant", "content": "Item 1A liste 12 risques principaux..."},
        {"role": "user", "content": "Focus sur la supply chain"},
        {"role": "assistant", "content": "NVDA dépend de TSMC pour la fabrication..."},
    ]
    toon_chat = format_chat_context(chat)
    legacy_chat = json.dumps({"messages": chat}, ensure_ascii=False, indent=2)
    m2 = _measure("P2-C/chat_context", legacy_chat, toon_chat)
    results.append(m2)
    print(f"  chat_context:   legacy={m2['json_tokens']:4d} tok → TOON={m2['toon_tokens']:4d} tok "
          f"(saved {m2['saved_percent']:+.1f}%)")

    return results


def audit_p3_other_serializers() -> list[dict]:
    """P3 — scan remaining serializers in rag/ for TOON candidates."""
    print("\n=== P3 : scan autres serializers rag/ ===")

    # P3-A: NLI judge prompt (mixed format)
    nli_json = json.dumps({
        "claim": "NVDA a généré 60.9B$ de revenus en FY2024",
        "context_chunks": [
            "Total revenue for fiscal 2024 was $60.9 billion, up 125% year over year.",
            "Data Center revenue was a record $47.5 billion.",
        ],
        "metadata": {"ticker": "NVDA", "year": 2024, "form": "10-K"},
        "instruction": "Classify the claim as supported/partial/unsupported.",
    }, ensure_ascii=False, indent=2)

    try:
        toon_nli = toon_format.encode(json.loads(nli_json))
    except Exception as e:
        return [{
            "name": "P3/nli_prompt",
            "json_tokens": _count(nli_json),
            "toon_tokens": 0,
            "saved_tokens": 0,
            "saved_percent": 0,
            "note": f"encode failed: {e}",
        }]

    m = _measure("P3/nli_prompt", nli_json, toon_nli)
    m["note"] = "NLI judge prompt — mixed format (text+structured), partial gain expected"
    print(f"  NLI prompt:     legacy={m['json_tokens']:4d} tok → TOON={m['toon_tokens']:4d} tok "
          f"(saved {m['saved_percent']:+.1f}%) — mixed-format, partial gain")

    return [m]


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("TOON AUDIT — Option 1 (P1+P2+P3) — volume réel indexé")
    print("=" * 70)
    print(f"Univers 20 tickers : {len(_TICKERS_20)} (15 SEC + 5 .PA)")
    print(f"Indexés (partiel)  : {len(_TICKERS_INDEXED)} — {_TICKERS_INDEXED}")
    print(f"Cause partielle    : GitHub Models rate limit 150/jour (text-embedding-3-small)")

    all_results = []
    all_results += audit_p1a_rag_excerpts()
    all_results += audit_p1b_tool_schema()
    all_results += audit_p2c_memory_context()
    all_results += audit_p3_other_serializers()

    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    total_json = sum(r["json_tokens"] for r in all_results)
    total_toon = sum(r["toon_tokens"] for r in all_results)
    saved = total_json - total_toon
    pct = (saved / total_json * 100) if total_json else 0
    print(f"Total tokens legacy  : {total_json:>6d}")
    print(f"Total tokens TOON    : {total_toon:>6d}")
    print(f"Tokens économisés    : {saved:>+6d} ({pct:+.1f}%)")
    print(f"Durée audit          : {time.time() - t0:.1f}s")

    out = {
        "audit_date": time.strftime("%Y-%m-%d"),
        "universe_size": len(_TICKERS_20),
        "indexed_size": len(_TICKERS_INDEXED),
        "indexed_tickers": _TICKERS_INDEXED,
        "results": all_results,
        "totals": {
            "json_tokens": total_json,
            "toon_tokens": total_toon,
            "saved_tokens": saved,
            "saved_percent": round(pct, 1),
        },
    }
    out_path = Path("docs/superpowers/plans/2026-06-06-toon-audit-real-volume-data.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRésultats JSON : {out_path}")


if __name__ == "__main__":
    main()
