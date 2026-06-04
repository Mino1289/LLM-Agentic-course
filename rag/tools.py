from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from rag.config import TRACKED_TICKERS
from rag.nodes.decompose_node import decompose_query
from rag.nodes.rerank_node import _balanced_rerank_indices, _ticker_counts
from rag.nodes.retrieval_node import multi_retrieve_node
from rag.nodes.tool_nodes import fetch_price_context
from rag.paths import REPORTS_DIR, ensure_dir

ALLOWED_DOC_TYPES = ["10-K", "10-Q", "8-K", "EARNINGS_CALL"]

SEC_FILINGS_RAG_DESCRIPTION = (
    "Search SEC filings and earnings call transcripts in ChromaDB. "
    "Filter by document type: 10-K (annual), 10-Q (quarterly), 8-K (events), "
    "EARNINGS_CALL (conference call transcripts). "
    "Examples: 'MSFT 10-K risk factors 2024', 'NVDA earnings call transcript 2024'."
)

MARKET_PRICE_DESCRIPTION = (
    "Fetch stock price performance for tracked tickers (NVDA, AMD, MSFT) "
    "between start_date and end_date (YYYY-MM-DD)."
)

EXPORT_REPORT_DESCRIPTION = (
    "Save an investment report to the reports/ folder. "
    "Supported formats: md (default). Returns the file path."
)

VALIDATE_CLAIMS_DESCRIPTION = (
    "Check whether factual claims are supported by RAG excerpts already retrieved. "
    "Call after sec_filings_rag_tool. Returns supported/partial/unsupported per claim with source refs."
)

SIMULATE_PORTFOLIO_DESCRIPTION = (
    "Simulate a fictional portfolio allocation (no real trades). "
    "Weights must sum to 100% across NVDA, AMD, MSFT only. Max 3 positions."
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "et",
        "ou",
        "en",
        "sur",
        "pour",
        "est",
        "sont",
        "que",
        "qui",
        "avec",
        "dans",
        "par",
    }
)

_MAX_NOTIONAL_USD = 1_000_000
_WEIGHT_TOLERANCE = 0.01


def _normalize_doc_types(doc_types: Any) -> list[str]:
    if doc_types is None:
        return []
    if isinstance(doc_types, str):
        raw = [part.strip() for part in re.split(r"[,|]", doc_types) if part.strip()]
    elif isinstance(doc_types, list):
        raw = [str(item).strip() for item in doc_types]
    else:
        return []
    normalized: list[str] = []
    for item in raw:
        up = item.upper().replace(" ", "_")
        if up in {"EARNINGSCALL", "EARNINGS-CALL"}:
            up = "EARNINGS_CALL"
        if up in ALLOWED_DOC_TYPES and up not in normalized:
            normalized.append(up)
    return normalized


def _normalize_tickers(tickers: Any) -> list[str]:
    allowed = set(TRACKED_TICKERS)
    if tickers is None:
        return []
    if isinstance(tickers, str):
        raw = re.split(r"[,|\s]+", tickers.strip())
    elif isinstance(tickers, list):
        raw = [str(t).strip() for t in tickers]
    else:
        return []
    result: list[str] = []
    for token in raw:
        up = token.upper()
        if up in allowed and up not in result:
            result.append(up)
    return result


def _normalize_years(years: Any) -> list[str]:
    if years is None:
        return []
    if isinstance(years, (int, float)):
        return [str(int(years))]
    if isinstance(years, str):
        raw = re.split(r"[,|\s]+", years.strip())
    elif isinstance(years, list):
        raw = [str(y) for y in years]
    else:
        return []
    result: list[str] = []
    for token in raw:
        match = re.search(r"(20\d{2})", token)
        if match and match.group(1) not in result:
            result.append(match.group(1))
    return result


def format_rag_excerpts(chunks: list[str], metadatas: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No matching SEC or earnings-call excerpts found for the given filters."
    lines = []
    for idx, chunk in enumerate(chunks):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        lines.append(
            f"[{idx + 1}] ticker={meta.get('ticker', 'UNKNOWN')} "
            f"year={meta.get('year', 'unknown')} "
            f"file_type={meta.get('file_type', 'unknown')} "
            f"section={meta.get('section', 'unknown')} "
            f"source={meta.get('source', 'unknown')}\n{chunk[:1200]}"
        )
    return "\n\n---\n\n".join(lines)


def run_sec_filings_rag(
    agent: Any,
    query: str,
    tickers: list[str] | None = None,
    years: list[str] | None = None,
    doc_types: list[str] | None = None,
) -> dict[str, Any]:
    normalized_tickers = _normalize_tickers(tickers)
    normalized_years = _normalize_years(years)
    normalized_doc_types = _normalize_doc_types(doc_types)

    metadata_filter: dict[str, str] = {}
    if len(normalized_years) == 1:
        metadata_filter["year"] = normalized_years[0]
    if len(normalized_tickers) == 1:
        metadata_filter["ticker"] = normalized_tickers[0]

    if metadata_filter.get("ticker") and metadata_filter.get("year"):
        decomposed = [query]
    else:
        decomposed = decompose_query(agent, query)

    rag_state: dict[str, Any] = {
        "normalized_query": query,
        "metadata_filter": metadata_filter,
        "target_tickers": normalized_tickers,
        "doc_type_priority": normalized_doc_types,
        "decomposed_queries": decomposed,
        "stats": {},
    }
    retrieve_result = multi_retrieve_node(agent, rag_state)
    rag_state.update(retrieve_result)

    candidates = rag_state.get("candidate_indices", [])
    if not candidates:
        return {
            "text": format_rag_excerpts([], []),
            "final_chunks": [],
            "final_metadatas": [],
            "stats": rag_state.get("stats", {}),
        }

    top_indices = _balanced_rerank_indices(agent, rag_state, candidates)
    final_chunks = [agent.rag.documents[idx] for idx in top_indices]
    final_metadatas = [agent.rag.doc_metadata[idx] for idx in top_indices]
    stats = rag_state.get("stats", {})
    stats.update(
        {
            "rerank_final_ticker_counts": _ticker_counts(final_metadatas),
            "rerank_final_count": len(top_indices),
            "chunks_used": len(final_chunks),
        }
    )
    return {
        "text": format_rag_excerpts(final_chunks, final_metadatas),
        "final_chunks": final_chunks,
        "final_metadatas": final_metadatas,
        "stats": stats,
    }


def run_market_price_tool(
    agent: Any,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    normalized = _normalize_tickers(tickers)
    if not normalized:
        return {"text": "No valid tickers provided. Use NVDA, AMD, or MSFT.", "price_context": ""}
    summary = fetch_price_context(agent, normalized, start_date, end_date)
    if not summary:
        return {
            "text": f"No price data for {normalized} between {start_date} and {end_date}.",
            "price_context": "",
        }
    return {"text": summary, "price_context": summary}


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _overlap_score(claim_tokens: set[str], chunk_text: str) -> float:
    if not claim_tokens:
        return 0.0
    chunk_tokens = _tokenize_for_overlap(chunk_text)
    if not chunk_tokens:
        return 0.0
    matched = claim_tokens & chunk_tokens
    return len(matched) / len(claim_tokens)


def _status_from_score(score: float) -> str:
    if score >= 0.35:
        return "supported"
    if score >= 0.15:
        return "partial"
    return "unsupported"


def run_validate_claims(
    claims: list[str],
    chunks: list[str],
    metadatas: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        return {
            "text": (
                "Aucun extrait RAG disponible. Appelez d'abord sec_filings_rag_tool "
                "puis relancez validate_claims_tool."
            ),
            "validations": [],
        }

    validations: list[dict[str, Any]] = []
    lines: list[str] = ["## Validation des affirmations", ""]

    for claim in claims:
        claim_text = str(claim).strip()
        if not claim_text:
            continue
        claim_tokens = _tokenize_for_overlap(claim_text)
        best_score = 0.0
        best_idx = -1
        for idx, chunk in enumerate(chunks):
            score = _overlap_score(claim_tokens, chunk)
            if score > best_score:
                best_score = score
                best_idx = idx

        status = _status_from_score(best_score)
        meta = metadatas[best_idx] if 0 <= best_idx < len(metadatas) else {}
        excerpt = ""
        if best_idx >= 0:
            excerpt = chunks[best_idx][:200].replace("\n", " ")

        entry = {
            "claim": claim_text,
            "status": status,
            "score": round(best_score, 3),
            "best_source_index": best_idx + 1 if best_idx >= 0 else None,
            "ticker": meta.get("ticker"),
            "year": meta.get("year"),
            "file_type": meta.get("file_type"),
            "excerpt_snippet": excerpt,
        }
        validations.append(entry)
        src = f"[{entry['best_source_index']}]" if entry["best_source_index"] else "—"
        lines.append(
            f"- **{status}** ({best_score:.0%}) — {claim_text[:120]}\n"
            f"  Source {src} {meta.get('ticker', '')} {meta.get('year', '')} "
            f"{meta.get('file_type', '')}"
        )

    if not validations:
        return {"text": "Aucune affirmation fournie dans claims.", "validations": []}

    return {"text": "\n".join(lines), "validations": validations}


def _normalize_allocations(allocations: Any) -> dict[str, float]:
    if not isinstance(allocations, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in allocations.items():
        ticker = str(key).upper().strip()
        if ticker not in TRACKED_TICKERS:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight < 0:
            continue
        result[ticker] = weight
    return result


def run_simulate_portfolio(
    allocations: dict[str, float],
    notional_usd: float = 100_000,
) -> dict[str, Any]:
    normalized = _normalize_allocations(allocations)
    if not normalized:
        return {
            "text": "Allocations invalides. Utilisez NVDA, AMD ou MSFT avec des poids >= 0.",
            "positions": [],
            "error": "invalid_tickers",
        }

    if len(normalized) > 3:
        return {
            "text": "Maximum 3 positions simulées.",
            "positions": [],
            "error": "too_many_positions",
        }

    weight_sum = sum(normalized.values())
    if abs(weight_sum - 100.0) > _WEIGHT_TOLERANCE:
        return {
            "text": f"Les poids doivent totaliser 100% (actuel: {weight_sum:.2f}%).",
            "positions": [],
            "error": "invalid_weights",
        }

    try:
        notional = float(notional_usd)
    except (TypeError, ValueError):
        notional = 100_000.0
    notional = max(1_000.0, min(notional, _MAX_NOTIONAL_USD))

    positions: list[dict[str, Any]] = []
    lines = [
        "## Simulation de portefeuille (pédagogique — aucun ordre réel)",
        f"Capital notionnel: ${notional:,.0f} USD",
        "",
    ]
    for ticker, weight_pct in sorted(normalized.items()):
        amount = notional * (weight_pct / 100.0)
        positions.append(
            {"ticker": ticker, "weight_pct": weight_pct, "notional_usd": round(amount, 2)}
        )
        lines.append(f"- **{ticker}**: {weight_pct:.1f}% → ${amount:,.2f}")

    lines.append("")
    lines.append(
        "_Simulation contrôlée: tickers limités à NVDA/AMD/MSFT, pas d'exécution sur marché._"
    )

    return {
        "text": "\n".join(lines),
        "positions": positions,
        "notional_usd": notional,
        "disclaimer": "pedagogical_simulation_no_execution",
    }


def run_export_investment_report(title: str, content: str, fmt: str = "md") -> dict[str, Any]:
    ensure_dir(REPORTS_DIR)
    safe_title = re.sub(r"[^\w\-]+", "_", title.strip())[:80] or "investment_report"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    extension = "md" if fmt.lower() != "pdf" else "pdf"
    filename = f"{safe_title}_{timestamp}.{extension}"
    path = REPORTS_DIR / filename

    if extension == "pdf":
        path = REPORTS_DIR / f"{safe_title}_{timestamp}.md"
        path.write_text(f"# {title}\n\n{content}", encoding="utf-8")
        return {
            "text": "PDF not available in this build; saved as Markdown instead.",
            "path": str(path),
            "filename": path.name,
            "format": "md",
            "title": title,
        }

    path.write_text(f"# {title}\n\n{content}", encoding="utf-8")

    return {
        "text": f"Report saved to {path}",
        "path": str(path),
        "filename": path.name,
        "format": extension,
        "title": title,
    }


def get_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "sec_filings_rag_tool",
                "description": SEC_FILINGS_RAG_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for filings/transcripts."},
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tickers: NVDA, AMD, MSFT.",
                        },
                        "years": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional filing years, e.g. ['2024'].",
                        },
                        "doc_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: 10-K, 10-Q, 8-K, EARNINGS_CALL.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "market_price_tool",
                "description": MARKET_PRICE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tickers to fetch.",
                        },
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD."},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD."},
                    },
                    "required": ["tickers", "start_date", "end_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_investment_report_tool",
                "description": EXPORT_REPORT_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Report title."},
                        "content": {"type": "string", "description": "Full report body (Markdown)."},
                        "format": {
                            "type": "string",
                            "enum": ["md", "pdf"],
                            "description": "Output format.",
                        },
                    },
                    "required": ["title", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_claims_tool",
                "description": VALIDATE_CLAIMS_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Factual claims to verify against current RAG excerpts.",
                        },
                    },
                    "required": ["claims"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_portfolio_tool",
                "description": SIMULATE_PORTFOLIO_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "allocations": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": "Ticker to weight percent, e.g. {'MSFT': 50, 'NVDA': 50}.",
                        },
                        "notional_usd": {
                            "type": "number",
                            "description": "Simulated capital in USD (default 100000, max 1000000).",
                        },
                    },
                    "required": ["allocations"],
                },
            },
        },
    ]


def execute_tool(
    agent: Any,
    name: str,
    arguments: str | dict[str, Any],
    *,
    rag_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments

    if name == "sec_filings_rag_tool":
        return run_sec_filings_rag(
            agent,
            query=str(args.get("query", "")),
            tickers=args.get("tickers"),
            years=args.get("years"),
            doc_types=args.get("doc_types"),
        )
    if name == "market_price_tool":
        return run_market_price_tool(
            agent,
            tickers=args.get("tickers") or [],
            start_date=str(args.get("start_date", "")),
            end_date=str(args.get("end_date", "")),
        )
    if name == "export_investment_report_tool":
        return run_export_investment_report(
            title=str(args.get("title", "Investment Report")),
            content=str(args.get("content", "")),
            fmt=str(args.get("format", "md")),
        )
    if name == "validate_claims_tool":
        ctx = rag_context or {}
        raw_claims = args.get("claims") or []
        if isinstance(raw_claims, str):
            raw_claims = [raw_claims]
        claims = [str(c) for c in raw_claims if str(c).strip()]
        return run_validate_claims(
            claims=claims,
            chunks=list(ctx.get("final_chunks") or []),
            metadatas=list(ctx.get("final_metadatas") or []),
        )
    if name == "simulate_portfolio_tool":
        return run_simulate_portfolio(
            allocations=args.get("allocations") or {},
            notional_usd=float(args.get("notional_usd", 100_000)),
        )
    return {"text": f"Unknown tool: {name}"}
