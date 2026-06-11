from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from html import escape
from typing import Any

from pydantic import BaseModel
from toon_format import encode as toon_encode

from rag.config import TRACKED_TICKERS
from rag.nodes.decompose_node import decompose_query
from rag.nodes.rerank_node import _balanced_rerank_indices, _ticker_counts
from rag.nodes.retrieval_node import multi_retrieve_node
from rag.nodes.tool_nodes import fetch_price_context
from rag.paths import REPORTS_DIR, ensure_dir
from rag.tool_schemas import (
    AccountActivityArgs,
    ClosePositionArgs,
    ExportReportArgs,
    GetNewsArgs,
    MarketPriceArgs,
    PlaceTradeArgs,
    PortfolioHistoryArgs,
    PortfolioInfoArgs,
    SecFilingsRAGArgs,
    ValidateClaimsArgs,
    ValidateClaimsLLMArgs,
)

ALLOWED_DOC_TYPES = ["10-K", "10-Q", "8-K", "20-F", "6-K", "EARNINGS_CALL"]


def _tracked_tickers_text() -> str:
    return ", ".join(TRACKED_TICKERS)

SEC_FILINGS_RAG_DESCRIPTION = (
    "Search SEC filings and earnings call transcripts in ChromaDB. "
    "Filter by document type: 10-K (annual), 10-Q (quarterly), 8-K (events), "
    "20-F (foreign annual), 6-K (foreign interim), EARNINGS_CALL (conference call transcripts). "
    "Examples: 'MSFT 10-K risk factors 2024', 'ASML 20-F risk factors 2024'."
)

MARKET_PRICE_DESCRIPTION = (
    f"Fetch stock price performance for tracked tickers ({_tracked_tickers_text()}) "
    "between start_date and end_date (YYYY-MM-DD)."
)

EXPORT_REPORT_DESCRIPTION = (
    "Save an investment report to the reports/ folder. "
    "Supported formats: md (default) and pdf. Returns the file path."
)

VALIDATE_CLAIMS_DESCRIPTION = (
    "Check whether factual claims are supported by RAG excerpts already retrieved. "
    "Call after sec_filings_rag_tool. Returns supported/partial/unsupported per claim with source refs."
)

PORTFOLIO_INFO_DESCRIPTION = (
    "View your Alpaca paper trading account: cash balance, equity, buying power, "
    f"unrealized P&L, and all open positions. Restricted to {_tracked_tickers_text()}."
)

PLACE_TRADE_DESCRIPTION = (
    "Submit a real paper trade on Alpaca. "
    f"Support tickers: {_tracked_tickers_text()}. "
    "Supports market, limit, stop, and stop-limit orders. "
    f"Max notional per order: $10,000."
)

CLOSE_POSITION_DESCRIPTION = (
    "Close a specific position by ticker or liquidate all positions "
    "on your Alpaca paper trading account."
)

GET_NEWS_DESCRIPTION = (
    "Fetch latest news articles for one or more tickers from Alpaca News API. "
    f"Supports {_tracked_tickers_text()} and other US-traded symbols. "
    "Returns headlines, source, summary, date, and URL."
)

PORTFOLIO_HISTORY_DESCRIPTION = (
    "Get equity and P&L history for your Alpaca paper account over a period. "
    "Defaults to 1 month. Supports 1D, 1W, 1M, 1A periods with configurable timeframe."
)

ACCOUNT_ACTIVITY_DESCRIPTION = (
    "Retrieve account activity (fills, dividends, deposits, withdrawals, fees) "
    "from your Alpaca paper account. Filter by activity type, date range, and sort direction."
)

_LOGGER = logging.getLogger("rag.tools")


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
    """Serialize RAG excerpts as a TOON tabular array for LLM context.

    TOON format (Token-Oriented Object Notation) saves 30-60% tokens vs
    the legacy "key=value" text format. Output structure:
        excerpts[N]{i,ticker,year,file_type,section,source,text}:
          1,NVDA,"2024",10-K,Item_1A,nvda-10-k_2024.htm,"<truncated chunk>"
          ...

    Chunks are still truncated to 1200 chars (legacy constraint) to keep
    individual rows manageable. Empty input returns a human-readable
    fallback string (not a TOON "[0]:" header) so the LLM gets a clear
    "no results" signal.
    """
    if not chunks:
        return "No matching SEC or earnings-call excerpts found for the given filters."
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        rows.append(
            {
                "i": idx + 1,
                "ticker": str(meta.get("ticker", "UNKNOWN")),
                "year": str(meta.get("year", "unknown")),
                "file_type": str(meta.get("file_type", "unknown")),
                "section": str(meta.get("section", "unknown")),
                "source": str(meta.get("source", "unknown")),
                "text": chunk[:1200],
            }
        )
    return toon_encode({"excerpts": rows})


async def run_sec_filings_rag(
    args: SecFilingsRAGArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    """Run the SEC RAG pipeline for the user query.

    This is a native async function so it can share the agent's event loop
    with the LLM calls in ``agent_node`` and ``ainvoke_with_tools_stream``.
    The previous sync + ``asyncio.run`` pattern created a second event loop
    in a worker thread, which collided with the ``AsyncOpenAI`` httpx
    connection pool — manifesting as a 2-minute hang in the Streamlit UI.
    """
    pipeline_t0 = time.perf_counter()
    normalized_tickers = _normalize_tickers(args.tickers)
    normalized_years = _normalize_years(args.years)
    normalized_doc_types = _normalize_doc_types(args.doc_types)
    query = args.query

    metadata_filter: dict[str, str] = {}
    if len(normalized_years) == 1:
        metadata_filter["year"] = normalized_years[0]
    if len(normalized_tickers) == 1:
        metadata_filter["ticker"] = normalized_tickers[0]

    decomposed: list[str]
    decompose_skipped_reason: str | None = None
    if len(normalized_tickers) == 1:
        # Single-ticker queries are precise enough to skip the LLM-based
        # decomposition (saves ~3-7s per call). The year filter, when set,
        # still restricts retrieval via ``metadata_filter``.
        decomposed = [query]
        decompose_skipped_reason = "single_ticker"
        _LOGGER.info(
            "rag.tools: decompose skipped (single ticker=%s, years=%s)",
            normalized_tickers[0],
            normalized_years or "any",
        )
    else:
        decompose_t0 = time.perf_counter()
        decomposed = await decompose_query(agent, query)
        _LOGGER.info(
            "rag.tools: decompose took %.2fs (count=%d, tickers=%s, years=%s)",
            time.perf_counter() - decompose_t0,
            len(decomposed),
            normalized_tickers or "any",
            normalized_years or "any",
        )

    rag_state: dict[str, Any] = {
        "normalized_query": query,
        "metadata_filter": metadata_filter,
        "target_tickers": normalized_tickers,
        "doc_type_priority": normalized_doc_types,
        "decomposed_queries": decomposed,
        "stats": {"decomposed_count": len(decomposed)},
    }
    if decompose_skipped_reason:
        rag_state["stats"]["decompose_skipped_reason"] = decompose_skipped_reason

    retrieve_t0 = time.perf_counter()
    retrieve_result = await multi_retrieve_node(agent, rag_state)
    rag_state.update(retrieve_result)
    _LOGGER.info(
        "rag.tools: retrieve took %.2fs (candidates=%d)",
        time.perf_counter() - retrieve_t0,
        len(rag_state.get("candidate_indices", [])),
    )

    candidates = rag_state.get("candidate_indices", [])
    if not candidates:
        _LOGGER.info(
            "rag.tools: pipeline total %.2fs (no candidates, skipped rerank)",
            time.perf_counter() - pipeline_t0,
        )
        return {
            "text": format_rag_excerpts([], []),
            "final_chunks": [],
            "final_metadatas": [],
            "stats": rag_state.get("stats", {}),
        }

    rerank_t0 = time.perf_counter()
    top_indices = await _balanced_rerank_indices(agent, rag_state, candidates)
    _LOGGER.info(
        "rag.tools: rerank took %.2fs (selected=%d/%d)",
        time.perf_counter() - rerank_t0,
        len(top_indices),
        len(candidates),
    )
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
    _LOGGER.info(
        "rag.tools: pipeline total %.2fs (decompose=%d, candidates=%d, final=%d)",
        time.perf_counter() - pipeline_t0,
        len(decomposed),
        len(candidates),
        len(final_chunks),
    )
    return {
        "text": format_rag_excerpts(final_chunks, final_metadatas),
        "final_chunks": final_chunks,
        "final_metadatas": final_metadatas,
        "stats": stats,
    }


def run_market_price_tool(
    args: MarketPriceArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    normalized = _normalize_tickers(args.tickers)
    start_date = args.start_date
    end_date = args.end_date
    if not normalized:
        return {"text": f"No valid tickers provided. Use {_tracked_tickers_text()}.", "price_context": ""}
    summary = fetch_price_context(agent, normalized, start_date, end_date)
    if not summary:
        return {
            "text": f"No price data for {normalized} between {start_date} and {end_date}.",
            "price_context": "",
        }
    return {"text": summary, "price_context": summary}


NLI_SYSTEM_PROMPT = (
    "You are a precise NLI (Natural Language Inference) judge for financial "
    "claims grounded in SEC filings and earnings call transcripts. You must "
    "not use any external knowledge beyond the provided excerpts. "
    "For each claim, decide if it is 'supported', 'partial', or 'unsupported' "
    "based ONLY on the excerpts."
)


def _build_nli_prompt(claims: list[str], chunks: list[str], metadatas: list[dict[str, Any]]) -> str:
    exhibit_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
        header = (
            f"[{idx}] ticker={meta.get('ticker', 'UNKNOWN')} "
            f"year={meta.get('year', 'unknown')} "
            f"file_type={meta.get('file_type', 'unknown')} "
            f"section={meta.get('section', 'unknown')}"
        )
        excerpt = chunk[:1200].replace("\n", " ")
        exhibit_blocks.append(f"{header}\n{excerpt}")
    exhibits = "\n\n".join(exhibit_blocks) if exhibit_blocks else "(no excerpts)"

    claim_lines = "\n".join(f"- {claim}" for claim in claims)
    return (
        "OUTPUT — STRICT JSON only, with no surrounding text. Schema:\n"
        '{"results": ['
        '{"claim": "<verbatim claim>", "status": "supported|partial|unsupported",'
        ' "best_source_index": <1-based index or null>, "reasoning": "<one short sentence>"}'
        "]}\n\n"
        "Definitions:\n"
        "- supported: an excerpt explicitly confirms the claim.\n"
        "- partial: excerpts contain related but incomplete info (same topic, different period, partial figures).\n"
        "- unsupported: no relevant excerpt, or an excerpt contradicts the claim.\n\n"
        f"EXCERPTS ({len(chunks)}):\n{exhibits}\n\n"
        f"CLAIMS ({len(claims)}):\n{claim_lines}"
    )


def _render_validation_text(validations: list[dict[str, Any]]) -> str:
    if not validations:
        return "Aucune affirmation fournie dans claims."
    score_for_status = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    lines = ["## Validation des affirmations (NLI)", ""]
    for v in validations:
        status = v["status"]
        score = score_for_status.get(status, 0.0)
        src_idx = v.get("best_source_index")
        src = f"[{src_idx}]" if src_idx else "—"
        reason = v.get("reasoning", "")
        lines.append(
            f"- **{status}** ({score:.0%}) — {v['claim'][:120]}\n"
            f"  Source {src} {v.get('ticker', '')} {v.get('year', '')} "
            f"{v.get('file_type', '')}\n"
            f"  Raisonnement NLI: {reason}"
        )
    return "\n".join(lines)


def run_validate_claims(
    args: ValidateClaimsArgs,
    *,
    agent: Any,
) -> dict[str, Any]:
    claims = args.claims
    chunks = args.chunks
    metadatas = args.metadatas
    if not chunks:
        return {
            "text": (
                "Aucun extrait RAG disponible. Appelez d'abord sec_filings_rag_tool "
                "puis relancez validate_claims_tool."
            ),
            "validations": [],
            "stats": {"validate_nli_used": False, "validate_nli_claims": len(claims)},
        }

    cleaned_claims = [str(c).strip() for c in (claims or []) if str(c).strip()]
    if not cleaned_claims:
        return {
            "text": "Aucune affirmation fournie dans claims.",
            "validations": [],
            "stats": {"validate_nli_used": False, "validate_nli_claims": 0},
        }

    prompt = _build_nli_prompt(cleaned_claims, chunks, metadatas)
    max_tokens = min(800, 80 * len(cleaned_claims))
    raw = ""
    fallback_reason = "nli_provider_error"
    try:
        raw = agent.rag.provider.generate(
            prompt,
            system_prompt=NLI_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            parsed = json.loads(raw or "")
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list):
            fallback_reason = "nli_parse_error"
            results = None
    except Exception:
        results = None
        fallback_reason = "nli_provider_error"

    validations: list[dict[str, Any]] = []
    if results:
        for i, claim in enumerate(cleaned_claims):
            entry = results[i] if i < len(results) and isinstance(results[i], dict) else {}
            status = str(entry.get("status", "")).strip().lower()
            if status not in {"supported", "partial", "unsupported"}:
                status = "unsupported"
            raw_idx = entry.get("best_source_index")
            try:
                best_idx = int(raw_idx) if raw_idx is not None else None
            except (TypeError, ValueError):
                best_idx = None
            if best_idx is not None and not (1 <= best_idx <= len(chunks)):
                best_idx = None
            meta = metadatas[best_idx - 1] if best_idx and 0 <= best_idx - 1 < len(metadatas) else {}
            excerpt_snippet = ""
            if best_idx:
                excerpt_snippet = chunks[best_idx - 1][:200].replace("\n", " ")
            reasoning = str(entry.get("reasoning", "")).strip() or "nli_judge"
            validations.append(
                {
                    "claim": claim,
                    "status": status,
                    "best_source_index": best_idx,
                    "ticker": meta.get("ticker"),
                    "year": meta.get("year"),
                    "file_type": meta.get("file_type"),
                    "excerpt_snippet": excerpt_snippet,
                    "reasoning": reasoning,
                    "nli_used": True,
                }
            )
    else:
        for claim in cleaned_claims:
            validations.append(
                {
                    "claim": claim,
                    "status": "unsupported",
                    "best_source_index": None,
                    "ticker": None,
                    "year": None,
                    "file_type": None,
                    "excerpt_snippet": "",
                    "reasoning": fallback_reason,
                    "nli_used": True,
                }
            )

    return {
        "text": _render_validation_text(validations),
        "validations": validations,
        "stats": {
            "validate_nli_used": True,
            "validate_nli_claims": len(cleaned_claims),
        },
    }

    return {"text": "\n".join(lines), "validations": validations}


def run_portfolio_info(
    args: PortfolioInfoArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_portfolio_info as _run

    return _run(args)


def run_place_trade(
    args: PlaceTradeArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_place_trade as _run

    return _run(args)


def run_close_position(
    args: ClosePositionArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_close_position as _run

    return _run(args)


def run_get_news(
    args: GetNewsArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_get_news as _run

    return _run(args)


def run_portfolio_history(
    args: PortfolioHistoryArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_portfolio_history as _run

    return _run(args)


def run_account_activity(
    args: AccountActivityArgs,
) -> dict[str, Any]:
    from rag.alpaca_tools import run_account_activity as _run

    return _run(args)


def _markdown_line_to_pdf_flowable(line: str, styles: Any) -> Any:
    from reportlab.platypus import Paragraph, Spacer

    stripped = line.strip()
    if not stripped:
        return Spacer(1, 8)

    if stripped.startswith("### "):
        return Paragraph(escape(stripped[4:]), styles["Heading3"])
    if stripped.startswith("## "):
        return Paragraph(escape(stripped[3:]), styles["Heading2"])
    if stripped.startswith("# "):
        return Paragraph(escape(stripped[2:]), styles["Heading1"])
    if stripped.startswith("- "):
        return Paragraph(f"• {escape(stripped[2:])}", styles["BodyText"])

    # Keep basic Markdown emphasis readable without trying to implement a full parser.
    text = escape(stripped)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return Paragraph(text, styles["BodyText"])


def _write_pdf_report(path: Any, title: str, content: str) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install dependencies with: "
            ".venv/bin/pip install -r requirements.txt"
        ) from exc

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
        title=title,
    )
    story = [_markdown_line_to_pdf_flowable(f"# {title}", styles), Spacer(1, 12)]
    for line in content.splitlines():
        story.append(_markdown_line_to_pdf_flowable(line, styles))
    doc.build(story)


def run_export_investment_report(args: ExportReportArgs) -> dict[str, Any]:
    ensure_dir(REPORTS_DIR)
    title = args.title
    content = args.content
    fmt = args.format
    safe_title = re.sub(r"[^\w\-]+", "_", title.strip())[:80] or "investment_report"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    extension = "md" if fmt.lower() != "pdf" else "pdf"
    filename = f"{safe_title}_{timestamp}.{extension}"
    path = REPORTS_DIR / filename

    if extension == "pdf":
        _write_pdf_report(path, title, content)
        return {
            "text": f"PDF report saved to {path}",
            "path": str(path),
            "filename": path.name,
            "format": "pdf",
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
                            "description": f"Optional tickers: {_tracked_tickers_text()}.",
                        },
                        "years": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional filing years, e.g. ['2024'].",
                        },
                        "doc_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: 10-K, 10-Q, 8-K, 20-F, 6-K, EARNINGS_CALL.",
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
                "parameters": ValidateClaimsLLMArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_info_tool",
                "description": PORTFOLIO_INFO_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "place_trade_tool",
                "description": PLACE_TRADE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": f"Ticker: {_tracked_tickers_text()}.",
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Buy or sell.",
                        },
                        "qty": {
                            "type": "number",
                            "description": "Number of shares (fractional allowed).",
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop", "stop_limit"],
                            "description": "Default: market.",
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "Limit price (required for limit / stop_limit).",
                        },
                        "stop_price": {
                            "type": "number",
                            "description": "Stop price (required for stop / stop_limit).",
                        },
                    },
                    "required": ["ticker", "side", "qty"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_position_tool",
                "description": CLOSE_POSITION_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker to close (leave empty if all=true).",
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Set to true to liquidate all positions.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_news_tool",
                "description": GET_NEWS_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tickers to get news for.",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date YYYY-MM-DD (optional).",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date YYYY-MM-DD (optional).",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max articles (default 10, max 50).",
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Include full article content.",
                        },
                    },
                    "required": ["symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_history_tool",
                "description": PORTFOLIO_HISTORY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "description": "Duration: 1D, 1W, 1M (default), 1A.",
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Resolution: 1Min, 5Min, 15Min, 1H, 1D.",
                        },
                        "extended_hours": {
                            "type": "boolean",
                            "description": "Include extended hours.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "account_activity_tool",
                "description": ACCOUNT_ACTIVITY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "activity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter: FILL, DIV, CSD, CSW, INT, FEE, etc.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date filter YYYY-MM-DD.",
                        },
                        "after": {
                            "type": "string",
                            "description": "After this date YYYY-MM-DD.",
                        },
                        "until": {
                            "type": "string",
                            "description": "Before this date YYYY-MM-DD.",
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Max entries (default 20, max 100).",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "description": "Sort direction.",
                        },
                    },
                },
            },
        },
    ]


def execute_tool(
    name: str,
    args: BaseModel,
    *,
    agent: Any,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name == "sec_filings_rag_tool":
        return run_sec_filings_rag(args, agent=agent)
    if name == "market_price_tool":
        return run_market_price_tool(args, agent=agent)
    if name == "export_investment_report_tool":
        return run_export_investment_report(args)
    if name == "validate_claims_tool":
        # Resolve injected args from state (chunks + metadatas) if needed.
        if not isinstance(args, ValidateClaimsArgs):
            full = ValidateClaimsArgs(
                claims=args.claims,
                chunks=list((state or {}).get("final_chunks") or []),
                metadatas=list((state or {}).get("final_metadatas") or []),
            )
        else:
            full = args
        return run_validate_claims(full, agent=agent)
    if name == "portfolio_info_tool":
        return run_portfolio_info(args)
    if name == "place_trade_tool":
        return run_place_trade(args)
    if name == "close_position_tool":
        return run_close_position(args)
    if name == "get_news_tool":
        return run_get_news(args)
    if name == "portfolio_history_tool":
        return run_portfolio_history(args)
    if name == "account_activity_tool":
        return run_account_activity(args)
    return {"text": f"Unknown tool: {name}"}
