"""NLI-based claims validation against RAG excerpts."""

from __future__ import annotations

import json
from typing import Any

from src.tools.schemas import ValidateClaimsArgs

NLI_SYSTEM_PROMPT = (
    "You are a precise NLI (Natural Language Inference) judge for financial "
    "claims grounded in SEC filings and earnings call transcripts. You must "
    "not use any external knowledge beyond the provided excerpts. "
    "For each claim, decide if it is 'supported', 'partial', or 'unsupported' "
    "based ONLY on the excerpts."
)


def _build_nli_prompt(
    claims: list[str], chunks: list[str], metadatas: list[dict[str, Any]]
) -> str:
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
        "- partial: excerpts contain related but incomplete info.\n"
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
            entry = (
                results[i] if i < len(results) and isinstance(results[i], dict) else {}
            )
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
            meta = (
                metadatas[best_idx - 1]
                if best_idx and 0 <= best_idx - 1 < len(metadatas)
                else {}
            )
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
