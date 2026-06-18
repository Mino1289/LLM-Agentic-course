from __future__ import annotations

import re
from typing import Any


_DOLLAR_PATTERNS = (
    r"(\d[\d\s,.]*)\s*\$",
    r"\$\s*(\d[\d\s,.]*)",
    r"(\d[\d\s,.]*)\s*(?:dollars?|usd)\b",
    r"(\d[\d\s,.]*)\s*€",
    r"(\d[\d\s,.]*)\s*(?:euros?)\b",
)


def _normalize_amount(raw: str) -> float | None:
    cleaned = raw.strip().replace(" ", "")
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        left, _, right = cleaned.partition(",")
        if right.isdigit() and len(right) <= 2:
            cleaned = f"{left}.{right}"
        else:
            cleaned = cleaned.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_dollar_amount(query: str) -> float | None:
    text = (query or "").strip()
    if not text:
        return None
    for pattern in _DOLLAR_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _normalize_amount(match.group(1))
        if value is not None:
            return value
    return None


# Marqueur de la section destinée à l'utilisateur (FR/EN), suivi d'un éventuel
# parenthétique puis ":". On capture tout ce qui suit (dernière occurrence).
_RESPONSE_RE = re.compile(
    r"\b(?:R[ÉE]PONSE|RESPONSE)\b[^\n:]*:\s*",
    flags=re.IGNORECASE,
)


def extract_user_response(text: str) -> str:
    """Extraire la portion lisible (après le marqueur RESPONSE/RÉPONSE).

    Le PM produit un format structuré (PLAN/DECISION/SYNTHESIS + RESPONSE) dont
    seule la section RESPONSE doit être montrée à l'utilisateur. Si aucun
    marqueur n'est trouvé, on retourne le texte complet (compat. ascendante).
    """
    if not text:
        return text
    last = None
    for match in _RESPONSE_RE.finditer(text):
        last = match
    if last is None:
        return text.strip()
    return text[last.end() :].strip()


def parse_pm_response(text: str) -> dict[str, Any]:
    decision: dict[str, Any] = {"response": text}
    lines = text.split("\n")
    for line in lines:
        raw = line.strip()
        lower = raw.lower()
        for prefix in ("- ", "* ", "• "):
            if lower.startswith(prefix):
                lower = lower[len(prefix) :]
                raw = raw[len(prefix) :]
                break
        if lower.startswith("ticker:"):
            decision["ticker"] = raw.split(":", 1)[1].strip()
        elif lower.startswith("side:"):
            decision["side"] = raw.split(":", 1)[1].strip().lower()
        elif (
            lower.startswith("quantity")
            or lower.startswith("qty")
            or lower.startswith("amount")
        ):
            val = raw.split(":", 1)[1].strip() if ":" in raw else ""
            decision["qty"] = val
        elif lower.startswith("order type"):
            decision["order_type"] = (
                raw.split(":", 1)[1].strip().lower() if ":" in raw else "market"
            )
        elif lower.startswith("limit price"):
            val = raw.split(":", 1)[1].strip() if ":" in raw else ""
            decision["limit_price"] = val
    return decision


def enrich_pm_decision(state: dict[str, Any]) -> dict[str, Any]:
    decision = dict(state.get("pm_decision") or {})
    response_text = str(decision.get("response") or state.get("answer") or "")
    if response_text:
        parsed = parse_pm_response(response_text)
        for key in ("ticker", "side", "qty", "order_type", "limit_price"):
            if not decision.get(key) and parsed.get(key):
                decision[key] = parsed[key]
    return decision
