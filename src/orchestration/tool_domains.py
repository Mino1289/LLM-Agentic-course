from __future__ import annotations

from typing import Literal

ToolDomain = Literal["fundamental", "news", "quant", "portfolio", "trade", "report"]
IntentRoute = Literal["simple", "complex"]

TOOL_DOMAIN_KEYWORDS: dict[ToolDomain, tuple[str, ...]] = {
    "fundamental": (
        "sec", "10-k", "10-q", "8-k", "20-f", "6-k", "filing", "filings",
        "risque", "risk", "md&a", "document", "section", "earnings",
        "rapport annuel", "annual report", "10k", "10q",
    ),
    "news": (
        "news", "actualité", "actualite", "actualités", "headline", "headlines",
    ),
    "quant": (
        "prix", "price", "cours", "cotati", "cotation", "performance", "perf",
        "rendement", "volatilité", "volatility", "historique", "history",
        "compar", "compare", "combien coûte", "combien coute", "valeur boursière",
        "chart", "graphique", "6 mois", "6 month", "ytd",
    ),
    "portfolio": (
        "portefeuille", "portfolio", "compte", "account", "positions",
        "mon portefeuille", "mon compte", "mes positions", "mes actions",
        "buying power", "equity", "pnl", "solde", "balance",
        "activité", "activity", "transaction",
    ),
    "trade": (
        "achète", "acheter", "achat", "acheté", "achete", "action achet",
        "buy", "investi", "investis", "investir", "placement",
        "vends", "vendre", "vend", "sell", "vente", "trade", "order", "ordre",
        "rebalance", "rebalancer", "alloue", "allouer", "allocation",
        "place un ordre", "soumet un ordre", "exécute", "exécuter", "execute",
        "close position", "liquid", "couvre", "couverture", "hedge",
        "utilise mon", "prendre position",
    ),
    "report": (
        "exporter", "export", "sauvegarder", "save as", "save report",
        "generate report", "générer un rapport", "generer un rapport",
        "télécharger le rapport", "telecharger le rapport", "download report",
        ".pdf", "export_investment",
    ),
}


def detect_tool_domains(query: str) -> frozenset[ToolDomain]:
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return frozenset()

    matched: set[ToolDomain] = set()
    for domain, keywords in TOOL_DOMAIN_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched.add(domain)
    return frozenset(matched)


def resolve_route_from_domains(
    domains: frozenset[ToolDomain],
) -> tuple[IntentRoute, str] | None:
    """Return (route, intent_reason) or None if LLM classification is needed."""
    if len(domains) >= 2:
        return "complex", "multi_tool"
    if "trade" in domains:
        return "complex", "action_keyword"
    if len(domains) == 1:
        return "simple", "single_tool"
    return None
