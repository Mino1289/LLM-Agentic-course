from __future__ import annotations

from typing import Literal

ToolDomain = Literal["fundamental", "news", "quant", "portfolio", "trade", "report"]
IntentRoute = Literal["simple", "complex"]

TOOL_DOMAIN_KEYWORDS: dict[ToolDomain, tuple[str, ...]] = {
    "fundamental": (
        "sec",
        "10-k",
        "10-q",
        "8-k",
        "20-f",
        "6-k",
        "filing",
        "filings",
        "risque",
        "risk",
        "md&a",
        "document",
        "section",
        "earnings",
        "rapport annuel",
        "annual report",
        "10k",
        "10q",
    ),
    "news": (
        "news",
        "actualité",
        "actualite",
        "actualités",
        "headline",
        "headlines",
    ),
    "quant": (
        "prix",
        "price",
        "cours",
        "cotati",
        "cotation",
        "performance",
        "perf",
        "rendement",
        "volatilité",
        "volatility",
        "historique",
        "history",
        "compar",
        "compare",
        "combien coûte",
        "combien coute",
        "valeur boursière",
        "chart",
        "graphique",
        "6 mois",
        "6 month",
        "ytd",
    ),
    "portfolio": (
        "portefeuille",
        "portfolio",
        "compte",
        "account",
        "positions",
        "mon portefeuille",
        "mon compte",
        "mes positions",
        "mes actions",
        "buying power",
        "equity",
        "pnl",
        "solde",
        "balance",
        "activité",
        "activity",
        "transaction",
    ),
    # IMPORTANT : le domaine "trade" doit rester HAUTE PRÉCISION — il déclenche
    # une proposition d'ordre (PM -> compliance -> approbation humaine). On n'y
    # met QUE des verbes/commandes d'exécution explicites. Les mots topiques
    # mous (placement, allocation, "$"/dollars, mets/prends, "investissement"...)
    # sont volontairement EXCLUS : ils ne sont pas un ordre. Les cas ambigus
    # ("mets 5000$ sur NVDA") tombent à vide et sont tranchés par le classifieur
    # LLM, qui décide alors is_trade. Voir intent_router_node.
    "trade": (
        "achète",
        "acheter",
        "achat",
        "acheté",
        "achete",
        "action achet",
        "buy",
        # Forme impérative uniquement ("investis 5000$", "investir dans X").
        # PAS le nom "investissement"/"investisseur" (sujet d'analyse) — d'où
        # "investis " avec espace, et non "investi"/"investis".
        "investis ",
        "investir",
        "investissez",
        "vends",
        "vendez",
        "vendre ",  # espace : éviter "vendredi"
        "vend ",  # espace : éviter "vendredi"
        "sell",
        "vente ",  # espace : éviter "inventer"
        "trade",
        "order",
        "ordre",
        "rebalance",
        "rebalancer",
        "alloue",
        "allouer",
        "place un ordre",
        "passe un ordre",
        "passer un ordre",
        "soumet un ordre",
        "soumettre un ordre",
        "exécute",
        "exécuter",
        "execute",
        "close position",
        "ferme la position",
        "ferme ma position",
        "ferme mes positions",
        "liquide",  # PAS "liquid" qui matche "liquidité"
        "liquider",
        "prends position",
        "prendre position",
    ),
    "report": (
        "exporter",
        "export",
        "sauvegarder",
        "save as",
        "save report",
        "generate report",
        "générer un rapport",
        "generer un rapport",
        "télécharger le rapport",
        "telecharger le rapport",
        "download report",
        ".pdf",
        "export_investment",
    ),
}


# Tournures qui signalent une demande de CONSEIL / recommandation / comparaison
# plutôt qu'un ordre à exécuter. Quand l'une est présente, on retire le domaine
# "trade" : l'utilisateur veut une recommandation argumentée, pas une proposition
# d'ordre en attente d'approbation. Ex. "Recommande-moi entre NVDA et AMD pour un
# investissement long terme" ou "Devrais-je acheter NVDA ?" => analyse, pas trade.
ADVISORY_MARKERS: tuple[str, ...] = (
    "recommand",  # recommande, recommandation, recommend
    "conseil",  # conseille, conseil, conseiller
    "que penses",
    "qu'en penses",
    "qu en penses",
    "ton avis",
    "ton opinion",
    "devrais-je",
    "devrais je",
    "dois-je",
    "dois je",
    "faut-il",
    "faut il",
    "vaut-il mieux",
    "vaut il mieux",
    "vaut mieux",
    "bonne idée",
    "bonne idee",
    "should i",
    "is it worth",
    "good idea",
    "which is better",
)


def _is_advisory(query_lower: str) -> bool:
    return any(marker in query_lower for marker in ADVISORY_MARKERS)


def detect_tool_domains(query: str) -> frozenset[ToolDomain]:
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return frozenset()

    matched: set[ToolDomain] = set()
    for domain, keywords in TOOL_DOMAIN_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched.add(domain)
    # Demande de conseil/recommandation/comparaison : on n'exécute pas d'ordre.
    if "trade" in matched and _is_advisory(query_lower):
        matched.discard("trade")
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
