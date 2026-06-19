"""News tool — fetch financial news via Alpaca News API."""

from __future__ import annotations

from typing import Any

from src.tools.schemas import GetNewsArgs

_NOT_CONFIGURED_TEXT = (
    "Outils Alpaca non disponibles — clés API manquantes. "
    "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
)


def run_get_news(args: GetNewsArgs) -> dict[str, Any]:
    from src.alpaca.client import get_news_client, format_alpaca_error

    client = get_news_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}
    try:
        from alpaca.data.requests import NewsRequest

        request = NewsRequest(
            symbols=",".join(args.symbols),
            start=args.start,
            end=args.end,
            limit=args.limit,
            include_content=args.include_content,
        )
        news_set = client.get_news(request)
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    articles = []
    lines = [f"## News: {', '.join(args.symbols)}", ""]
    all_news = news_set.data.get("news", []) if hasattr(news_set, "data") else []
    if not all_news and hasattr(news_set, "data"):
        for val in news_set.data.values():
            if isinstance(val, list):
                all_news.extend(val)
                break
    for article in all_news[: args.limit]:
        headline = getattr(article, "headline", "")
        source = getattr(article, "source", "")
        summary = getattr(article, "summary", "")
        url = getattr(article, "url", "")
        created = getattr(article, "created_at", None)
        symbols = getattr(article, "symbols", [])
        articles.append(
            {
                "headline": headline,
                "source": source,
                "summary": summary,
                "url": url,
                "created_at": str(created) if created else "",
                "symbols": symbols,
            }
        )
        date_str = (
            created.strftime("%Y-%m-%d")
            if hasattr(created, "strftime")
            else str(created or "")
        )
        lines.append(f"### {headline}")
        lines.append(f"*{source}* — {date_str}")
        lines.append(f"{summary[:300]}{'...' if len(summary) > 300 else ''}")
        lines.append(f"[Lire plus]({url})" if url else "")
        lines.append("")
    if not articles:
        lines.append("*Aucun article trouvé.*")
    return {
        "text": "\n".join(lines),
        "articles": articles,
        "article_count": len(articles),
    }
