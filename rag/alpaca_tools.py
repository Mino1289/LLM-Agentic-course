from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from rag.alpaca_client import (
    AlpacaNotConfiguredError,
    fetch_account_activities,
    format_alpaca_error,
    get_alpaca_client,
    get_news_client,
)
from rag.config import TRACKED_TICKERS
from rag.tool_schemas import (
    AccountActivityArgs,
    ClosePositionArgs,
    GetNewsArgs,
    PlaceTradeArgs,
    PortfolioHistoryArgs,
    PortfolioInfoArgs,
)

_TRACKED = set(TRACKED_TICKERS)

_LOGGER = logging.getLogger("rag.alpaca_tools")

_MAX_NOTIONAL_PER_ORDER = 10_000.0


def _fmt_usd(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value:,.0f}"
    if abs(value) >= 1_000:
        return f"${value:,.2f}"
    return f"${value:.2f}"


def run_portfolio_info(args: PortfolioInfoArgs) -> dict[str, Any]:
    client = get_alpaca_client()
    if client is None:
        return {
            "text": (
                "Outils Alpaca non disponibles — clés API manquantes. "
                "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
            ),
            "error": "alpaca_not_configured",
        }

    try:
        account = client.get_account()
        positions = client.get_all_positions()
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    lines = [
        "## Portefeuille Alpaca (Paper Trading)",
        f"Solde: {_fmt_usd(float(account.cash))}",
        f"Equity: {_fmt_usd(float(account.equity))}",
        f"Buying Power: {_fmt_usd(float(account.buying_power))}",
        f"P&L non réalisé (intraday): {_fmt_usd(float(account.equity) - float(account.last_equity or account.equity))}",
        "",
    ]

    pos_list: list[dict[str, Any]] = []
    if positions:
        lines.append(f"### Positions ({len(positions)})")
        lines.append("")
        for pos in positions:
            sym = pos.symbol
            qty = float(pos.qty)
            mkt_val = float(pos.market_value)
            cost_basis = float(pos.cost_basis)
            upnl = float(pos.unrealized_pl)
            upnl_pct = float(pos.unrealized_plpc) * 100
            pos_list.append(
                {
                    "ticker": sym,
                    "qty": qty,
                    "market_value": round(mkt_val, 2),
                    "cost_basis": round(cost_basis, 2),
                    "unrealized_pl": round(upnl, 2),
                    "unrealized_pl_pct": round(upnl_pct, 2),
                }
            )
            lines.append(
                f"- **{sym}**: {qty:.4f} actions | "
                f"Val. marché: {_fmt_usd(mkt_val)} | "
                f"P&L: {_fmt_usd(upnl)} ({upnl_pct:+.2f}%)"
            )
    else:
        lines.append("*Aucune position ouverte.*")

    return {
        "text": "\n".join(lines),
        "account": {
            "cash": round(float(account.cash), 2),
            "equity": round(float(account.equity), 2),
            "buying_power": round(float(account.buying_power), 2),
            "unrealized_pl": round(float(account.equity) - float(account.last_equity or account.equity), 2),
        },
        "positions": pos_list,
        "position_count": len(pos_list),
    }


def run_place_trade(args: PlaceTradeArgs) -> dict[str, Any]:
    ticker = args.ticker.upper().strip()
    if ticker not in _TRACKED:
        return {
            "text": (
                f"Ticker {ticker} non autorisé. "
                f"Tickers supportés: {', '.join(sorted(_TRACKED))}."
            ),
            "error": "invalid_ticker",
        }

    side = args.side.lower()
    if side not in ("buy", "sell"):
        return {"text": "Le côté doit être 'buy' ou 'sell'.", "error": "invalid_side"}

    qty = args.qty
    order_type = (args.order_type or "market").lower()
    if order_type not in ("market", "limit", "stop_limit", "stop"):
        return {"text": "Type d'ordre invalide: market, limit, stop ou stop_limit.", "error": "invalid_order_type"}

    notional = qty * 200  # rough estimate for validation
    if notional > _MAX_NOTIONAL_PER_ORDER:
        lines = [
            f"## Ordre non exécuté — montant estimé trop élevé",
            f"Ticker: {ticker}",
            f"Quantité: {qty}",
            f"Estimation: {_fmt_usd(notional)} (max: {_fmt_usd(_MAX_NOTIONAL_PER_ORDER)})",
            "",
            f"Réduis la quantité ou contacte l'administrateur.",
        ]
        return {"text": "\n".join(lines), "error": "notional_exceeded"}

    client = get_alpaca_client()
    if client is None:
        return {
            "text": (
                "Outils Alpaca non disponibles — clés API manquantes. "
                "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
            ),
            "error": "alpaca_not_configured",
        }

    try:
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        time_in_force = TimeInForce.DAY

        if order_type == "market":
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        elif order_type == "limit":
            if args.limit_price is None:
                return {"text": "Prix limite requis pour un ordre limit.", "error": "missing_limit_price"}
            order_data = LimitOrderRequest(
                symbol=ticker,
                limit_price=args.limit_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        elif order_type == "stop":
            if args.stop_price is None:
                return {"text": "Prix stop requis pour un ordre stop.", "error": "missing_stop_price"}
            order_data = StopOrderRequest(
                symbol=ticker,
                stop_price=args.stop_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        elif order_type == "stop_limit":
            if args.stop_price is None or args.limit_price is None:
                return {"text": "Prix stop et limite requis pour stop_limit.", "error": "missing_stop_or_limit"}
            order_data = StopLimitOrderRequest(
                symbol=ticker,
                stop_price=args.stop_price,
                limit_price=args.limit_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        else:
            return {"text": f"Type d'ordre non supporté: {order_type}", "error": "unsupported_order_type"}

        order = client.submit_order(order_data)
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    return {
        "text": (
            f"## Ordre soumis (Paper Trading)\n"
            f"- **{side.upper()}** {qty} x **{ticker}**\n"
            f"- Type: {order_type}\n"
            f"- ID: {order.id}\n"
            f"- Statut: {order.status}\n"
            f"- Soumis à: {datetime.now(UTC).strftime('%H:%M:%S UTC')}\n"
            f"\n*Ordre exécuté sur le compte Paper Alpaca — aucun capital réel engagé.*"
        ),
        "order": {
            "id": str(order.id),
            "symbol": ticker,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        },
    }


_NOT_CONFIGURED_TEXT = (
    "Outils Alpaca non disponibles — clés API manquantes. "
    "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
)


def run_get_news(args: GetNewsArgs) -> dict[str, Any]:
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
        date_str = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else str(created or "")
        lines.append(f"### {headline}")
        lines.append(f"*{source}* — {date_str}")
        lines.append(f"{summary[:300]}" + ("..." if len(summary) > 300 else ""))
        lines.append(f"[Lire plus]({url})" if url else "")
        lines.append("")

    if not articles:
        lines.append("*Aucun article trouvé.*")

    return {
        "text": "\n".join(lines),
        "articles": articles,
        "article_count": len(articles),
    }


def run_portfolio_history(args: PortfolioHistoryArgs) -> dict[str, Any]:
    client = get_alpaca_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}

    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        req = GetPortfolioHistoryRequest(
            period=args.period,
            timeframe=args.timeframe,
            extended_hours=args.extended_hours,
        )
        if args.start:
            req.start = args.start
        if args.end:
            req.end = args.end

        history = client.get_portfolio_history(history_filter=req)
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    if not history or not hasattr(history, "timestamp") or not history.timestamp:
        return {"text": "## Historique du portefeuille\n*Aucune donnée disponible.*", "data_points": 0}

    from datetime import timezone

    lines = [
        "## Historique du portefeuille",
        f"Période: {args.period or '1M'} | Timeframe: {history.timeframe or args.timeframe or 'auto'}",
        f"Base value: ${history.base_value:,.2f}" if history.base_value else "",
        "",
        "| Date | Equity | P&L | P&L % |",
        "|---|---|---|---|",
    ]

    timestamps = history.timestamp
    equities = history.equity
    pls = history.profit_loss
    pl_pcts = history.profit_loss_pct

    data_points = []
    for i in range(len(timestamps)):
        ts = timestamps[i]
        eq = equities[i] if i < len(equities) else 0
        pl = pls[i] if i < len(pls) else 0
        plp = pl_pcts[i] if i < len(pl_pcts) else 0

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d %H:%M")
        equity_str = _fmt_usd(eq)
        pl_str = _fmt_usd(pl) if pl else "$0"
        plp_str = f"{plp:+.4f}%" if plp else "0%"

        lines.append(f"| {date_str} | {equity_str} | {pl_str} | {plp_str} |")
        data_points.append(
            {
                "timestamp": ts,
                "datetime": date_str,
                "equity": round(float(eq), 2),
                "profit_loss": round(float(pl), 2) if pl else 0,
                "profit_loss_pct": round(float(plp), 4) if plp else 0,
            }
        )

    return {
        "text": "\n".join(lines),
        "data_points": data_points,
        "data_count": len(data_points),
        "base_value": round(float(history.base_value), 2) if history.base_value else 0,
        "timeframe": history.timeframe or "",
    }


def run_account_activity(args: AccountActivityArgs) -> dict[str, Any]:
    activities = fetch_account_activities(
        activity_types=args.activity_types,
        date=args.date,
        after=args.after,
        until=args.until,
        direction=args.direction,
        page_size=args.page_size,
    )

    if activities is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}

    if not activities:
        return {"text": "## Activité du compte\n*Aucune activité trouvée.*", "activities": [], "activity_count": 0}

    lines = [
        "## Activité du compte",
        "",
        "| Date | Type | Symbole | Qté | Prix | Montant net | Statut |",
        "|---|---|---|---|---|---|---|",
    ]

    activity_list = []
    for act in activities:
        aid = act.get("id", "")
        atype = act.get("activity_type", act.get("activityType", ""))
        symbol = act.get("symbol", "")
        qty = act.get("qty", "")
        price = act.get("price", "")
        net_amount = act.get("net_amount", "")
        status = act.get("transaction_time", act.get("date", ""))
        desc = act.get("description", "")

        activity_list.append(
            {
                "id": aid,
                "activity_type": atype,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "net_amount": net_amount,
                "date": status,
                "description": desc,
            }
        )

        qty_str = f"{float(qty):.4f}" if qty else ""
        price_str = _fmt_usd(float(price)) if price else ""
        net_str = _fmt_usd(float(net_amount)) if net_amount else ""

        lines.append(f"| {status} | {atype} | {symbol} | {qty_str} | {price_str} | {net_str} | {desc[:40]} |")

    return {
        "text": "\n".join(lines),
        "activities": activity_list,
        "activity_count": len(activity_list),
    }


def run_close_position(args: ClosePositionArgs) -> dict[str, Any]:
    client = get_alpaca_client()
    if client is None:
        return {
            "text": (
                "Outils Alpaca non disponibles — clés API manquantes. "
                "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
            ),
            "error": "alpaca_not_configured",
        }

    try:
        if args.all:
            closes = client.close_all_positions(cancel_orders=True)
            count = len(closes) if closes else 0
            return {
                "text": (
                    f"## Positions liquidées\n"
                    f"Toutes les positions ({count}) ont été fermées sur le compte Paper."
                ),
                "closed_count": count,
            }

        ticker = args.ticker.upper().strip()
        if ticker not in _TRACKED:
            return {
                "text": (
                    f"Ticker {ticker} non autorisé. "
                    f"Tickers supportés: {', '.join(sorted(_TRACKED))}."
                ),
                "error": "invalid_ticker",
            }

        client.close_position(ticker)
        return {
            "text": f"## Position {ticker} fermée\nLa position {ticker} a été liquidée sur le compte Paper.",
            "closed_ticker": ticker,
        }
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}
