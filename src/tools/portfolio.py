"""Portfolio tools — info, history, account activity (Alpaca paper trading)."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from src.config import TRACKED_TICKERS
from src.tools.schemas import PortfolioInfoArgs, PortfolioHistoryArgs, AccountActivityArgs

_LOGGER = logging.getLogger("src.tools.portfolio")
_TRACKED = set(TRACKED_TICKERS)


def _fmt_usd(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value:,.0f}"
    if abs(value) >= 1_000:
        return f"${value:,.2f}"
    return f"${value:.2f}"


_NOT_CONFIGURED_TEXT = (
    "Outils Alpaca non disponibles — clés API manquantes. "
    "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
)


def run_portfolio_info(args: PortfolioInfoArgs) -> dict[str, Any]:
    from src.alpaca.client import get_alpaca_client, format_alpaca_error

    client = get_alpaca_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}
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
        f"P&L non réalisé: {_fmt_usd(float(account.equity) - float(account.last_equity or account.equity))}",
        "",
    ]
    pos_list: list[dict[str, Any]] = []
    if positions:
        lines.append(f"### Positions ({len(positions)})")
        for pos in positions:
            sym = pos.symbol
            qty = float(pos.qty)
            mkt_val = float(pos.market_value)
            upnl = float(pos.unrealized_pl)
            upnl_pct = float(pos.unrealized_plpc) * 100
            pos_list.append({
                "ticker": sym, "qty": qty, "market_value": round(mkt_val, 2),
                "cost_basis": round(float(pos.cost_basis), 2),
                "unrealized_pl": round(upnl, 2), "unrealized_pl_pct": round(upnl_pct, 2),
            })
            lines.append(f"- **{sym}**: {qty:.4f} | Val.: {_fmt_usd(mkt_val)} | P&L: {_fmt_usd(upnl)} ({upnl_pct:+.2f}%)")
    else:
        lines.append("*Aucune position ouverte.*")
    return {
        "text": "\n".join(lines),
        "account": {
            "cash": round(float(account.cash), 2), "equity": round(float(account.equity), 2),
            "buying_power": round(float(account.buying_power), 2),
            "unrealized_pl": round(float(account.equity) - float(account.last_equity or account.equity), 2),
        },
        "positions": pos_list, "position_count": len(pos_list),
    }


def run_portfolio_history(args: PortfolioHistoryArgs) -> dict[str, Any]:
    from src.alpaca.client import get_alpaca_client, format_alpaca_error

    client = get_alpaca_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(
            period=args.period, timeframe=args.timeframe, extended_hours=args.extended_hours,
        )
        if args.start:
            req.start = args.start
        if args.end:
            req.end = args.end
        history = client.get_portfolio_history(history_filter=req)
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    if not history or not hasattr(history, "timestamp") or not history.timestamp:
        return {"text": "## Historique\n*Aucune donnée.*", "data_points": 0}

    from datetime import timezone

    lines = [
        "## Historique du portefeuille",
        f"Période: {args.period or '1M'} | Timeframe: {history.timeframe or args.timeframe or 'auto'}",
        f"Base value: ${history.base_value:,.2f}" if history.base_value else "",
        "", "| Date | Equity | P&L | P&L % |", "|---|---|---|---|",
    ]
    data_points = []
    for i in range(len(history.timestamp)):
        ts = history.timestamp[i]
        eq = history.equity[i] if i < len(history.equity) else 0
        pl = history.profit_loss[i] if i < len(history.profit_loss) else 0
        plp = history.profit_loss_pct[i] if i < len(history.profit_loss_pct) else 0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        lines.append(f"| {dt.strftime('%Y-%m-%d %H:%M')} | {_fmt_usd(eq)} | {_fmt_usd(pl) if pl else '$0'} | {plp:+.4f}% |")
        data_points.append({
            "timestamp": ts, "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "equity": round(float(eq), 2), "profit_loss": round(float(pl), 2) if pl else 0,
            "profit_loss_pct": round(float(plp), 4) if plp else 0,
        })
    return {"text": "\n".join(lines), "data_points": data_points, "data_count": len(data_points),
            "base_value": round(float(history.base_value), 2) if history.base_value else 0,
            "timeframe": history.timeframe or ""}


def run_account_activity(args: AccountActivityArgs) -> dict[str, Any]:
    from src.alpaca.client import fetch_account_activities, format_alpaca_error

    activities = fetch_account_activities(
        activity_types=args.activity_types, date=args.date, after=args.after,
        until=args.until, direction=args.direction, page_size=args.page_size,
    )
    if activities is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}
    if not activities:
        return {"text": "## Activité\n*Aucune activité.*", "activities": [], "activity_count": 0}

    lines = ["## Activité du compte", "", "| Date | Type | Symbole | Qté | Prix | Montant net | Statut |", "|---|---|---|---|---|---|---|"]
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
        activity_list.append({
            "id": aid, "activity_type": atype, "symbol": symbol, "qty": qty,
            "price": price, "net_amount": net_amount, "date": status, "description": desc,
        })
        qty_str = f"{float(qty):.4f}" if qty else ""
        price_str = _fmt_usd(float(price)) if price else ""
        net_str = _fmt_usd(float(net_amount)) if net_amount else ""
        lines.append(f"| {status} | {atype} | {symbol} | {qty_str} | {price_str} | {net_str} | {desc[:40]} |")
    return {"text": "\n".join(lines), "activities": activity_list, "activity_count": len(activity_list)}
