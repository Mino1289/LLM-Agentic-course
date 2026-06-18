"""Trading tools — place orders and close positions (Alpaca paper trading)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from src.config import TRACKED_TICKERS
from src.tools.schemas import PlaceTradeArgs, ClosePositionArgs

_LOGGER = logging.getLogger("src.tools.trading")

_TRACKED = set(TRACKED_TICKERS)
_MAX_NOTIONAL_PER_ORDER = 10_000.0
# Estimation prudente quand aucun prix réel n'est disponible (ex. clés data absentes).
# Volontairement élevée pour ne pas sous-estimer le notionnel et laisser passer un gros ordre.
_FALLBACK_PRICE_PER_SHARE = 1_000.0

_NOT_CONFIGURED_TEXT = (
    "Outils Alpaca non disponibles — clés API manquantes. "
    "Configure ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env."
)

_APPROVAL_REQUIRED_TEXT = (
    "## Ordre non exécuté — approbation humaine requise\n"
    "Les ordres ne peuvent être soumis qu'après validation Compliance "
    "et approbation explicite de l'utilisateur."
)


def _trade_allowed_without_approval(state: dict[str, Any] | None) -> bool:
    if (state or {}).get("human_approved"):
        return True
    flag = os.getenv("ALLOW_TRADE_WITHOUT_APPROVAL", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        _LOGGER.warning(
            "ALLOW_TRADE_WITHOUT_APPROVAL actif — le garde-fou d'approbation humaine "
            "est CONTOURNÉ. À n'utiliser qu'en développement."
        )
        return True
    return False


def _estimate_notional(ticker: str, qty: float, limit_price: float | None) -> float:
    """Notionnel basé sur le prix réel (limite > dernier prix > estimation prudente)."""
    if limit_price is not None and limit_price > 0:
        return qty * limit_price
    from src.alpaca.client import fetch_latest_price

    last_price = fetch_latest_price(ticker)
    if last_price is not None and last_price > 0:
        return qty * last_price
    _LOGGER.warning(
        "Prix réel indisponible pour %s — estimation prudente à %s/action.",
        ticker,
        _FALLBACK_PRICE_PER_SHARE,
    )
    return qty * _FALLBACK_PRICE_PER_SHARE


def _fmt_usd(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value:,.0f}"
    if abs(value) >= 1_000:
        return f"${value:,.2f}"
    return f"${value:.2f}"


def run_place_trade(
    args: PlaceTradeArgs, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not _trade_allowed_without_approval(state):
        return {"text": _APPROVAL_REQUIRED_TEXT, "error": "human_approval_required"}
    from src.alpaca.client import get_alpaca_client, format_alpaca_error

    ticker = args.ticker.upper().strip()
    if ticker not in _TRACKED:
        return {
            "text": f"Ticker {ticker} non autorisé. Supportés: {', '.join(sorted(_TRACKED))}.",
            "error": "invalid_ticker",
        }
    side = args.side.lower()
    if side not in ("buy", "sell"):
        return {"text": "Le côté doit être 'buy' ou 'sell'.", "error": "invalid_side"}
    qty = args.qty
    order_type = (args.order_type or "market").lower()
    if order_type not in ("market", "limit", "stop_limit", "stop"):
        return {
            "text": "Type d'ordre invalide: market, limit, stop ou stop_limit.",
            "error": "invalid_order_type",
        }
    notional = _estimate_notional(ticker, qty, args.limit_price)
    if notional > _MAX_NOTIONAL_PER_ORDER:
        return {
            "text": f"## Ordre non exécuté — montant estimé trop élevé\nTicker: {ticker}\nQuantité: {qty}\nEstimation: {_fmt_usd(notional)} (max: {_fmt_usd(_MAX_NOTIONAL_PER_ORDER)})",
            "error": "notional_exceeded",
        }

    client = get_alpaca_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}

    try:
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        time_in_force = TimeInForce.DAY
        if order_type == "market":
            order_data = MarketOrderRequest(
                symbol=ticker, qty=qty, side=side_enum, time_in_force=time_in_force
            )
        elif order_type == "limit":
            if args.limit_price is None:
                return {"text": "Prix limite requis.", "error": "missing_limit_price"}
            order_data = LimitOrderRequest(
                symbol=ticker,
                limit_price=args.limit_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        elif order_type == "stop":
            if args.stop_price is None:
                return {"text": "Prix stop requis.", "error": "missing_stop_price"}
            order_data = StopOrderRequest(
                symbol=ticker,
                stop_price=args.stop_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        elif order_type == "stop_limit":
            if args.stop_price is None or args.limit_price is None:
                return {
                    "text": "Prix stop et limite requis.",
                    "error": "missing_stop_or_limit",
                }
            order_data = StopLimitOrderRequest(
                symbol=ticker,
                stop_price=args.stop_price,
                limit_price=args.limit_price,
                qty=qty,
                side=side_enum,
                time_in_force=time_in_force,
            )
        else:
            return {
                "text": f"Type non supporté: {order_type}",
                "error": "unsupported_order_type",
            }
        order = client.submit_order(order_data)
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}

    return {
        "text": f"## Ordre soumis (Paper)\n- **{side.upper()}** {qty} x **{ticker}**\n- Type: {order_type}\n- ID: {order.id}\n- Statut: {order.status}\n- Soumis à: {datetime.now(UTC).strftime('%H:%M:%S UTC')}",
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


def run_close_position(
    args: ClosePositionArgs, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not _trade_allowed_without_approval(state):
        return {"text": _APPROVAL_REQUIRED_TEXT, "error": "human_approval_required"}
    from src.alpaca.client import get_alpaca_client, format_alpaca_error

    client = get_alpaca_client()
    if client is None:
        return {"text": _NOT_CONFIGURED_TEXT, "error": "alpaca_not_configured"}
    try:
        if args.all:
            closes = client.close_all_positions(cancel_orders=True)
            count = len(closes) if closes else 0
            return {
                "text": f"## Positions liquidées\nToutes les positions ({count}) fermées.",
                "closed_count": count,
            }
        ticker = args.ticker.upper().strip()
        if ticker not in _TRACKED:
            return {
                "text": f"Ticker {ticker} non autorisé. Supportés: {', '.join(sorted(_TRACKED))}.",
                "error": "invalid_ticker",
            }
        client.close_position(ticker)
        return {"text": f"## Position {ticker} fermée.", "closed_ticker": ticker}
    except Exception as exc:
        return {"text": format_alpaca_error(exc), "error": "alpaca_error"}
