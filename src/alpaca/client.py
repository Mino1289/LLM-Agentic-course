"""Client Alpaca — TradingClient + NewsClient + account activities via HTTP."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx

from src.paths import load_project_env

load_project_env()

_LOGGER = logging.getLogger("src.alpaca.client")


class AlpacaNotConfiguredError(RuntimeError):
    """Raised when Alpaca API keys are missing from the environment."""


def _api_key() -> str | None:
    return os.getenv("ALPACA_API_KEY")


def _secret_key() -> str | None:
    return os.getenv("ALPACA_SECRET_KEY")


def _is_paper() -> bool:
    return os.getenv("ALPACA_PAPER_TRADE", "true").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def _trading_base_url() -> str:
    return (
        "https://paper-api.alpaca.markets"
        if _is_paper()
        else "https://api.alpaca.markets"
    )


def _data_base_url() -> str:
    return "https://data.alpaca.markets"


def _check_keys() -> bool:
    k1, k2 = _api_key(), _secret_key()
    if not k1 or not k2:
        _LOGGER.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — Alpaca tools unavailable."
        )
        return False
    return True


@lru_cache(maxsize=1)
def get_alpaca_client() -> Any:
    if not _check_keys():
        return None
    try:
        from alpaca.trading.client import TradingClient
    except ModuleNotFoundError:
        _LOGGER.error("alpaca-py is not installed. Run: pip install alpaca-py")
        return None
    client = TradingClient(_api_key(), _secret_key(), paper=_is_paper())
    try:
        _ = client.get_account()
    except Exception as exc:
        _LOGGER.warning("Alpaca API probe failed: %s", exc)
        return None
    return client


@lru_cache(maxsize=1)
def get_news_client() -> Any:
    if not _check_keys():
        return None
    try:
        from alpaca.data.historical.news import NewsClient
    except ModuleNotFoundError:
        _LOGGER.error("alpaca-py is not installed. Run: pip install alpaca-py")
        return None
    return NewsClient(_api_key(), _secret_key())


def fetch_account_activities(
    activity_types: list[str] | None = None,
    date: str | None = None,
    after: str | None = None,
    until: str | None = None,
    direction: str | None = "desc",
    page_size: int = 50,
) -> list[dict[str, Any]]:
    if not _check_keys():
        return []
    params: dict[str, Any] = {"direction": direction or "desc", "page_size": page_size}
    if activity_types:
        params["activity_types"] = ",".join(activity_types)
    if date:
        params["date"] = date
    if after:
        params["after"] = after
    if until:
        params["until"] = until
    url = f"{_trading_base_url()}/v2/account/activities"
    headers = {
        "APCA-API-KEY-ID": _api_key(),
        "APCA-API-SECRET-KEY": _secret_key(),
    }
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        _LOGGER.warning("Account activities fetch failed: %s", exc)
        return []


def format_alpaca_error(exc: Exception) -> str:
    msg = str(exc)
    if "401" in msg:
        return "Erreur d'authentification Alpaca — vérifie tes API keys."
    if "403" in msg:
        return "Accès refusé — vérifie les permissions de tes clés Alpaca."
    if "429" in msg:
        return "Trop de requêtes Alpaca — attends un moment puis réessaie."
    return f"Erreur Alpaca: {msg}"
