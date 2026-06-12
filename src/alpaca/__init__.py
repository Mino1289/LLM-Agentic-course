"""Module Alpaca — client API + trading tools."""
from src.alpaca.client import (
    AlpacaNotConfiguredError,
    fetch_account_activities,
    format_alpaca_error,
    get_alpaca_client,
    get_news_client,
)

__all__ = [
    "AlpacaNotConfiguredError",
    "fetch_account_activities",
    "format_alpaca_error",
    "get_alpaca_client",
    "get_news_client",
]
