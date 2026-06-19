"""Exponential backoff pour les appels API avec gestion des erreurs permanentes."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

_PERMANENT_TYPE_TOKENS = (
    "Authentication",
    "Permission",
    "InvalidApiKey",
    "Unauthorized",
    "Forbidden",
    "AccessDenied",
    "DimMismatch",
    "DimensionMismatch",
)

_PERMANENT_MESSAGE_TOKENS = (
    "401",
    "403",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "dim mismatch",
    "dimension mismatch",
)


def is_permanent_error(exc: BaseException) -> bool:
    """Return True for errors that must not be retried."""
    qualified = type(exc).__name__
    for token in _PERMANENT_TYPE_TOKENS:
        if token in qualified:
            return True
    message = str(exc).lower()
    for token in _PERMANENT_MESSAGE_TOKENS:
        if token in message:
            return True
    return False


@dataclass(frozen=True)
class BackoffConfig:
    max_retries: int = 3
    base_sec: float = 1.0
    cap_sec: float = 60.0
    jitter: str = "full"  # "full" | "equal" | "none"

    @classmethod
    def from_env(cls) -> "BackoffConfig":
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        jitter = os.getenv("EMBEDDING_BACKOFF_JITTER", "full").lower()
        if jitter not in {"full", "equal", "none"}:
            jitter = "full"

        return cls(
            max_retries=_int("EMBEDDING_BACKOFF_MAX_RETRIES", 3),
            base_sec=_float("EMBEDDING_BACKOFF_BASE_SEC", 1.0),
            cap_sec=_float("EMBEDDING_BACKOFF_CAP_SEC", 60.0),
            jitter=jitter,
        )


def _compute_sleep(config: BackoffConfig, attempt: int, rng: random.Random) -> float:
    """Return sleep seconds for a given retry attempt (1-indexed)."""
    raw = config.base_sec * (2 ** (attempt - 1))
    capped = min(config.cap_sec, raw)
    if capped <= 0:
        return 0.0
    if config.jitter == "none":
        return capped
    if config.jitter == "equal":
        half = capped / 2.0
        return half + rng.uniform(0.0, half)
    # full jitter (AWS-recommended)
    return rng.uniform(0.0, capped)


def with_exponential_backoff(
    fn: Callable[[], T],
    *,
    config: BackoffConfig,
    sleep: Optional[Callable[[float], None]] = None,
    is_permanent: Callable[[BaseException], bool] = is_permanent_error,
    rng: Optional[random.Random] = None,
) -> T:
    """Call fn with exponential backoff + full jitter on transient errors.

    Fail-fast on errors classified as permanent (auth, dim mismatch, ...).
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    rng = rng or random.Random()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if is_permanent(exc):
                raise
            last_exc = exc
            if attempt >= config.max_retries:
                break
            sleep_fn(_compute_sleep(config, attempt, rng))
    assert last_exc is not None
    raise last_exc


__all__ = [
    "BackoffConfig",
    "is_permanent_error",
    "with_exponential_backoff",
]
