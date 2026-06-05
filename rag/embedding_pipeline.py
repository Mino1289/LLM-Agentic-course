"""Exponential backoff + granular quota tracking for the embedding pipeline.

PRD etape 3 (§2.3) : backoff exponentiel natif + suivi granulaire de
``quota-used`` à chaud, avec persistance pour reprise entre runs.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

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
    """Return True for errors that must not be retried.

    Blacklist explicite (PRD etape 3 — decision 1) : on inspecte le type
    puis le message de l'exception pour matcher les signaux auth/dim.
    """
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
    """Call ``fn`` with exponential backoff + full jitter on transient errors.

    Fail-fast on errors classified as permanent (auth, dim mismatch, ...).
    ``sleep`` is resolved lazily so test-time patches of ``time.sleep`` apply.
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    rng = rng or random.Random()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — we classify below
            if is_permanent(exc):
                raise
            last_exc = exc
            if attempt >= config.max_retries:
                break
            sleep_fn(_compute_sleep(config, attempt, rng))
    assert last_exc is not None
    raise last_exc


@dataclass
class QuotaState:
    """Persistent per-day embedding quota counter.

    Format JSON enrichi (PRD etape 3 — decision 3) :
    ``{date, quota_used, last_batch_size, last_error, last_updated}``.
    """

    path: Path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_payload()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_payload()
        if not isinstance(data, dict):
            return self._default_payload()
        return {
            "date": str(data.get("date", self._today())),
            "quota_used": int(data.get("quota_used", 0)),
            "last_batch_size": int(data.get("last_batch_size", 0)),
            "last_error": data.get("last_error"),
            "last_updated": str(data.get("last_updated", "")),
        }

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def quota_used(self) -> int:
        return self.load()["quota_used"]

    def update(self, *, batch_size: int, last_error: Optional[str] = None) -> None:
        current = self.load()
        new_used = max(0, int(current["quota_used"])) + max(0, int(batch_size))
        # ``last_error`` is sticky: a successful batch keeps the previous error
        # visible for ops/debug until an explicit None is recorded.
        recorded_error = (
            last_error if last_error is not None else current.get("last_error")
        )
        payload = {
            "date": self._today(),
            "quota_used": new_used,
            "last_batch_size": int(batch_size),
            "last_error": recorded_error,
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.save(payload)

    def reset(self) -> None:
        self.save(self._default_payload())

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "quota_used": 0,
            "last_batch_size": 0,
            "last_error": None,
            "last_updated": "",
        }

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()


__all__ = [
    "BackoffConfig",
    "QuotaState",
    "is_permanent_error",
    "with_exponential_backoff",
]


def asdict_safe(obj: Any) -> dict[str, Any]:
    """Helper for callers that want a plain dict view of a BackoffConfig."""
    return asdict(obj)
