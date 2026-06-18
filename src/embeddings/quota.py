"""Gestion du quota d'embeddings avec persistance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class QuotaState:
    """Persistent per-day embedding quota counter."""

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
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    def quota_used(self) -> int:
        return self.load()["quota_used"]

    def update(self, *, batch_size: int, last_error: Optional[str] = None) -> None:
        current = self.load()
        new_used = max(0, int(current["quota_used"])) + max(0, int(batch_size))
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


__all__ = ["QuotaState"]
