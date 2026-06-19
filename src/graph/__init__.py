"""Package graph — API publique exposée en lazy pour éviter les imports circulaires.

Importer un module feuille (ex. ``src.graph.state``) ne doit pas tirer ``flow``,
qui dépend de ``src.tools.execute`` — lequel importe lui-même ``src.graph.state``.
On expose donc les symboles à la demande via PEP 562.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.graph.flow import FinanceLangGraphAgent
    from src.graph.memory_store import MemoryStore
    from src.graph.state import GraphState

__all__ = ["FinanceLangGraphAgent", "GraphState", "MemoryStore"]

_LAZY_EXPORTS = {
    "FinanceLangGraphAgent": "src.graph.flow",
    "GraphState": "src.graph.state",
    "MemoryStore": "src.graph.memory_store",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
