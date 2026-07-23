"""Namespace snapshot adapters for AppWorld Python execution state."""

from __future__ import annotations

from typing import Any, Protocol


class NamespaceAdapter(Protocol):
    """Protocol for Python namespace snapshot and restore support."""

    def snapshot(self, namespace: Any) -> Any:
        """Return a serializable namespace snapshot."""

    def restore(self, snapshot: Any) -> Any:
        """Restore a namespace snapshot."""

    def is_supported(self) -> bool:
        """Return whether reliable namespace restore is supported."""


class ReplayRequiredNamespaceAdapter:
    """Conservative adapter used until AppWorld REPL locals are safely serializable."""

    def snapshot(self, namespace: Any) -> dict[str, str]:
        """Return a marker explaining that prefix replay is required."""
        return {"namespace_restore_mode": "replay_required"}

    def restore(self, snapshot: Any) -> None:
        """Reject namespace restoration because it is not supported."""
        raise NotImplementedError("Python namespace restore is not supported; replay is required.")

    def is_supported(self) -> bool:
        """Return False because AppWorld REPL locals are not proven serializable."""
        return False

