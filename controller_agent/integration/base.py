"""Integration boundary for applying execution interception decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from controller_agent.state.schemas import InterceptionResult


class ExecutionInterceptionAdapter(Protocol):
    """Apply a core interception result to a concrete agent runtime."""

    def apply(self, original_result: Any, interception: InterceptionResult, agent: Any) -> Any:
        """Return the runtime-specific result for the wrapped agent."""


@dataclass(frozen=True)
class OfflineInterceptionResult:
    """Explicit offline result for blocked execution without AppWorld semantics."""

    integration_status: str
    original_result: Any
    interception: InterceptionResult


class OfflineMockAdapter:
    """Test adapter that never mutates proposed execution code."""

    def __init__(self) -> None:
        self.applied: list[InterceptionResult] = []

    def apply(self, original_result: Any, interception: InterceptionResult, agent: Any) -> Any:
        """Return original execution or an explicit unconfigured block result."""
        self.applied.append(interception)
        if interception.allow_execution:
            return original_result
        return OfflineInterceptionResult(
            integration_status="integration not configured",
            original_result=original_result,
            interception=interception,
        )
