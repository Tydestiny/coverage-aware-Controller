"""Trigger detector protocol."""

from __future__ import annotations

from typing import Protocol

from controller_agent.state.schemas import TriggerContext, TriggerResult


class TriggerDetector(Protocol):
    """Detect whether a Controller call is required."""

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Inspect the current Agent runtime context."""
