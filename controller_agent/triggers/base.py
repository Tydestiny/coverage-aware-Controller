"""Trigger detector protocol.

Core interface status: frozen for the Coverage-Aware Controller v0 core.
New trigger detectors should implement this protocol and consume only the
visible TriggerContext.
"""

from __future__ import annotations

from typing import Protocol

from controller_agent.state.schemas import TriggerContext, TriggerResult


class TriggerDetector(Protocol):
    """Detect whether a Controller call is required."""

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Inspect the current Agent runtime context."""
