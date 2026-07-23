"""Trigger detector protocol."""

from __future__ import annotations

from typing import Protocol

from controller_agent.state.schemas import TriggerResult


class TriggerDetector(Protocol):
    """Detect whether a Controller call is required."""

    def detect(self, current_agent_output: str) -> TriggerResult:
        """Inspect the current Agent output."""
