"""Formatting of temporary guidance messages."""

from __future__ import annotations

from controller_agent.state.schemas import ControllerDecision


class GuidanceBuilder:
    """Build short, explicit guidance for the next Agent turn."""

    def build(self, decision: ControllerDecision) -> str:
        """Return empty guidance for PROCEED and a temporary message otherwise."""
        if decision.action.value == "PROCEED":
            return ""
        instruction = decision.instruction.strip() or "Re-check the task state and continue safely before finishing."
        return f"[Temporary Controller guidance: {decision.action.value}] {instruction}"
