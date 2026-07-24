"""Formatting of temporary guidance messages."""

from __future__ import annotations

from controller_agent.state.schemas import Action, ControllerDecision


class GuidanceBuilder:
    """Build short, explicit guidance for the next Agent turn."""

    def build(self, decision: ControllerDecision) -> str:
        """Return empty guidance for PROCEED; otherwise Action/Advice block.

        Format matches offline branch experiments that improved pass_rate.
        """
        if decision.action == Action.PROCEED:
            return ""
        instruction = (
            decision.instruction.strip()
            or "Re-check the task state and continue safely before finishing."
        )
        return (
            f"Action: {decision.action.value}\n"
            f"Advice: {instruction}\n"
            "Follow this advice in your next steps. "
            "Revise the plan before repeating a blocked or premature action."
        )