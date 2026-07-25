"""Deterministic Controller backend for integration smoke tests."""

from __future__ import annotations

from controller_agent.state.schemas import Action, ControllerDecision, ControllerInput, ControllerResponse


class DeterministicSmokeController:
    """Block the first finish attempt and allow the second one."""

    def __init__(self) -> None:
        self.before_finish_calls = 0

    def decide(self, controller_input: ControllerInput) -> ControllerResponse:
        """Return deterministic decisions without any model call."""
        if controller_input.trigger_reason == "before_finish":
            self.before_finish_calls += 1
        if self.before_finish_calls == 1:
            return ControllerResponse(
                ControllerDecision(
                    Action.VERIFY_TARGET_SET,
                    "smoke_first_finish_requires_verification",
                    (
                        "Verify that the task target and answer are correct. "
                        "In the next turn you must output an executable Python code block, not natural-language reflection only. "
                        "If verification confirms the task is complete, call apis.supervisor.complete_task(...) again with the answer."
                    ),
                )
            )
        return ControllerResponse(
            ControllerDecision(Action.PROCEED, "smoke_second_finish_allowed")
        )
