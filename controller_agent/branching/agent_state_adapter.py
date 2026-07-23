"""Helpers for reconstructing visible Agent state from trajectory prefixes."""

from __future__ import annotations

import copy
from typing import Any

from controller_agent.branching.schemas import StaticTrajectory


class AgentStateAdapter:
    """Build model-visible state from static trajectory prefixes."""

    def messages_at_checkpoint(self, trajectory: StaticTrajectory, checkpoint_step: int) -> list[dict[str, Any]]:
        """Return messages visible immediately after checkpoint_step."""
        self._validate_checkpoint_step(trajectory, checkpoint_step)
        return copy.deepcopy(trajectory.steps[checkpoint_step].messages_after_step)

    def last_execution_outputs(self, trajectory: StaticTrajectory, checkpoint_step: int) -> list[Any]:
        """Return the last environment output visible to the next Agent turn."""
        self._validate_checkpoint_step(trajectory, checkpoint_step)
        return [trajectory.steps[checkpoint_step].execution_output]

    def assert_no_future_messages(self, trajectory: StaticTrajectory, checkpoint_step: int) -> None:
        """Raise if checkpoint messages contain future step content."""
        self._validate_checkpoint_step(trajectory, checkpoint_step)
        visible = str(trajectory.steps[checkpoint_step].messages_after_step)
        for future_step in trajectory.steps[checkpoint_step + 1 :]:
            if future_step.execution_code and future_step.execution_code in visible:
                raise ValueError(f"Future execution leaked into checkpoint messages: step {future_step.step_index}")
            if future_step.execution_output and future_step.execution_output in visible:
                raise ValueError(f"Future observation leaked into checkpoint messages: step {future_step.step_index}")

    def _validate_checkpoint_step(self, trajectory: StaticTrajectory, checkpoint_step: int) -> None:
        """Validate checkpoint bounds."""
        if checkpoint_step < 0 or checkpoint_step >= len(trajectory.steps):
            raise IndexError(f"checkpoint_step out of range: {checkpoint_step}")

