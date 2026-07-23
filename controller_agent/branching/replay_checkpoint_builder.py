"""Replay static trajectory prefixes to reconstruct checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from controller_agent.branching.agent_state_adapter import AgentStateAdapter
from controller_agent.branching.schemas import ReconstructedCheckpoint, StaticTrajectory


class ReplayCheckpointBuilder:
    """Build checkpoints by replaying static code into a fresh world."""

    def __init__(
        self,
        world_factory: Callable[[str], Any],
        state_adapter: AgentStateAdapter | None = None,
    ) -> None:
        self.world_factory = world_factory
        self.state_adapter = state_adapter or AgentStateAdapter()

    def build(self, trajectory: StaticTrajectory, checkpoint_step: int) -> ReconstructedCheckpoint:
        """Replay trajectory through checkpoint_step and return checkpoint metadata."""
        if checkpoint_step < 0 or checkpoint_step >= len(trajectory.steps):
            raise IndexError(f"checkpoint_step out of range: {checkpoint_step}")
        world = self.world_factory(trajectory.task_id)
        validation_errors: list[str] = []
        execution_count = 0
        for step in trajectory.steps[: checkpoint_step + 1]:
            if not step.execution_code:
                return self._failed_checkpoint(
                    trajectory,
                    checkpoint_step,
                    "invalid_trajectory",
                    [f"Missing execution_code at step {step.step_index}"],
                    execution_count,
                )
            output = self._execute(world, step.execution_code)
            execution_count += 1
            if _normalize_output(output) != _normalize_output(step.execution_output):
                return self._failed_checkpoint(
                    trajectory,
                    checkpoint_step,
                    "replay_diverged",
                    [
                        f"First replay divergence at step {step.step_index}",
                        f"expected={step.execution_output!r}",
                        f"actual={output!r}",
                    ],
                    execution_count,
                )
        try:
            self.state_adapter.assert_no_future_messages(trajectory, checkpoint_step)
        except ValueError as exc:
            validation_errors.append(str(exc))
        state_id = self._save_state(world, trajectory.task_id, checkpoint_step)
        return ReconstructedCheckpoint(
            task_id=trajectory.task_id,
            checkpoint_step=checkpoint_step,
            environment_state_id=state_id,
            agent_messages=self.state_adapter.messages_at_checkpoint(trajectory, checkpoint_step),
            last_execution_outputs=self.state_adapter.last_execution_outputs(trajectory, checkpoint_step),
            agent_turn_count=checkpoint_step + 1,
            environment_execution_count=execution_count,
            source_trajectory=trajectory.source_path,
            validation_status="valid" if not validation_errors else "invalid_agent_state",
            validation_errors=validation_errors,
        )

    def _execute(self, world: Any, code: str) -> Any:
        """Execute one code block against a world-like object."""
        if hasattr(world, "batch_execute"):
            outputs = world.batch_execute([code])
            return outputs[0] if outputs else ""
        if hasattr(world, "execute"):
            return world.execute(code)
        raise TypeError("world must expose batch_execute or execute")

    def _save_state(self, world: Any, task_id: str, checkpoint_step: int) -> Any:
        """Save world state and return a state identifier."""
        if not hasattr(world, "save_state"):
            raise TypeError("world must expose save_state")
        state_id = f"replay_{task_id}_step_{checkpoint_step}"
        try:
            return world.save_state(state_id)
        except TypeError:
            return world.save_state()

    def _failed_checkpoint(
        self,
        trajectory: StaticTrajectory,
        checkpoint_step: int,
        status: str,
        errors: list[str],
        execution_count: int,
    ) -> ReconstructedCheckpoint:
        """Return a failed checkpoint without pretending replay succeeded."""
        visible_step = min(max(checkpoint_step, 0), len(trajectory.steps) - 1)
        return ReconstructedCheckpoint(
            task_id=trajectory.task_id,
            checkpoint_step=checkpoint_step,
            environment_state_id=None,
            agent_messages=(
                self.state_adapter.messages_at_checkpoint(trajectory, visible_step) if trajectory.steps else []
            ),
            last_execution_outputs=[],
            agent_turn_count=visible_step + 1 if trajectory.steps else 0,
            environment_execution_count=execution_count,
            source_trajectory=trajectory.source_path,
            validation_status=status,
            validation_errors=errors,
        )


def _normalize_output(value: Any) -> str:
    """Normalize output for deterministic comparison."""
    return str(value).replace("\r\n", "\n").strip()
