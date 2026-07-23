"""Validation helpers for reconstructed checkpoints."""

from __future__ import annotations

from typing import Any

from controller_agent.branching.schemas import ReconstructedCheckpoint, StaticTrajectory


class CheckpointValidator:
    """Validate reconstructed checkpoints without evaluator access."""

    def validate_basic(self, trajectory: StaticTrajectory, checkpoint: ReconstructedCheckpoint) -> list[str]:
        """Return validation errors for static checkpoint invariants."""
        errors: list[str] = []
        if checkpoint.task_id != trajectory.task_id:
            errors.append("task_id mismatch")
        if checkpoint.checkpoint_step < 0 or checkpoint.checkpoint_step >= len(trajectory.steps):
            errors.append("checkpoint_step out of range")
            return errors
        if checkpoint.environment_execution_count != checkpoint.checkpoint_step + 1:
            errors.append("environment execution count mismatch")
        visible = str(checkpoint.agent_messages)
        for future_step in trajectory.steps[checkpoint.checkpoint_step + 1 :]:
            if future_step.execution_code and future_step.execution_code in visible:
                errors.append(f"future execution leaked: step {future_step.step_index}")
            if future_step.execution_output and future_step.execution_output in visible:
                errors.append(f"future observation leaked: step {future_step.step_index}")
        forbidden = ["evaluator", "ground_truth", "predicted_answer"]
        lowered = visible.lower()
        for key in forbidden:
            if key in lowered:
                errors.append(f"forbidden field leaked into agent messages: {key}")
        return errors

    def validate_save_load_read_only(self, world: Any, state_id: Any, probe_code: str) -> list[str]:
        """Validate save/load consistency with a read-only probe."""
        if not hasattr(world, "load_state") or not hasattr(world, "batch_execute"):
            return ["world must expose load_state and batch_execute"]
        before = world.batch_execute([probe_code])[0]
        world.load_state(state_id)
        after = world.batch_execute([probe_code])[0]
        return [] if before == after else ["read-only probe mismatch after load_state"]

