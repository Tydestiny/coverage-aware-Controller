"""Validation for canonical trajectories."""

from __future__ import annotations

from .schemas import CanonicalTrajectory


class TrajectoryValidationError(ValueError):
    """A canonical trajectory violates a required invariant."""


def _fail(trajectory_id: str, field: str, message: str) -> None:
    """Raise a validation error identifying the trajectory and field."""
    raise TrajectoryValidationError(f"trajectory_id={trajectory_id!r} field={field}: {message}")


def validate_trajectory(trajectory: CanonicalTrajectory) -> None:
    """Validate required labels, indices, and source-turn preservation metadata."""
    if not trajectory.trajectory_id.strip():
        _fail(trajectory.trajectory_id, "trajectory_id", "must be non-empty")
    if not trajectory.steps:
        _fail(trajectory.trajectory_id, "steps", "must be non-empty")
    if trajectory.outcome not in {"success", "failure"}:
        _fail(trajectory.trajectory_id, "outcome", "must be 'success' or 'failure'")
    for expected, step in enumerate(trajectory.steps):
        if step.step_index != expected:
            _fail(trajectory.trajectory_id, "steps.step_index", "must start at 0 and be continuous")
        if step.source_step_index is None:
            _fail(trajectory.trajectory_id, "steps.source_step_index", "must be present")
    labels = trajectory.labels
    if trajectory.outcome == "failure" and labels.error_onset_step is None:
        _fail(trajectory.trajectory_id, "labels.error_onset_step", "failure requires an onset step")
    if trajectory.outcome == "success" and labels.error_onset_step is not None:
        _fail(trajectory.trajectory_id, "labels.error_onset_step", "success cannot have an onset step")
    if labels.is_failure != (trajectory.outcome == "failure"):
        _fail(trajectory.trajectory_id, "labels.is_failure", "must agree with outcome")
    if labels.source_error_onset_step is None and labels.is_failure:
        _fail(trajectory.trajectory_id, "labels.source_error_onset_step", "must preserve a failure source label")
    if not labels.is_failure and labels.source_error_onset_step is not None:
        _fail(trajectory.trajectory_id, "labels.source_error_onset_step", "success cannot preserve a failure source label")
    source_base = trajectory.metadata.get("mistake_step_base")
    if labels.is_failure and source_base is not None:
        if source_base not in {0, 1}:
            _fail(trajectory.trajectory_id, "metadata.mistake_step_base", "must be 0 or 1")
        assert labels.source_error_onset_step is not None
        if labels.error_onset_step != labels.source_error_onset_step - source_base:
            _fail(trajectory.trajectory_id, "labels.error_onset_step", "must match source_error_onset_step and mistake_step_base")
    if labels.error_onset_step is not None and not 0 <= labels.error_onset_step < len(trajectory.steps):
        _fail(trajectory.trajectory_id, "labels.error_onset_step", "is outside the step range")
    if trajectory.metadata.get("num_steps") != len(trajectory.steps):
        _fail(trajectory.trajectory_id, "metadata.num_steps", "must equal the number of steps")
    raw_count = trajectory.metadata.get("source_turn_count")
    if raw_count != len(trajectory.steps):
        _fail(trajectory.trajectory_id, "metadata.source_turn_count", "must equal canonical step count")
