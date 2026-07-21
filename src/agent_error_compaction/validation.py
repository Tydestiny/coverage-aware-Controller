"""Validation for Canonical Trajectory Schema v1.1."""

from __future__ import annotations

from .schemas import CanonicalTrajectory, SCHEMA_VERSION


class TrajectoryValidationError(ValueError):
    """A v1.1 canonical trajectory violates a required invariant."""


def _fail(trajectory_id: str, field: str, message: str) -> None:
    """Raise an error identifying the trajectory and field."""
    raise TrajectoryValidationError(f"trajectory_id={trajectory_id!r} field={field}: {message}")


def validate_trajectory(trajectory: CanonicalTrajectory) -> None:
    """Validate v1.1 structure, labels, step boundaries, and outcomes."""
    trajectory_id = trajectory.trajectory_id
    if trajectory.schema_version != SCHEMA_VERSION:
        _fail(trajectory_id, "schema_version", f"must be {SCHEMA_VERSION}")
    if not trajectory_id.strip():
        _fail(trajectory_id, "trajectory_id", "must be non-empty")
    if not trajectory.steps:
        _fail(trajectory_id, "steps", "must be non-empty")
    if trajectory.outcome.status not in {"success", "failure", "partial_success", "timeout", "aborted", "unknown"}:
        _fail(trajectory_id, "outcome.status", "has an unsupported value")
    for expected, step in enumerate(trajectory.steps):
        if step.step_index != expected:
            _fail(trajectory_id, "steps.step_index", "must start at 0 and be continuous")
        if step.source_step_index is None:
            _fail(trajectory_id, "steps.source_step_index", "must be present")
        if step.decision_step_index != step.step_index:
            _fail(trajectory_id, "steps.decision_step_index", "must equal step_index for AFTraj")
    if trajectory.metadata.get("num_steps") != len(trajectory.steps):
        _fail(trajectory_id, "metadata.num_steps", "must equal the number of steps")
    if trajectory.metadata.get("source_turn_count") != len(trajectory.steps):
        _fail(trajectory_id, "metadata.source_turn_count", "must equal the number of steps")
    primary = trajectory.annotations.primary_error
    if trajectory.outcome.status == "failure" and primary is None:
        _fail(trajectory_id, "annotations.primary_error", "failure requires a primary error")
    if trajectory.outcome.status == "success" and primary is not None:
        _fail(trajectory_id, "annotations.primary_error", "success cannot have a primary error")
    if primary is not None:
        if not 0 <= primary.step_index < len(trajectory.steps):
            _fail(trajectory_id, "annotations.primary_error.step_index", "is outside the step range")
        if primary.label_semantics != "decisive_mistake_step":
            _fail(trajectory_id, "annotations.primary_error.label_semantics", "must identify decisive_mistake_step")
        if primary.annotation_source != "dataset_author":
            _fail(trajectory_id, "annotations.primary_error.annotation_source", "must be dataset_author")
        if primary.source_step_index != trajectory.steps[primary.step_index].source_step_index:
            _fail(trajectory_id, "annotations.primary_error.source_step_index", "does not match the canonical step")
    derived = trajectory.derived_annotations
    if derived.first_error_step is not None or derived.earliest_detectable_step is not None or derived.evidence_steps or derived.provenance is not None:
        _fail(trajectory_id, "derived_annotations", "must remain empty during preprocessing")

