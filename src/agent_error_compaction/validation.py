"""Generic and dataset-specific validation for Canonical Schema v1.1."""

from __future__ import annotations

from .schemas import CanonicalTrajectory, SCHEMA_VERSION


class TrajectoryValidationError(ValueError):
    """A canonical trajectory violates a required invariant."""


def _fail(trajectory_id: str, field: str, message: str) -> None:
    raise TrajectoryValidationError(f"trajectory_id={trajectory_id!r} field={field}: {message}")


class CanonicalValidator:
    """Validate invariants shared by all canonical datasets."""

    allowed_statuses = {"success", "failure", "partial_success", "timeout", "aborted", "unknown"}
    allowed_alignment = {"exact", "span", "ambiguous", "unmapped"}
    allowed_annotation_status = {"source_provided", "rule_mapped", "not_provided", "unavailable", "not_applicable"}

    def validate(self, trajectory: CanonicalTrajectory) -> None:
        """Raise when a cross-dataset invariant is violated."""
        tid = trajectory.trajectory_id
        if trajectory.schema_version != SCHEMA_VERSION:
            _fail(tid, "schema_version", f"must be {SCHEMA_VERSION}")
        if not isinstance(tid, str) or not tid.strip():
            _fail(tid, "trajectory_id", "must be non-empty")
        if not trajectory.steps:
            _fail(tid, "steps", "must be non-empty")
        if trajectory.outcome.status not in self.allowed_statuses:
            _fail(tid, "outcome.status", "has an unsupported value")
        for expected, step in enumerate(trajectory.steps):
            if step.step_index != expected:
                _fail(tid, "steps.step_index", "must start at 0 and be continuous")
            if step.source_step_index is None:
                _fail(tid, "steps.source_step_index", "must be present")
            for event_expected, event in enumerate(step.events):
                if event.event_index != event_expected:
                    _fail(tid, "steps.events.event_index", "must be continuous within each step")
        if trajectory.metadata.get("num_steps") != len(trajectory.steps):
            _fail(tid, "metadata.num_steps", "must equal the number of steps")
        if trajectory.annotations.annotation_status not in self.allowed_annotation_status:
            _fail(tid, "annotations.annotation_status", "has an unsupported value")
        primary = trajectory.annotations.primary_error
        if primary is not None:
            if primary.alignment_status not in self.allowed_alignment:
                _fail(tid, "annotations.primary_error.alignment_status", "has an unsupported value")
            if primary.alignment_status == "exact":
                if primary.step_index is None or not 0 <= primary.step_index < len(trajectory.steps):
                    _fail(tid, "annotations.primary_error.step_index", "is outside the step range")
            if primary.alignment_status == "span":
                span = primary.step_span
                if span is None or not 0 <= span.start_step_index <= span.end_step_index < len(trajectory.steps):
                    _fail(tid, "annotations.primary_error.step_span", "is invalid")
            if primary.source_step_index is not None and primary.step_index is not None:
                step = trajectory.steps[primary.step_index]
                if step.source_step_index != primary.source_step_index:
                    _fail(tid, "annotations.primary_error.source_step_index", "does not match the canonical step")
        for name in ("first_error_step", "earliest_detectable_step", "evidence_steps"):
            field = getattr(trajectory.derived_annotations, name)
            if field.status != "not_annotated" or field.value not in (None, [] ) or field.provenance is not None:
                _fail(tid, f"derived_annotations.{name}", "must remain not_annotated during preprocessing")


class AFTrajValidator(CanonicalValidator):
    """Validate AFTraj-specific one-turn-one-step and label invariants."""

    def validate(self, trajectory: CanonicalTrajectory) -> None:
        """Validate generic rules plus AFTraj source alignment rules."""
        super().validate(trajectory)
        tid = trajectory.trajectory_id
        if trajectory.source.dataset != "AFTraj-2K":
            _fail(tid, "source.dataset", "must be AFTraj-2K for AFTrajValidator")
        if trajectory.metadata.get("source_turn_count") != len(trajectory.steps):
            _fail(tid, "metadata.source_turn_count", "must equal the number of canonical steps")
        for step in trajectory.steps:
            if step.source_unit_type != "turn":
                _fail(tid, "steps.source_unit_type", "must be turn for AFTraj")
            if step.source_step_index != step.step_index or step.decision_step_index != step.step_index:
                _fail(tid, "steps", "AFTraj turn and decision indices must align to the canonical step")
        primary = trajectory.annotations.primary_error
        if trajectory.outcome.status == "failure" and primary is None:
            _fail(tid, "annotations.primary_error", "failure requires a primary error")
        if trajectory.outcome.status == "success" and primary is not None:
            _fail(tid, "annotations.primary_error", "success cannot have a primary error")
        if primary is not None and primary.alignment_status != "exact":
            _fail(tid, "annotations.primary_error.alignment_status", "AFTraj labels must be exact")


def validate_trajectory(trajectory: CanonicalTrajectory) -> None:
    """Validate a trajectory using the generic cross-dataset validator."""
    CanonicalValidator().validate(trajectory)
