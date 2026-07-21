"""Leakage-safe model and evaluation views for Canonical Schema v1.1.

CanonicalTrajectory is a storage and evaluation object. ModelInputView is the
only valid input to future Compactors and Verifiers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

MODEL_STEP_FIELDS = ("step_index", "decision_step_index", "actor", "message_role", "event_type", "content", "reasoning", "events")
MODEL_EVENT_FIELDS = ("event_index", "event_type", "actor", "content", "tool_call", "tool_result")
FORBIDDEN_MODEL_FIELDS = frozenset({
    "schema_version", "trajectory_id", "trajectory_family_id", "source", "source_split", "original_split",
    "experiment_split", "environment", "gold_answer", "outcome", "annotations", "derived_annotations", "metadata",
    "labels", "is_failure", "error_onset_step", "source_error_onset_step", "critical_error_step", "responsible_actor",
    "error_reason", "error_type_original", "error_type_unified", "label_source", "unsafe_source", "raw_step",
    "source_step_index", "source_step_id", "source_unit_type", "source_event_index", "source_event_id", "raw_event",
})


def _validate_prefix(trajectory: dict[str, Any], prefix_end_step: int) -> list[Any]:
    """Validate a trajectory and inclusive prefix position."""
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be a dictionary")
    if isinstance(prefix_end_step, bool) or not isinstance(prefix_end_step, int):
        raise ValueError("prefix_end_step must be an integer")
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory.steps must be a non-empty list")
    if prefix_end_step < 0 or prefix_end_step >= len(steps):
        raise ValueError(f"prefix_end_step={prefix_end_step} is outside steps range 0..{len(steps) - 1}")
    return steps


def build_model_input_view(trajectory: dict[str, Any], prefix_end_step: int) -> dict[str, Any]:
    """Build a whitelist-only model input containing no labels or future steps."""
    steps = _validate_prefix(trajectory, prefix_end_step)
    task = trajectory.get("task")
    environment = trajectory.get("environment")
    if not isinstance(task, dict) or "instruction" not in task:
        raise ValueError("trajectory.task.instruction is required for model input")
    if not isinstance(environment, dict):
        raise ValueError("trajectory.environment is required for model input")
    output_steps = []
    for index, step in enumerate(steps[: prefix_end_step + 1]):
        if not isinstance(step, dict):
            raise ValueError(f"trajectory.steps[{index}] must be a dictionary")
        missing = [field for field in MODEL_STEP_FIELDS if field not in step]
        if missing:
            raise ValueError(f"trajectory.steps[{index}] missing fields: {', '.join(missing)}")
        events = []
        if not isinstance(step["events"], list):
            raise ValueError(f"trajectory.steps[{index}].events must be a list")
        for event_index, event in enumerate(step["events"]):
            if not isinstance(event, dict):
                raise ValueError(f"trajectory.steps[{index}].events[{event_index}] must be a dictionary")
            missing_event = [field for field in MODEL_EVENT_FIELDS if field not in event]
            if missing_event:
                raise ValueError(f"trajectory.steps[{index}].events[{event_index}] missing fields: {', '.join(missing_event)}")
            events.append({field: deepcopy(event[field]) for field in MODEL_EVENT_FIELDS})
        output_steps.append({field: deepcopy(step[field]) for field in MODEL_STEP_FIELDS if field != "events"} | {"events": events})
    model_input = {"task": deepcopy(task["instruction"]), "tools": deepcopy(environment.get("tools", [])), "steps": output_steps}
    validate_model_input_view(model_input)
    return model_input


def build_evaluation_view(trajectory: dict[str, Any], prefix_end_step: int) -> dict[str, Any]:
    """Build an offline evaluation view that is never sent to a model."""
    _validate_prefix(trajectory, prefix_end_step)
    annotations = trajectory.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("trajectory.annotations must be a dictionary")
    primary = annotations.get("primary_error")
    outcome = trajectory.get("outcome")
    if isinstance(outcome, dict) and outcome.get("status") == "success":
        primary = None
    if isinstance(primary, dict) and primary.get("alignment_status", "exact") != "exact":
        primary = None
    critical_step = primary.get("step_index") if isinstance(primary, dict) else None
    occurred = critical_step is not None and prefix_end_step >= critical_step
    return {
        "trajectory_id": deepcopy(trajectory.get("trajectory_id")),
        "trajectory_family_id": deepcopy(trajectory.get("trajectory_family_id")),
        "source": {"split": deepcopy(trajectory.get("experiment_split"))},
        "prefix_end_step": prefix_end_step,
        "critical_error_occurred": occurred,
        "critical_error_step": critical_step,
        "distance_to_critical_error": prefix_end_step - critical_step if critical_step is not None else None,
        "responsible_actor": deepcopy(primary.get("actor_id")) if isinstance(primary, dict) else None,
        "error_reason": deepcopy(primary.get("reason")) if isinstance(primary, dict) else None,
        "annotation_method": deepcopy(primary.get("annotation_method")) if isinstance(primary, dict) else None,
    }


def _walk(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield every dictionary key and its path."""
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{path}.{key}" if path else str(key)
            yield str(key), current
            yield from _walk(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_model_input_view(model_input: dict[str, Any]) -> None:
    """Reject forbidden fields and non-contiguous model-visible steps/events."""
    if not isinstance(model_input, dict):
        raise ValueError("model input must be a dictionary")
    for field, path in _walk(model_input):
        if field in FORBIDDEN_MODEL_FIELDS:
            raise ValueError(f"Forbidden field found in model input: {path}")
    if "task" not in model_input or "steps" not in model_input:
        raise ValueError("model input must contain task and steps")
    if not isinstance(model_input["steps"], list):
        raise ValueError("model input steps must be a list")
    for expected, step in enumerate(model_input["steps"]):
        if not isinstance(step, dict) or step.get("step_index") != expected:
            raise ValueError(f"model input steps[{expected}] step_index must be continuous from 0")
        if step.get("decision_step_index") is not None and step.get("decision_step_index") != expected:
            raise ValueError(f"model input steps[{expected}] decision_step_index is invalid")
        events = step.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"model input steps[{expected}].events must be a list")
        for event_expected, event in enumerate(events):
            if not isinstance(event, dict) or event.get("event_index") != event_expected:
                raise ValueError(f"model input steps[{expected}].events event_index must be continuous from 0")
