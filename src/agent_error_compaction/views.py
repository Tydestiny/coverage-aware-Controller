"""Leakage-safe model and evaluation views for Canonical Schema v1.1.

``CanonicalTrajectory`` is a storage and evaluation object. ``ModelInputView``
is the only valid input to future Compactors and Verifiers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

MODEL_STEP_FIELDS = (
    "step_index",
    "decision_step_index",
    "actor",
    "message_role",
    "event_type",
    "content",
    "reasoning",
    "tool_call",
    "tool_result",
)

FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "schema_version",
        "trajectory_id",
        "trajectory_family_id",
        "source",
        "source_split",
        "environment",
        "gold_answer",
        "outcome",
        "annotations",
        "derived_annotations",
        "metadata",
        "labels",
        "is_failure",
        "error_onset_step",
        "source_error_onset_step",
        "critical_error_step",
        "responsible_actor",
        "error_reason",
        "error_type_original",
        "error_type_unified",
        "label_source",
        "unsafe_source",
        "raw_step",
    }
)


def _validate_prefix(trajectory: dict[str, Any], prefix_end_step: int) -> list[Any]:
    """Validate a v1.1 trajectory and return its steps."""
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
    """Build a whitelist-only model input from a v1.1 trajectory prefix."""
    steps = _validate_prefix(trajectory, prefix_end_step)
    task = trajectory.get("task")
    environment = trajectory.get("environment")
    if not isinstance(task, dict) or "instruction" not in task:
        raise ValueError("trajectory.task.instruction is required for model input")
    if not isinstance(environment, dict):
        raise ValueError("trajectory.environment is required for model input")
    model_input = {
        "task": deepcopy(task["instruction"]),
        "tools": deepcopy(environment.get("tools", [])),
        "steps": [
            {field: deepcopy(step[field]) for field in MODEL_STEP_FIELDS}
            for step in steps[: prefix_end_step + 1]
            if isinstance(step, dict) and all(field in step for field in MODEL_STEP_FIELDS)
        ],
    }
    if len(model_input["steps"]) != prefix_end_step + 1:
        raise ValueError("trajectory.steps must contain complete v1.1 step dictionaries")
    validate_model_input_view(model_input)
    return model_input


def build_evaluation_view(trajectory: dict[str, Any], prefix_end_step: int) -> dict[str, Any]:
    """Build an offline evaluation view from v1.1 primary-error annotations."""
    _validate_prefix(trajectory, prefix_end_step)
    annotations = trajectory.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("trajectory.annotations must be a dictionary")
    primary = annotations.get("primary_error")
    if primary is not None and not isinstance(primary, dict):
        raise ValueError("annotations.primary_error must be an object or null")
    critical_step = primary.get("step_index") if primary is not None else None
    occurred = critical_step is not None and prefix_end_step >= critical_step
    source = trajectory.get("source")
    if not isinstance(source, dict):
        raise ValueError("trajectory.source must be a dictionary")
    return {
        "trajectory_id": deepcopy(trajectory.get("trajectory_id")),
        "trajectory_family_id": deepcopy(trajectory.get("trajectory_family_id")),
        "source": {"split": deepcopy(source.get("split"))},
        "prefix_end_step": prefix_end_step,
        "critical_error_occurred": occurred,
        "critical_error_step": critical_step,
        "distance_to_critical_error": prefix_end_step - critical_step if critical_step is not None else None,
        "responsible_actor": deepcopy(primary.get("actor_id")) if primary is not None else None,
        "error_reason": deepcopy(primary.get("reason")) if primary is not None else None,
        "annotation_method": deepcopy(primary.get("annotation_method")) if primary is not None else None,
    }


def _walk(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield every dictionary key and its fully qualified path."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            yield str(key), key_path
            yield from _walk(child, key_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_model_input_view(model_input: dict[str, Any]) -> None:
    """Validate v1.1 model-input requirements and recursively reject labels."""
    if not isinstance(model_input, dict):
        raise ValueError("model input must be a dictionary")
    for field, path in _walk(model_input):
        if field in FORBIDDEN_MODEL_FIELDS:
            raise ValueError(f"Forbidden field found in model input: {path}")
    if "task" not in model_input:
        raise ValueError("model input must contain task")
    if "steps" not in model_input:
        raise ValueError("model input must contain steps")
    if not isinstance(model_input["steps"], list):
        raise ValueError("model input steps must be a list")
    for expected_index, step in enumerate(model_input["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"model input steps[{expected_index}] must be a dictionary")
        if step.get("step_index") != expected_index or step.get("decision_step_index") != expected_index:
            raise ValueError("model input step indices must be continuous from 0")

