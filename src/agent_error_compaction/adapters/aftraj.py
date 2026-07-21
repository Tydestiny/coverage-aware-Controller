"""Defensive AFTraj-to-Canonical-Trajectory v1.1 adapter."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseTrajectoryAdapter
from ..schemas import (
    ActorInfo,
    CanonicalStep,
    CanonicalTrajectory,
    DerivedAnnotations,
    EnvironmentInfo,
    ErrorAnnotations,
    OutcomeInfo,
    PrimaryErrorAnnotation,
    SCHEMA_VERSION,
    SourceInfo,
    TaskInfo,
    ToolCall,
    ToolResult,
)


def _as_text(value: Any, field: str) -> str | None:
    """Convert common scalar/object text forms without discarding source data."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("content", "text", "message", "value", "name"):
            if isinstance(value.get(key), str):
                return value[key]
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"{field} has unsupported type {type(value).__name__}")


def _has_value(value: Any) -> bool:
    """Return whether a source field contains a meaningful value."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _actor_type(actor_id: str | None) -> str:
    """Classify actors using only explicit source strings."""
    normalized = (actor_id or "").strip().lower()
    if normalized == "environment":
        return "environment"
    if normalized == "tool":
        return "tool"
    if normalized == "user":
        return "user"
    if actor_id:
        return "agent"
    return "unknown"


def _parse_tool_call(action: Any) -> ToolCall:
    """Extract only explicitly named tool-call fields while preserving raw action."""
    parsed: Any = action
    if isinstance(action, str) and action.strip():
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return ToolCall(None, None, None, action)
    call_id = parsed.get("call_id", parsed.get("id"))
    tool_name = parsed.get("tool_name", parsed.get("name"))
    arguments = parsed.get("arguments", parsed.get("args"))
    return ToolCall(_as_text(call_id, "call_id"), _as_text(tool_name, "tool_name"), arguments, action)


def _parse_tool_result(observation: Any) -> ToolResult:
    """Extract only explicitly named tool-result fields while preserving raw observation."""
    parsed: Any = observation
    if isinstance(observation, str) and observation.strip():
        try:
            parsed = json.loads(observation)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return ToolResult(None, observation, None, None, observation)
    return ToolResult(
        _as_text(parsed.get("call_id"), "call_id"),
        parsed.get("content", parsed.get("result", observation)),
        _as_text(parsed.get("status"), "status"),
        _as_text(parsed.get("error_code"), "error_code"),
        observation,
    )


def _infer_event_type(turn: dict[str, Any]) -> str:
    """Classify a turn from explicit fields and source role only."""
    explicit = turn.get("event_type") or turn.get("type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if _has_value(turn.get("action")):
        return "action"
    if _has_value(turn.get("observation")) or str(turn.get("role", "")).lower() == "environment":
        return "observation"
    return "message"


class AFTrajAdapter(BaseTrajectoryAdapter):
    """Convert confirmed 0-based AFTraj records to schema v1.1."""

    def __init__(self, mistake_step_base: int = 0, source_dataset: str = "AFTraj-2K") -> None:
        """Configure the confirmed AFTraj source indexing convention."""
        if mistake_step_base != 0:
            raise ValueError("AFTraj-2K mistake_step_base is fixed at 0")
        self.mistake_step_base = mistake_step_base
        self.source_dataset = source_dataset

    def convert_record(self, record: dict[str, Any]) -> CanonicalTrajectory:
        """Convert one AFTraj record without inferring new annotations."""
        trajectory_id = _as_text(record.get("conv_id"), "conv_id")
        if not trajectory_id:
            raise ValueError("conv_id must be a non-empty string-like value")
        turns = record.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"trajectory {trajectory_id}: turns must be a list")
        steps: list[CanonicalStep] = []
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"trajectory {trajectory_id}: turns[{index}] must be an object")
            source_index = turn.get("step_index", turn.get("turn_index", turn.get("index", index)))
            if not isinstance(source_index, int) or isinstance(source_index, bool):
                raise ValueError(f"trajectory {trajectory_id}: source step index must be an integer")
            actor_id = _as_text(next((turn[key] for key in ("actor", "agent", "name", "sender", "role") if key in turn), None), "actor")
            role = _as_text(turn.get("role"), "role")
            action = turn.get("action")
            observation = turn.get("observation", turn.get("tool_result"))
            steps.append(
                CanonicalStep(
                    step_index=index,
                    source_step_index=source_index,
                    source_step_id=_as_text(next((turn[key] for key in ("step_id", "turn_id", "id") if key in turn), None), "source_step_id"),
                    event_type=_infer_event_type(turn),
                    content=_as_text(next((turn[key] for key in ("content", "message", "text") if key in turn), None), "content"),
                    raw_step=turn,
                    token_count=turn.get("token_count"),
                    actor=ActorInfo(actor_id, _actor_type(actor_id)),
                    message_role=role,
                    reasoning=_as_text(next((turn[key] for key in ("thought", "reasoning", "analysis") if key in turn), None), "thought"),
                    tool_call=_parse_tool_call(action),
                    tool_result=_parse_tool_result(observation),
                    decision_step_index=index,
                )
            )

        source_step = record.get("mistake_step")
        if source_step is not None and (not isinstance(source_step, int) or isinstance(source_step, bool)):
            raise ValueError(f"trajectory {trajectory_id}: mistake_step must be int or null")
        explicit_outcome = record.get("outcome")
        if explicit_outcome is not None and explicit_outcome not in {"success", "failure"}:
            raise ValueError(f"trajectory {trajectory_id}: outcome must be success or failure")
        is_failure = source_step is not None if explicit_outcome is None else explicit_outcome == "failure"
        if not is_failure and source_step is not None:
            raise ValueError(f"trajectory {trajectory_id}: success outcome conflicts with mistake_step")
        primary_error = None
        unsafe_source = _as_text(record.get("unsafe_source"), "unsafe_source") or ("safe" if not is_failure else None)
        if is_failure:
            primary_error = PrimaryErrorAnnotation(
                step_index=source_step,
                source_step_index=source_step,
                actor_id=_as_text(record.get("mistake_agent"), "mistake_agent"),
                reason=_as_text(record.get("mistake_reason"), "mistake_reason"),
                error_type_original=_as_text(record.get("mistake_type"), "mistake_type"),
                error_type_unified=None,
                label_semantics="decisive_mistake_step",
                annotation_source="dataset_author",
                annotation_method={"injected": "controlled_injection", "diagnosed": "post_hoc_diagnosis"}.get(unsafe_source, unsafe_source),
                confidence=None,
            )
        return CanonicalTrajectory(
            schema_version=SCHEMA_VERSION,
            trajectory_id=trajectory_id,
            trajectory_family_id=None,
            source=SourceInfo(self.source_dataset, _as_text(record.get("split"), "split"), trajectory_id),
            task=TaskInfo(_as_text(record.get("task_id"), "task_id"), record.get("task"), record.get("gold_answer")),
            environment=EnvironmentInfo(_as_text(record.get("domain"), "domain"), record.get("tools")),
            steps=steps,
            outcome=OutcomeInfo("failure" if is_failure else "success", None, None, "failure" if is_failure else "success"),
            annotations=ErrorAnnotations(primary_error, []),
            derived_annotations=DerivedAnnotations(),
            metadata={"num_steps": len(steps), "source_turn_count": len(turns), "source_unsafe_source": unsafe_source, "source_record": {key: record[key] for key in ("trajectory_family_id", "parent_trajectory_id", "parent_id", "related_trajectory_id", "family_id") if key in record}},
        )
