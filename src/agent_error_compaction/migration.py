"""Streaming migration from Canonical Trajectory v1.0 to v1.1."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterator

from .io_utils import read_records
from .schemas import (
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


def _actor_type(actor_id: str | None) -> str:
    """Classify a source actor without semantic inference."""
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


def _actor(value: Any, role: Any) -> ActorInfo:
    """Convert a v1.0 actor string/object with role fallback."""
    if isinstance(value, dict):
        actor_id = value.get("id")
        actor_type = value.get("type") or _actor_type(actor_id)
        return ActorInfo(actor_id, actor_type)
    actor_id = value if isinstance(value, str) and value else role if isinstance(role, str) and role else None
    return ActorInfo(actor_id, _actor_type(actor_id))


def _tool_call(value: Any) -> ToolCall:
    """Wrap the exact v1.0 action as tool_call.raw without guessing fields."""
    return ToolCall(None, None, None, deepcopy(value))


def _tool_result(value: Any) -> ToolResult:
    """Wrap the exact v1.0 observation as tool_result.raw without guessing fields."""
    return ToolResult(None, None, None, None, deepcopy(value))


def _step(old_step: dict[str, Any], index: int) -> CanonicalStep:
    """Migrate one v1.0 step while preserving its raw object."""
    role = old_step.get("role")
    return CanonicalStep(
        step_index=old_step["step_index"],
        source_step_index=old_step["source_step_index"],
        source_step_id=old_step.get("source_step_id"),
        event_type=old_step.get("event_type", "message"),
        content=old_step.get("content"),
        raw_step=deepcopy(old_step["raw_step"]),
        token_count=old_step.get("token_count"),
        actor=_actor(old_step.get("actor"), role),
        message_role=role,
        reasoning=old_step.get("thought"),
        tool_call=_tool_call(old_step.get("action")),
        tool_result=_tool_result(old_step.get("observation")),
        decision_step_index=old_step["step_index"],
    )


def migrate_record(old: dict[str, Any]) -> CanonicalTrajectory:
    """Migrate one v1.0 canonical record to v1.1."""
    labels = old.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"trajectory_id={old.get('trajectory_id')!r}: labels must be an object")
    status = old.get("outcome")
    if isinstance(status, dict):
        status = status.get("status")
    if status not in {"success", "failure"}:
        raise ValueError(f"trajectory_id={old.get('trajectory_id')!r}: unsupported v1.0 outcome")
    primary = None
    if labels.get("is_failure"):
        step_index = labels.get("error_onset_step")
        if not isinstance(step_index, int):
            raise ValueError(f"trajectory_id={old.get('trajectory_id')!r}: failure lacks error step")
        unsafe_source = labels.get("unsafe_source")
        method = {"injected": "controlled_injection", "diagnosed": "post_hoc_diagnosis"}.get(unsafe_source, unsafe_source)
        primary = PrimaryErrorAnnotation(
            step_index=step_index,
            source_step_index=labels.get("source_error_onset_step"),
            actor_id=labels.get("responsible_actor"),
            reason=labels.get("error_reason"),
            error_type_original=labels.get("error_type_original"),
            error_type_unified=labels.get("error_type_unified"),
            label_semantics="decisive_mistake_step",
            annotation_source="dataset_author",
            annotation_method=method,
            confidence=None,
        )
    steps = [_step(step, index) for index, step in enumerate(old.get("steps", []))]
    old_task = old.get("task")
    old_domain = old.get("domain")
    old_tools = old.get("tools")
    old_metadata = old.get("metadata") if isinstance(old.get("metadata"), dict) else {}
    return CanonicalTrajectory(
        schema_version=SCHEMA_VERSION,
        trajectory_id=old["trajectory_id"],
        trajectory_family_id=old.get("trajectory_family_id"),
        source=SourceInfo(old.get("source_dataset", "AFTraj-2K"), old.get("source_split"), old["trajectory_id"]),
        task=TaskInfo(old.get("task_id"), deepcopy(old_task), deepcopy(old.get("gold_answer"))),
        environment=EnvironmentInfo(deepcopy(old_domain), deepcopy(old_tools)),
        steps=steps,
        outcome=OutcomeInfo(status, None, None, deepcopy(status)),
        annotations=ErrorAnnotations(primary, []),
        derived_annotations=DerivedAnnotations(),
        metadata={"migration": {"from_schema_version": "1.0", "method": "deterministic_field_mapping"}, "num_steps": len(steps), "source_turn_count": len(steps), "legacy_metadata": old_metadata},
    )


def migrate_file(input_path: Path, output_path: Path, rejected_path: Path) -> dict[str, int]:
    """Stream-migrate a v1.0 JSONL file and record rejected records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "migrated": 0, "rejected": 0}
    with output_path.open("w", encoding="utf-8", newline="\n") as output, rejected_path.open("w", encoding="utf-8", newline="\n") as rejected:
        for index, old in enumerate(read_records(input_path)):
            stats["total"] += 1
            try:
                migrated = migrate_record(old)
            except Exception as exc:
                rejected.write(json.dumps({"record_index": index, "reason": str(exc), "raw_record": old}, ensure_ascii=False) + "\n")
                stats["rejected"] += 1
                continue
            output.write(json.dumps(migrated.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            stats["migrated"] += 1
    return stats

