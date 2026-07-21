"""Canonical Trajectory Schema v1.1.

CanonicalTrajectory is a storage and evaluation object. Model input views are
defined separately in :mod:`agent_error_compaction.views`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

SCHEMA_VERSION = "1.1"


def _serialize(value: Any) -> Any:
    """Recursively convert dataclasses and containers to JSON values."""
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class ActorInfo:
    """Structured actor identity and source-role type."""

    id: str | None
    type: str


@dataclass(frozen=True)
class ToolCall:
    """Structured tool-call envelope with lossless raw action."""

    call_id: str | None
    tool_name: str | None
    arguments: Any
    raw: Any


@dataclass(frozen=True)
class ToolResult:
    """Structured tool-result envelope with lossless raw observation."""

    call_id: str | None
    content: Any
    status: str | None
    error_code: str | None
    raw: Any


@dataclass(frozen=True)
class CanonicalStep:
    """One source turn represented as exactly one v1.1 step."""

    step_index: int
    source_step_index: int
    source_step_id: str | None
    event_type: str
    content: str | None
    raw_step: Any
    token_count: int | None
    actor: ActorInfo
    message_role: str | None
    reasoning: str | None
    tool_call: ToolCall
    tool_result: ToolResult
    decision_step_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class PrimaryErrorAnnotation:
    """The dataset-provided decisive mistake annotation."""

    step_index: int
    source_step_index: int | None
    actor_id: str | None
    reason: str | None
    error_type_original: str | None
    error_type_unified: str | None
    label_semantics: str
    annotation_source: str
    annotation_method: str | None
    confidence: float | None


@dataclass(frozen=True)
class ErrorAnnotations:
    """Source annotations separated from derived annotations."""

    primary_error: PrimaryErrorAnnotation | None
    error_events: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DerivedAnnotations:
    """Reserved derived fields; all are empty at the preprocessing stage."""

    first_error_step: int | None = None
    earliest_detectable_step: int | None = None
    evidence_steps: list[int] = field(default_factory=list)
    provenance: Any = None


@dataclass(frozen=True)
class SourceInfo:
    """Dataset provenance fields."""

    dataset: str
    split: str | None
    record_id: str | None


@dataclass(frozen=True)
class TaskInfo:
    """Task data separated from environment and outcome."""

    task_id: str | None
    instruction: Any
    gold_answer: Any


@dataclass(frozen=True)
class EnvironmentInfo:
    """Execution environment fields."""

    domain: str | None
    tools: Any


@dataclass(frozen=True)
class OutcomeInfo:
    """Normalized outcome with original status retained in raw."""

    status: str
    score: float | None
    termination_reason: str | None
    raw: Any


@dataclass(frozen=True)
class CanonicalTrajectory:
    """A complete v1.1 trajectory for storage and offline evaluation."""

    schema_version: str
    trajectory_id: str
    trajectory_family_id: str | None
    source: SourceInfo
    task: TaskInfo
    environment: EnvironmentInfo
    steps: list[CanonicalStep]
    outcome: OutcomeInfo
    annotations: ErrorAnnotations
    derived_annotations: DerivedAnnotations
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class PrefixSample:
    """A v1.1 trajectory prefix with separated evaluation annotations."""

    prefix_id: str
    trajectory_id: str
    trajectory_family_id: str | None
    source_split: str | None
    prefix_end_step: int
    steps: list[CanonicalStep]
    critical_error_occurred: bool
    critical_error_step: int | None
    distance_to_critical_error: int | None
    responsible_actor: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


def _actor_from_dict(value: dict[str, Any]) -> ActorInfo:
    """Restore an actor object."""
    return ActorInfo(**value)


def _tool_call_from_dict(value: dict[str, Any]) -> ToolCall:
    """Restore a tool-call object."""
    return ToolCall(**value)


def _tool_result_from_dict(value: dict[str, Any]) -> ToolResult:
    """Restore a tool-result object."""
    return ToolResult(**value)


def _step_from_dict(value: dict[str, Any]) -> CanonicalStep:
    """Restore a canonical step from JSON."""
    payload = dict(value)
    payload["actor"] = _actor_from_dict(payload["actor"])
    payload["tool_call"] = _tool_call_from_dict(payload["tool_call"])
    payload["tool_result"] = _tool_result_from_dict(payload["tool_result"])
    return CanonicalStep(**payload)


def trajectory_from_dict(payload: dict[str, Any]) -> CanonicalTrajectory:
    """Restore a v1.1 canonical trajectory from JSON."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {SCHEMA_VERSION}, got {payload.get('schema_version')!r}")
    return CanonicalTrajectory(
        schema_version=payload["schema_version"],
        trajectory_id=payload["trajectory_id"],
        trajectory_family_id=payload.get("trajectory_family_id"),
        source=SourceInfo(**payload["source"]),
        task=TaskInfo(**payload["task"]),
        environment=EnvironmentInfo(**payload["environment"]),
        steps=[_step_from_dict(step) for step in payload["steps"]],
        outcome=OutcomeInfo(**payload["outcome"]),
        annotations=ErrorAnnotations(
            primary_error=PrimaryErrorAnnotation(**payload["annotations"]["primary_error"]) if payload["annotations"].get("primary_error") is not None else None,
            error_events=payload["annotations"].get("error_events", []),
        ),
        derived_annotations=DerivedAnnotations(**payload.get("derived_annotations", {})),
        metadata=payload.get("metadata", {}),
    )


def prefix_from_dict(payload: dict[str, Any]) -> PrefixSample:
    """Restore a v1.1 prefix sample from JSON."""
    return PrefixSample(
        prefix_id=payload["prefix_id"],
        trajectory_id=payload["trajectory_id"],
        trajectory_family_id=payload.get("trajectory_family_id"),
        source_split=payload.get("source_split"),
        prefix_end_step=payload["prefix_end_step"],
        steps=[_step_from_dict(step) for step in payload["steps"]],
        critical_error_occurred=payload["critical_error_occurred"],
        critical_error_step=payload.get("critical_error_step"),
        distance_to_critical_error=payload.get("distance_to_critical_error"),
        responsible_actor=payload.get("responsible_actor"),
        metadata=payload.get("metadata", {}),
    )

