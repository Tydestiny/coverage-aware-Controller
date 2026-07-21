"""Canonical Trajectory Schema v1.1 data classes."""

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
    """Structured actor identity."""

    id: str | None
    type: str


@dataclass(frozen=True)
class ToolCall:
    """Structured tool-call payload."""

    call_id: str | None
    tool_name: str | None
    arguments: Any
    raw: Any


@dataclass(frozen=True)
class ToolResult:
    """Structured tool-result payload."""

    call_id: str | None
    content: Any
    status: str | None
    error_code: str | None
    raw: Any


@dataclass(frozen=True)
class CanonicalEvent:
    """One ordered event within a canonical step."""

    event_index: int
    event_type: str
    actor: ActorInfo
    content: Any = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    source_event_index: int | None = None
    source_event_id: str | None = None
    raw_event: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class CanonicalStep:
    """A canonical decision step containing zero or more ordered events."""

    step_index: int
    source_step_index: int | None
    source_step_id: str | None
    source_unit_type: str
    decision_step_index: int | None
    actor: ActorInfo
    message_role: str | None
    event_type: str
    content: Any
    reasoning: Any
    events: list[CanonicalEvent]
    raw_step: Any
    token_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class StepSpan:
    """Optional source span for an error annotation."""

    start_step_index: int
    end_step_index: int


@dataclass(frozen=True)
class PrimaryErrorAnnotation:
    """Dataset-provided primary error annotation."""

    step_index: int | None
    source_step_index: int | None
    actor_id: str | None
    reason: str | None
    error_type_original: str | None
    error_type_unified: str | None
    label_semantics: str
    annotation_source: str
    annotation_method: str | None
    confidence: float | None
    alignment_status: str = "exact"
    step_span: StepSpan | None = None


@dataclass(frozen=True)
class ErrorAnnotations:
    """Source annotations and their availability status."""

    annotation_status: str
    primary_error: PrimaryErrorAnnotation | None
    error_events: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DerivedField:
    """A derived annotation with explicit non-annotation status."""

    value: Any = None
    status: str = "not_annotated"
    provenance: Any = None


@dataclass(frozen=True)
class DerivedAnnotations:
    """Derived fields reserved for later processing."""

    first_error_step: DerivedField = field(default_factory=DerivedField)
    earliest_detectable_step: DerivedField = field(default_factory=DerivedField)
    evidence_steps: DerivedField = field(default_factory=lambda: DerivedField(value=[]))


@dataclass(frozen=True)
class SourceInfo:
    """Original dataset provenance."""

    dataset: str
    original_split: str | None
    record_id: str | None


@dataclass(frozen=True)
class TaskInfo:
    """Task instruction and answer fields."""

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
    """Normalized outcome with original value retained."""

    status: str
    score: float | None
    termination_reason: str | None
    raw: Any


@dataclass(frozen=True)
class CanonicalTrajectory:
    """Complete storage and evaluation object."""

    schema_version: str
    trajectory_id: str
    trajectory_family_id: str | None
    source: SourceInfo
    experiment_split: str | None
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
    """A prefix and evaluation-only critical-error labels."""

    prefix_id: str
    trajectory_id: str
    trajectory_family_id: str | None
    experiment_split: str | None
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


def _actor(value: dict[str, Any]) -> ActorInfo:
    return ActorInfo(**value)


def _event(value: dict[str, Any]) -> CanonicalEvent:
    payload = dict(value)
    payload["actor"] = _actor(payload["actor"])
    if payload.get("tool_call") is not None:
        payload["tool_call"] = ToolCall(**payload["tool_call"])
    if payload.get("tool_result") is not None:
        payload["tool_result"] = ToolResult(**payload["tool_result"])
    return CanonicalEvent(**payload)


def _step(value: dict[str, Any]) -> CanonicalStep:
    payload = dict(value)
    payload["actor"] = _actor(payload["actor"])
    payload["events"] = [_event(event) for event in payload.get("events", [])]
    return CanonicalStep(**payload)


def _derived_field(value: Any, default: Any = None) -> DerivedField:
    if isinstance(value, dict):
        return DerivedField(**value)
    return DerivedField(value=value if value is not None else default)


def trajectory_from_dict(payload: dict[str, Any]) -> CanonicalTrajectory:
    """Restore a v1.1 canonical trajectory from JSON."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {SCHEMA_VERSION}, got {payload.get('schema_version')!r}")
    annotation = payload["annotations"]
    primary = annotation.get("primary_error")
    if primary is not None and primary.get("step_span") is not None:
        primary = {**primary, "step_span": StepSpan(**primary["step_span"])}
    derived = payload.get("derived_annotations", {})
    return CanonicalTrajectory(
        schema_version=payload["schema_version"], trajectory_id=payload["trajectory_id"],
        trajectory_family_id=payload.get("trajectory_family_id"), source=SourceInfo(**payload["source"]),
        experiment_split=payload.get("experiment_split"), task=TaskInfo(**payload["task"]),
        environment=EnvironmentInfo(**payload["environment"]), steps=[_step(s) for s in payload["steps"]],
        outcome=OutcomeInfo(**payload["outcome"]),
        annotations=ErrorAnnotations(annotation_status=annotation.get("annotation_status", "not_provided"),
                                     primary_error=PrimaryErrorAnnotation(**primary) if primary else None,
                                     error_events=annotation.get("error_events", [])),
        derived_annotations=DerivedAnnotations(
            first_error_step=_derived_field(derived.get("first_error_step")),
            earliest_detectable_step=_derived_field(derived.get("earliest_detectable_step")),
            evidence_steps=_derived_field(derived.get("evidence_steps"), [])),
        metadata=payload.get("metadata", {}),
    )


def prefix_from_dict(payload: dict[str, Any]) -> PrefixSample:
    """Restore a v1.1 prefix sample from JSON."""
    return PrefixSample(prefix_id=payload["prefix_id"], trajectory_id=payload["trajectory_id"],
                        trajectory_family_id=payload.get("trajectory_family_id"),
                        experiment_split=payload.get("experiment_split"),
                        prefix_end_step=payload["prefix_end_step"], steps=[_step(s) for s in payload["steps"]],
                        critical_error_occurred=payload["critical_error_occurred"],
                        critical_error_step=payload.get("critical_error_step"),
                        distance_to_critical_error=payload.get("distance_to_critical_error"),
                        responsible_actor=payload.get("responsible_actor"), metadata=payload.get("metadata", {}))
