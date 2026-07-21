"""Canonical, JSON-serializable data schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def _serialize(value: Any) -> Any:
    """Recursively convert dataclasses and containers to JSON-compatible values."""
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalStep:
    """One source turn preserved as exactly one canonical step."""

    step_index: int
    source_step_index: int
    source_step_id: str | None
    actor: str | None
    role: str | None
    event_type: str
    content: str | None
    thought: str | None
    action: Any
    observation: Any
    raw_step: Any
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class TrajectoryLabels:
    """Source labels without inferred or rewritten error annotations."""

    is_failure: bool
    error_onset_step: int | None
    source_error_onset_step: int | None
    responsible_actor: str | None
    error_type_original: str | None
    error_type_unified: str | None
    error_reason: str | None
    label_source: str | None
    unsafe_source: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class CanonicalTrajectory:
    """A fully normalized trajectory with lossless source-step retention."""

    trajectory_id: str
    trajectory_family_id: str | None
    source_dataset: str
    source_split: str | None
    domain: str | None
    task_id: str | None
    task: Any
    gold_answer: Any
    outcome: str
    tools: Any
    steps: list[CanonicalStep]
    labels: TrajectoryLabels
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


@dataclass(frozen=True)
class PrefixSample:
    """A trajectory prefix containing no steps after its endpoint."""

    prefix_id: str
    trajectory_id: str
    trajectory_family_id: str | None
    source_split: str | None
    prefix_end_step: int
    steps: list[CanonicalStep]
    error_occurred: bool
    distance_to_error_onset: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _serialize(self)


def trajectory_from_dict(payload: dict[str, Any]) -> CanonicalTrajectory:
    """Restore a canonical trajectory from a JSON-compatible dictionary."""
    steps = [CanonicalStep(**step) for step in payload["steps"]]
    labels = TrajectoryLabels(**payload["labels"])
    values = dict(payload)
    values["steps"] = steps
    values["labels"] = labels
    return CanonicalTrajectory(**values)


def prefix_from_dict(payload: dict[str, Any]) -> PrefixSample:
    """Restore a prefix sample from a JSON-compatible dictionary."""
    values = dict(payload)
    values["steps"] = [CanonicalStep(**step) for step in payload["steps"]]
    return PrefixSample(**values)
