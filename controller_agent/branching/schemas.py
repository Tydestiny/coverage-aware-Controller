"""Data structures for static and runtime checkpoint handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TrajectoryStep:
    """One static Agent execution and environment observation."""

    step_index: int
    execution_code: str | None
    execution_output: str
    messages_before_step: list[dict[str, Any]]
    messages_after_step: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticTrajectory:
    """Loaded static trajectory with replayable execution steps."""

    task_id: str
    steps: list[TrajectoryStep]
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructedCheckpoint:
    """Checkpoint reconstructed by replaying a static trajectory prefix."""

    task_id: str
    checkpoint_step: int
    environment_state_id: Any
    agent_messages: list[dict[str, Any]]
    last_execution_outputs: list[Any]
    agent_turn_count: int
    environment_execution_count: int
    source_trajectory: str
    validation_status: str
    validation_errors: list[str] = field(default_factory=list)


class CheckpointCaptureHook(Protocol):
    """Future online checkpoint capture hook; not wired into the runner."""

    def capture(self, world: Any, agent_state: Any, step_index: int) -> ReconstructedCheckpoint:
        """Capture a checkpoint from an online run."""


@dataclass(frozen=True)
class RuntimePostObservationCheckpoint:
    """Dense checkpoint captured after environment observation."""

    checkpoint_id: str
    task_id: str
    agent_turn_index: int
    environment_execution_index: int
    appworld_state_id: str | None
    agent_messages: list[dict[str, Any]]
    last_execution_outputs: list[Any]
    executed_code: str
    executed_code_prefix: list[str]
    cumulative_agent_tokens: dict[str, int]
    cumulative_controller_tokens: dict[str, int]
    cumulative_cost: float | None
    controller_runtime_state: dict[str, Any]
    appworld_code_version: str | None
    appworld_data_version: str | None
    timestamp: str
    phase: str = "post_observation"
    namespace_restore_mode: str = "replay_required"


@dataclass(frozen=True)
class RuntimePreActionRecord:
    """Lightweight record captured after Agent proposes execution."""

    record_id: str
    task_id: str
    agent_turn_index: int
    proposed_execution: str
    messages: list[dict[str, Any]]
    last_execution_outputs: list[Any]
    latest_post_observation_checkpoint_id: str | None
    whether_contains_complete_task: bool
    whether_contains_probable_write_operation: bool
    previous_execution_had_error: bool
    timestamp: str
    phase: str = "pre_action"


@dataclass(frozen=True)
class RuntimeTerminalRecord:
    """Terminal task lifecycle record."""

    task_id: str
    before_complete_task: bool
    complete_task_blocked_by_controller: bool
    complete_task_entered_world_batch_execute: bool
    task_completed: bool
    supervisor_status: Any
    termination_reason: str | None
    evaluator_executed: bool
    timestamp: str
    phase: str = "terminal"
