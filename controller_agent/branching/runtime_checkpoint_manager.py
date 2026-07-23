"""Runtime manager for dense AppWorld checkpoint collection."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from controller_agent.branching.checkpoint_storage import CheckpointStorage
from controller_agent.branching.namespace_adapter import NamespaceAdapter, ReplayRequiredNamespaceAdapter
from controller_agent.branching.schemas import (
    RuntimePostObservationCheckpoint,
    RuntimePreActionRecord,
    RuntimeTerminalRecord,
)


WRITE_OPERATION_PATTERN = re.compile(
    r"\b(create|update|delete|send|post|patch|approve|deny|mark|add|remove|withdraw|complete_task)\b",
    re.IGNORECASE,
)


class RuntimeCheckpointManager:
    """Create runtime checkpoint records and persist them through storage."""

    def __init__(
        self,
        root_dir: Path,
        run_id: str,
        task_id: str,
        namespace_adapter: NamespaceAdapter | None = None,
        appworld_code_version: str | None = None,
        appworld_data_version: str | None = None,
    ) -> None:
        self.storage = CheckpointStorage(root_dir, run_id, task_id)
        self.task_id = task_id
        self.namespace_adapter = namespace_adapter or ReplayRequiredNamespaceAdapter()
        self.appworld_code_version = appworld_code_version
        self.appworld_data_version = appworld_data_version
        self.latest_post_observation_checkpoint_id: str | None = None
        self.executed_code_prefix: list[str] = []
        self.pre_action_count = 0
        self.post_observation_count = 0

    def capture_pre_action(
        self,
        *,
        proposed_execution: str,
        agent: Any,
        last_execution_outputs: list[Any],
        previous_execution_had_error: bool,
        agent_turn_index: int,
    ) -> RuntimePreActionRecord:
        """Capture lightweight pre-action metadata."""
        self.pre_action_count += 1
        record = RuntimePreActionRecord(
            record_id=f"{self.task_id}_pre_{self.pre_action_count:04d}",
            task_id=self.task_id,
            agent_turn_index=agent_turn_index,
            proposed_execution=proposed_execution,
            messages=self._messages(agent),
            last_execution_outputs=copy.deepcopy(last_execution_outputs),
            latest_post_observation_checkpoint_id=self.latest_post_observation_checkpoint_id,
            whether_contains_complete_task="complete_task" in proposed_execution,
            whether_contains_probable_write_operation=bool(WRITE_OPERATION_PATTERN.search(proposed_execution)),
            previous_execution_had_error=previous_execution_had_error,
            timestamp=self._timestamp(),
        )
        self.storage.write_pre_action(record)
        return record

    def capture_post_observation(
        self,
        *,
        world: Any,
        agent: Any,
        last_execution_outputs: list[Any],
        executed_code: str,
        agent_turn_index: int,
        environment_execution_index: int,
        cumulative_agent_tokens: dict[str, int],
        cumulative_controller_tokens: dict[str, int],
        cumulative_cost: float | None,
        controller_runtime_state: dict[str, Any],
    ) -> RuntimePostObservationCheckpoint:
        """Capture full checkpoint after environment observation."""
        self.post_observation_count += 1
        checkpoint_id = f"{self.task_id}_post_{self.post_observation_count:04d}"
        state_id = self._save_world_state(world, checkpoint_id)
        self.executed_code_prefix.append(executed_code)
        checkpoint = RuntimePostObservationCheckpoint(
            checkpoint_id=checkpoint_id,
            task_id=self.task_id,
            agent_turn_index=agent_turn_index,
            environment_execution_index=environment_execution_index,
            appworld_state_id=state_id,
            agent_messages=self._messages(agent),
            last_execution_outputs=copy.deepcopy(last_execution_outputs),
            executed_code=executed_code,
            executed_code_prefix=list(self.executed_code_prefix),
            cumulative_agent_tokens=dict(cumulative_agent_tokens),
            cumulative_controller_tokens=dict(cumulative_controller_tokens),
            cumulative_cost=cumulative_cost,
            controller_runtime_state=dict(controller_runtime_state),
            appworld_code_version=self.appworld_code_version,
            appworld_data_version=self.appworld_data_version,
            timestamp=self._timestamp(),
            namespace_restore_mode=(
                "snapshot" if self.namespace_adapter.is_supported() else "replay_required"
            ),
        )
        self.storage.write_post_observation(checkpoint)
        self.latest_post_observation_checkpoint_id = checkpoint_id
        return checkpoint

    def capture_terminal(
        self,
        *,
        before_complete_task: bool,
        complete_task_blocked_by_controller: bool,
        complete_task_entered_world_batch_execute: bool,
        task_completed: bool,
        supervisor_status: Any,
        termination_reason: str | None,
        evaluator_executed: bool,
    ) -> RuntimeTerminalRecord:
        """Capture terminal lifecycle metadata."""
        record = RuntimeTerminalRecord(
            task_id=self.task_id,
            before_complete_task=before_complete_task,
            complete_task_blocked_by_controller=complete_task_blocked_by_controller,
            complete_task_entered_world_batch_execute=complete_task_entered_world_batch_execute,
            task_completed=task_completed,
            supervisor_status=supervisor_status,
            termination_reason=termination_reason,
            evaluator_executed=evaluator_executed,
            timestamp=self._timestamp(),
        )
        self.storage.write_terminal(record)
        return record

    def _messages(self, agent: Any) -> list[dict[str, Any]]:
        """Return a copied Agent message list."""
        target = getattr(agent, "agent", agent)
        messages = getattr(target, "messages", [])
        return copy.deepcopy(messages if isinstance(messages, list) else [])

    def _save_world_state(self, world: Any, checkpoint_id: str) -> str | None:
        """Save AppWorld state if supported."""
        if not hasattr(world, "save_state"):
            return None
        try:
            value = world.save_state(checkpoint_id)
        except TypeError:
            value = world.save_state()
        return None if value is None else str(value)

    def _timestamp(self) -> str:
        """Return an ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

