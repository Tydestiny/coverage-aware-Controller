"""Tests for Python namespace reconstruction in branch restores."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from controller_agent.branching.namespace_restore import restore_python_namespace_from_prefix
from scripts.run_lightweight_branch_runs import validate_pre_action_anchor


class FakeRequester:
    """Minimal requester exposing AppWorld's reset method."""

    def __init__(self) -> None:
        self.reset_count = 0

    def reset_requests(self) -> None:
        """Record that API request logs were reset."""
        self.reset_count += 1


class FakeWorld:
    """Minimal AppWorld-like object used to verify replay behavior."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.outputs: list[str] = []
        self.environment_io: list[dict[str, Any]] = [{"input": "old", "output": "old"}]
        self.num_interactions = 7
        self.num_sub_interactions = 3
        self.requester = FakeRequester()
        self.save_logs_count = 0

    def execute(self, code: str) -> str:
        """Record replayed code and simulate AppWorld logging."""
        self.executed.append(code)
        self.environment_io.append({"input": code, "output": "ok"})
        self.num_interactions += 1
        if self.outputs:
            return self.outputs.pop(0)
        return "ok"

    def save_logs(self) -> None:
        """Record that logs were flushed after cleanup."""
        self.save_logs_count += 1


def test_rebuild_python_namespace_replays_prefix_and_clears_restore_logs() -> None:
    """Replay-required checkpoints rebuild the REPL namespace from code prefix."""
    world = FakeWorld()
    checkpoint = {
        "namespace_restore_mode": "replay_required",
        "executed_code_prefix": ["login_result = {'access_token': 'x'}", "access_token = login_result['access_token']"],
    }

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is True
    assert result.mode == "replay_required"
    assert result.steps == 2
    assert result.error is None
    assert world.executed == checkpoint["executed_code_prefix"]
    assert world.environment_io == []
    assert world.num_interactions == 0
    assert world.num_sub_interactions == 0
    assert world.requester.reset_count == 1
    assert world.save_logs_count == 1


def test_rebuild_python_namespace_skips_non_replay_checkpoint() -> None:
    """Snapshot-capable checkpoints do not replay code prefix."""
    world = FakeWorld()
    checkpoint = {
        "namespace_restore_mode": "snapshot",
        "executed_code_prefix": ["should_not_execute()"],
    }

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is True
    assert result.mode == "snapshot"
    assert result.steps == 0
    assert result.error is None
    assert world.executed == []


def test_rebuild_python_namespace_reports_missing_prefix() -> None:
    """Replay-required checkpoints fail cleanly when the prefix is absent."""
    world = FakeWorld()
    checkpoint = {"namespace_restore_mode": "replay_required"}

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is False
    assert result.steps == 0
    assert "missing executed_code_prefix" in str(result.error)
    assert world.executed == []


def test_rebuild_python_namespace_reports_invalid_prefix() -> None:
    """Replay-required checkpoints must carry a list code prefix."""
    world = FakeWorld()
    checkpoint = {
        "namespace_restore_mode": "replay_required",
        "executed_code_prefix": "not-a-list",
    }

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is False
    assert "executed_code_prefix" in str(result.error)


def test_rebuild_python_namespace_continues_after_execution_failure() -> None:
    """Replay continues so a later retry can reconstruct required variables."""
    world = FakeWorld()
    world.outputs = ["ok", "Execution failed. Traceback:\nNameError: missing"]
    checkpoint = {
        "namespace_restore_mode": "replay_required",
        "executed_code_prefix": ["x = 1", "y = missing"],
    }

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is True
    assert result.steps == 2
    assert result.error is None
    assert result.replay_warnings == ("executed_code_prefix[1]: Execution failed. Traceback:",)
    assert world.executed == ["x = 1", "y = missing"]
    assert world.environment_io == []


def test_rebuild_python_namespace_does_not_modify_checkpoint_record() -> None:
    """Namespace replay reads checkpoint data without rewriting the source record."""
    world = FakeWorld()
    checkpoint = {
        "namespace_restore_mode": "replay_required",
        "executed_code_prefix": ["login_result = {'access_token': 'x'}"],
        "agent_messages": [{"role": "user", "content": "task"}],
    }
    before = copy.deepcopy(checkpoint)

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is True
    assert checkpoint == before


def test_pre_action_anchor_must_reference_selected_post_observation_checkpoint(tmp_path: Path) -> None:
    """A light pre-action record can only anchor a matching post-observation checkpoint."""
    pre_action_dir = tmp_path / "checkpoints" / "pre_action"
    pre_action_dir.mkdir(parents=True)
    record = {
        "record_id": "task_pre_0009",
        "task_id": "task",
        "phase": "pre_action",
        "latest_post_observation_checkpoint_id": "task_post_0008",
    }
    (pre_action_dir / "task_pre_0009.json").write_text(json.dumps(record), encoding="utf-8")

    result = validate_pre_action_anchor(
        tmp_path,
        {"task_id": "task", "checkpoint_id": "task_post_0008", "pre_action_record_id": "task_pre_0009"},
    )

    assert result == {"record_id": "task_pre_0009", "checkpoint_id": "task_post_0008"}


def test_pre_action_anchor_rejects_non_matching_checkpoint(tmp_path: Path) -> None:
    """A pre-action record must never be paired with a different environment state."""
    pre_action_dir = tmp_path / "checkpoints" / "pre_action"
    pre_action_dir.mkdir(parents=True)
    (pre_action_dir / "task_pre_0009.json").write_text(
        json.dumps(
            {
                "record_id": "task_pre_0009",
                "task_id": "task",
                "phase": "pre_action",
                "latest_post_observation_checkpoint_id": "task_post_0008",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not reference selected checkpoint"):
        validate_pre_action_anchor(
            tmp_path,
            {"task_id": "task", "checkpoint_id": "task_post_0007", "pre_action_record_id": "task_pre_0009"},
        )
