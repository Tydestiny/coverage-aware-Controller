"""Tests for Python namespace reconstruction in branch restores."""

from __future__ import annotations

import copy
from typing import Any

from controller_agent.branching.namespace_restore import restore_python_namespace_from_prefix


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


def test_rebuild_python_namespace_reports_execution_failure() -> None:
    """Replay stops before branch execution when a prefix cell fails."""
    world = FakeWorld()
    world.outputs = ["ok", "Execution failed. Traceback:\nNameError: missing"]
    checkpoint = {
        "namespace_restore_mode": "replay_required",
        "executed_code_prefix": ["x = 1", "y = missing"],
    }

    result = restore_python_namespace_from_prefix(world, checkpoint)

    assert result.success is False
    assert result.steps == 2
    assert "executed_code_prefix[1]" in str(result.error)
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
