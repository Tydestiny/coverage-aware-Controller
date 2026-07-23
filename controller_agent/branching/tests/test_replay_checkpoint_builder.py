"""Tests for static trajectory checkpoint reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from controller_agent.branching.checkpoint_validator import CheckpointValidator
from controller_agent.branching.replay_checkpoint_builder import ReplayCheckpointBuilder
from controller_agent.branching.schemas import StaticTrajectory, TrajectoryStep
from controller_agent.branching.trajectory_loader import load_appworld_logger_jsonl


class MockWorld:
    """World-like object that records replayed code without LLM calls."""

    def __init__(self, outputs_by_code: dict[str, str]) -> None:
        self.outputs_by_code = outputs_by_code
        self.executed: list[str] = []
        self.saved_states: dict[str, list[str]] = {}

    def batch_execute(self, codes: list[str]) -> list[str]:
        """Execute code by deterministic lookup."""
        outputs = []
        for code in codes:
            self.executed.append(code)
            outputs.append(self.outputs_by_code.get(code, f"unexpected:{code}"))
        return outputs

    def save_state(self) -> str:
        """Save executed-code state."""
        state_id = f"state-{len(self.saved_states)}"
        self.saved_states[state_id] = list(self.executed)
        return state_id

    def load_state(self, state_id: str) -> None:
        """Restore executed-code state."""
        self.executed = list(self.saved_states[state_id])


def test_normal_trajectory_reconstruction() -> None:
    """Replay matching static steps into a checkpoint."""
    trajectory = make_trajectory()
    world = MockWorld({"print('one')": "one", "print('two')": "two"})
    checkpoint = ReplayCheckpointBuilder(lambda task_id: world).build(trajectory, 1)
    assert checkpoint.validation_status == "valid"
    assert checkpoint.task_id == "task-1"
    assert checkpoint.checkpoint_step == 1
    assert checkpoint.environment_execution_count == 2
    assert checkpoint.agent_turn_count == 2
    assert checkpoint.environment_state_id == "state-0"
    assert checkpoint.last_execution_outputs == ["two"]
    assert world.executed == ["print('one')", "print('two')"]


def test_missing_execution_code_marks_invalid() -> None:
    """Missing code is an invalid trajectory, not a silent replay."""
    trajectory = StaticTrajectory(
        task_id="task-1",
        source_path="mock",
        steps=[
            TrajectoryStep(0, None, "one", [], [{"role": "assistant", "content": "no code"}]),
        ],
    )
    checkpoint = ReplayCheckpointBuilder(lambda task_id: MockWorld({})).build(trajectory, 0)
    assert checkpoint.validation_status == "invalid_trajectory"
    assert "Missing execution_code at step 0" in checkpoint.validation_errors
    assert checkpoint.environment_state_id is None


def test_observation_mismatch_marks_replay_diverged() -> None:
    """Replay stops and records the first divergent step."""
    trajectory = make_trajectory()
    world = MockWorld({"print('one')": "one", "print('two')": "wrong"})
    checkpoint = ReplayCheckpointBuilder(lambda task_id: world).build(trajectory, 1)
    assert checkpoint.validation_status == "replay_diverged"
    assert any("step 1" in error for error in checkpoint.validation_errors)
    assert checkpoint.environment_state_id is None


def test_checkpoint_step_out_of_range() -> None:
    """Invalid checkpoint index raises."""
    with pytest.raises(IndexError):
        ReplayCheckpointBuilder(lambda task_id: MockWorld({})).build(make_trajectory(), 3)


def test_save_load_state_consistency() -> None:
    """Validator checks read-only probe before and after load_state."""
    world = MockWorld({"print('one')": "one", "probe()": "same"})
    state_id = world.save_state()
    errors = CheckpointValidator().validate_save_load_read_only(world, state_id, "probe()")
    assert errors == []


def test_checkpoint_messages_do_not_include_future_steps() -> None:
    """Checkpoint agent history excludes future code and observations."""
    trajectory = make_trajectory()
    checkpoint = ReplayCheckpointBuilder(lambda task_id: MockWorld({"print('one')": "one"})).build(trajectory, 0)
    assert checkpoint.validation_status == "valid"
    visible = str(checkpoint.agent_messages)
    assert "print('two')" not in visible
    assert "two" not in visible


def test_future_message_leak_is_detected() -> None:
    """Future messages in a static prefix invalidate agent state."""
    trajectory = StaticTrajectory(
        task_id="task-1",
        source_path="mock",
        steps=[
            TrajectoryStep(
                0,
                "print('one')",
                "one",
                [],
                [{"role": "assistant", "content": "print('two')"}],
            ),
            TrajectoryStep(1, "print('two')", "two", [], []),
        ],
    )
    checkpoint = ReplayCheckpointBuilder(lambda task_id: MockWorld({"print('one')": "one"})).build(trajectory, 0)
    assert checkpoint.validation_status == "invalid_agent_state"
    assert any("Future execution leaked" in error for error in checkpoint.validation_errors)


def test_loader_reconstructs_steps_from_logger_jsonl(tmp_path: Path) -> None:
    """AppWorld logger.jsonl can be converted into static replay steps."""
    task_dir = tmp_path / "task-1"
    logs_dir = task_dir / "logs"
    version_dir = task_dir / "version"
    logs_dir.mkdir(parents=True)
    version_dir.mkdir()
    (version_dir / "code.txt").write_text("code-version", encoding="utf-8")
    (version_dir / "data.txt").write_text("data-version", encoding="utf-8")
    records: list[dict[str, Any]] = [
        {
            "role": "agent",
            "content": "```python\nprint('one')\n```",
            "raw_message": {"role": "assistant", "content": "```python\nprint('one')\n```"},
            "step_number": 1,
        },
        {"role": "environment", "content": "one\n", "raw_message": {}, "step_number": 2},
    ]
    (logs_dir / "logger.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    trajectory = load_appworld_logger_jsonl(task_dir)
    assert trajectory.task_id == "task-1"
    assert trajectory.steps[0].execution_code == "print('one')"
    assert trajectory.steps[0].execution_output == "one\n"
    assert trajectory.metadata["appworld_code_version"] == "code-version"
    assert trajectory.metadata["appworld_data_version"] == "data-version"


def test_no_llm_backend_is_called() -> None:
    """Replay uses only the injected world factory."""
    called = {"world": 0, "llm": 0}

    def world_factory(task_id: str) -> MockWorld:
        called["world"] += 1
        return MockWorld({"print('one')": "one"})

    checkpoint = ReplayCheckpointBuilder(world_factory).build(make_trajectory(), 0)
    assert checkpoint.validation_status == "valid"
    assert called == {"world": 1, "llm": 0}


def make_trajectory() -> StaticTrajectory:
    """Return a two-step static trajectory."""
    return StaticTrajectory(
        task_id="task-1",
        source_path="mock",
        steps=[
            TrajectoryStep(
                0,
                "print('one')",
                "one",
                [],
                [
                    {"role": "assistant", "content": "```python\nprint('one')\n```"},
                    {"role": "user", "content": "Output:\n```\none\n```"},
                ],
            ),
            TrajectoryStep(
                1,
                "print('two')",
                "two",
                [{"role": "user", "content": "Output:\n```\none\n```"}],
                [
                    {"role": "assistant", "content": "```python\nprint('one')\n```"},
                    {"role": "user", "content": "Output:\n```\none\n```"},
                    {"role": "assistant", "content": "```python\nprint('two')\n```"},
                    {"role": "user", "content": "Output:\n```\ntwo\n```"},
                ],
            ),
        ],
    )

