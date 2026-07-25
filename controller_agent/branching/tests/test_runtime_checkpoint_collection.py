"""Tests for runtime dense checkpoint collection."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.branching.checkpoint_capture_hook import RuntimeCheckpointCaptureHook
from controller_agent.branching.namespace_adapter import ReplayRequiredNamespaceAdapter
from controller_agent.integration.appworld_adapter import AppWorldInterceptionAdapter, ControlledAppWorldRunner
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.controller.smoke import DeterministicSmokeController
from controller_agent.state.schemas import ControllerConfig
from controller_agent.tests.test_end_to_end_smoke import ScriptedAgent, ScriptedWorld


class SavingScriptedWorld(ScriptedWorld):
    """Scripted world with named state save support."""

    def __init__(self) -> None:
        super().__init__()
        self.saved_states: list[str] = []

    def save_state(self, state_id: str | None = None) -> str | None:
        """Save a named state id."""
        if state_id is not None:
            self.saved_states.append(state_id)
            return state_id
        self.saved_states.append("unnamed")
        return None


def test_runtime_hook_saves_pre_and_post_records(tmp_path: Path) -> None:
    """Every Agent turn gets pre-action and every environment execution gets post-observation."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    task_dir = tmp_path / "run-1" / "scripted-task"
    manifest = read_json(task_dir / "manifest.json")
    assert len(manifest["pre_action"]) == 4
    assert len(manifest["post_observation"]) == 3
    assert len(world.saved_states) == 3
    assert all(state_id.startswith("scripted-task_post_") for state_id in world.saved_states)


def test_pre_action_references_recent_post_checkpoint(tmp_path: Path) -> None:
    """Pre-action records refer to the latest full checkpoint instead of duplicating state."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    task_dir = tmp_path / "run-1" / "scripted-task"
    pre_paths = sorted((task_dir / "pre_action").glob("*.json"))
    pre_records = [read_json(path) for path in pre_paths]
    assert pre_records[0]["latest_post_observation_checkpoint_id"] is None
    assert pre_records[1]["latest_post_observation_checkpoint_id"] == "scripted-task_post_0001"
    assert "appworld_state_id" not in pre_records[1]


def test_task_local_base_dir_stores_checkpoints_next_to_trajectory(tmp_path: Path) -> None:
    """Task-local layout writes manifest directly under the requested checkpoint directory."""
    task_output_dir = tmp_path / "outputs" / "controlled_run" / "tasks" / "scripted-task"
    hook = RuntimeCheckpointCaptureHook(
        root_dir=tmp_path / "unused-central-root",
        run_id="run-1",
        task_id="scripted-task",
        base_dir=task_output_dir / "checkpoints",
    )
    agent = ScriptedAgent()
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, checkpoint_capture_hook=hook)

    runner.run_initialized_world(SavingScriptedWorld())

    assert (task_output_dir / "checkpoints" / "manifest.json").exists()
    assert not (tmp_path / "unused-central-root" / "run-1" / "scripted-task" / "manifest.json").exists()


def test_blocked_complete_task_has_no_post_observation_checkpoint(tmp_path: Path) -> None:
    """Blocked complete_task is recorded as pre-action but does not save environment state."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    task_dir = tmp_path / "run-1" / "scripted-task"
    pre_records = [read_json(path) for path in sorted((task_dir / "pre_action").glob("*.json"))]
    post_records = [read_json(path) for path in sorted((task_dir / "post_observation").glob("*.json"))]
    blocked_pre = pre_records[1]
    assert blocked_pre["whether_contains_complete_task"] is True
    assert blocked_pre["proposed_execution"] == "apis.supervisor.complete_task()"
    assert len(post_records) == len(world.executed_codes)
    assert "apis.supervisor.complete_task()" not in [record["executed_code"] for record in post_records[:2]]


def test_allowed_complete_task_generates_terminal_record(tmp_path: Path) -> None:
    """Allowed final complete_task is represented in terminal metadata."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    terminal = read_json(tmp_path / "run-1" / "scripted-task" / "terminal.json")
    assert terminal["before_complete_task"] is True
    assert terminal["complete_task_blocked_by_controller"] is True
    assert terminal["complete_task_entered_world_batch_execute"] is True
    assert terminal["task_completed"] is True
    assert terminal["termination_reason"] == "task_completed"
    assert terminal["evaluator_executed"] is False


def test_hook_disabled_preserves_runner_behavior() -> None:
    """Hook None keeps the original execution behavior."""
    agent = ScriptedAgent()
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, checkpoint_capture_hook=None)
    world = SavingScriptedWorld()

    runner.run_initialized_world(world)

    assert world.executed_codes == [
        "print('query current state')",
        "print('verify target set')",
        "apis.supervisor.complete_task()",
    ]
    assert world.saved_states == []
    assert runner.checkpoint_hook_errors == []


def test_hook_failure_does_not_interrupt_task() -> None:
    """Hook exceptions are collected and do not stop environment execution."""

    class FailingHook:
        def on_pre_action(self, **kwargs):
            raise RuntimeError("pre failed")

        def on_post_observation(self, **kwargs):
            raise RuntimeError("post failed")

        def on_terminal(self, **kwargs):
            raise RuntimeError("terminal failed")

    agent = ScriptedAgent()
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, checkpoint_capture_hook=FailingHook())
    world = SavingScriptedWorld()

    runner.run_initialized_world(world)

    assert world.task_completed() is True
    assert any("on_pre_action" in error for error in runner.checkpoint_hook_errors)
    assert any("on_post_observation" in error for error in runner.checkpoint_hook_errors)
    assert any("on_terminal" in error for error in runner.checkpoint_hook_errors)


def test_checkpoint_records_do_not_include_evaluator_or_ground_truth(tmp_path: Path) -> None:
    """Stored runtime checkpoint records do not include forbidden evaluator fields."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    task_dir = tmp_path / "run-1" / "scripted-task"
    checkpoint_paths = list((task_dir / "pre_action").glob("*.json")) + list(
        (task_dir / "post_observation").glob("*.json")
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checkpoint_paths)
    assert "ground_truth" not in combined
    assert "evaluator" not in combined


def test_namespace_adapter_marks_replay_required() -> None:
    """Namespace restore is explicitly unsupported in the first version."""
    adapter = ReplayRequiredNamespaceAdapter()
    assert adapter.is_supported() is False
    assert adapter.snapshot({}) == {"namespace_restore_mode": "replay_required"}


def test_post_observation_marks_replay_required_namespace(tmp_path: Path) -> None:
    """Full checkpoints record replay_required rather than fake namespace support."""
    runner, world, _ = make_collecting_runner(tmp_path)

    runner.run_initialized_world(world)

    first_post_path = sorted((tmp_path / "run-1" / "scripted-task" / "post_observation").glob("*.json"))[0]
    first_post = read_json(first_post_path)
    assert first_post["namespace_restore_mode"] == "replay_required"
    assert first_post["executed_code_prefix"] == ["print('query current state')"]


def make_collecting_runner(
    tmp_path: Path,
) -> tuple[ControlledAppWorldRunner, SavingScriptedWorld, ControlledReActAgent]:
    """Create a scripted runner with runtime checkpoint capture enabled."""
    agent = ScriptedAgent()
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    hook = RuntimeCheckpointCaptureHook(
        root_dir=tmp_path,
        run_id="run-1",
        task_id="scripted-task",
        appworld_code_version="code-v",
        appworld_data_version="data-v",
    )
    return ControlledAppWorldRunner(controlled, checkpoint_capture_hook=hook), SavingScriptedWorld(), controlled


def read_json(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
