"""End-to-end Controller loop tests using the existing fake AppWorld runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger
from controller_agent.controller.controller import Controller
from controller_agent.integration.appworld_adapter import AppWorldInterceptionAdapter, ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.state.schemas import Action, ControllerConfig, ControllerDecision, ControllerInput, ControllerResponse
from controller_agent.tests.test_end_to_end_smoke import ScriptedAgent, ScriptedWorld


@dataclass
class RecordingBackend:
    """Return a fixed decision while retaining each visible Controller input."""

    action: Action

    def __post_init__(self) -> None:
        self.inputs: list[ControllerInput] = []

    def __call__(self, controller_input: ControllerInput) -> ControllerResponse:
        self.inputs.append(controller_input)
        return ControllerResponse(
            ControllerDecision(self.action, "fake_backend", "Verify before finishing."),
            model_name="fake-controller",
            input_tokens=11,
            output_tokens=7,
        )


def _run(
    outputs: list[str], backend: object, config: ControllerConfig, max_agent_turns: int = 6
) -> tuple[ControlledAppWorldRunner, ScriptedWorld, ControlledReActAgent]:
    agent = ScriptedAgent()
    agent.outputs = outputs
    controlled = ControlledReActAgent(
        agent,
        controller=Controller(backend=backend),
        config=config,
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(
        controlled,
        budget=RunnerBudget(max_agent_turns=max_agent_turns, max_environment_executions=6, max_controller_calls=1, max_controller_blocks=1),
    )
    world = ScriptedWorld()
    runner.run_initialized_world(world)
    return runner, world, controlled


def test_read_action_skips_controller_and_executes_once() -> None:
    backend = RecordingBackend(Action.PROCEED)
    _, world, controlled = _run(["print('read state')"], backend, ControllerConfig(enabled=True), max_agent_turns=1)
    assert world.executed_codes == ["print('read state')"]
    assert backend.inputs == []
    assert controlled.runtime_state.total_calls == 0


def test_before_finish_proceed_executes_original_once_with_full_output() -> None:
    backend = RecordingBackend(Action.PROCEED)
    runner, world, controlled = _run(
        ["I verified the result.\n```python\napis.supervisor.complete_task()\n```"],
        backend,
        ControllerConfig(enabled=True, max_controller_calls_per_task=1, max_controller_blocks=1, max_consecutive_blocks=1),
    )
    assert world.executed_codes == ["I verified the result.\n```python\napis.supervisor.complete_task()\n```"]
    assert len(backend.inputs) == 1
    assert backend.inputs[0].proposed_agent_output.startswith("I verified")
    assert "complete_task" not in "\n".join(
        str(message.get("content", "")) for message in backend.inputs[0].agent_context_messages
    )
    assert runner.turn_events[0]["controller_action"] == "PROCEED"
    assert runner.turn_events[0]["guidance_injected"] is False
    assert controlled.runtime_state.total_calls == 1


def test_before_finish_replan_blocks_then_agent_regenerates() -> None:
    backend = RecordingBackend(Action.REPLAN)
    runner, world, _ = _run(
        ["apis.supervisor.complete_task()", "print('verify first')", "apis.supervisor.complete_task()"],
        backend,
        ControllerConfig(enabled=True, max_controller_calls_per_task=1, max_controller_blocks=1, max_consecutive_blocks=1),
    )
    assert world.executed_codes == ["print('verify first')", "apis.supervisor.complete_task()"]
    assert runner.turn_events[0]["world_batch_execute_called"] is False
    assert runner.turn_events[0]["guidance_injected"] is True
    assert runner.turn_events[1]["world_batch_execute_called"] is True


def test_controller_exception_fails_open_and_executes_original() -> None:
    def broken_backend(_: ControllerInput) -> object:
        raise TimeoutError("controller timeout")

    runner, world, controlled = _run(
        ["apis.supervisor.complete_task()"],
        broken_backend,
        ControllerConfig(enabled=True, max_controller_calls_per_task=1),
    )
    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert runner.turn_events[0]["execution_allowed"] is True
    assert controlled.logger.events[-1]["reason"] == "controller_failure_fallback"


def test_controller_call_and_block_budgets_fail_open() -> None:
    backend = RecordingBackend(Action.REPLAN)
    runner, world, controlled = _run(
        ["apis.supervisor.complete_task()", "apis.supervisor.complete_task()"],
        backend,
        ControllerConfig(enabled=True, max_controller_calls_per_task=1, max_controller_blocks=1, max_consecutive_blocks=1),
    )
    assert len(backend.inputs) == 1
    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert runner.turn_events[1]["execution_allowed"] is True
    assert controlled.runtime_state.total_blocks == 1


def test_runner_writes_direct_branch_records_with_raw_step_numbers(tmp_path: Path) -> None:
    """Blocked and regenerated actions are recorded independently after a checkpoint."""
    backend = RecordingBackend(Action.REPLAN)
    agent = ScriptedAgent()
    agent.outputs = ["apis.supervisor.complete_task()", "print('verify')", "apis.supervisor.complete_task()"]
    controlled = ControlledReActAgent(
        agent,
        controller=Controller(backend=backend),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=1, max_controller_blocks=1),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    logger = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=8)
    runner = ControlledAppWorldRunner(
        controlled,
        budget=RunnerBudget(max_agent_turns=4, max_environment_executions=4, max_controller_calls=1, max_controller_blocks=1),
        branch_runtime_logger=logger,
        controller_prompt_hash="g0-test",
    )
    world = ScriptedWorld()
    agent.initialize(world)
    agent.step_number = 8
    runner.run_restored_world(world)

    continuation = [json.loads(line) for line in logger.continuation_path.read_text(encoding="utf-8").splitlines()]
    calls = [json.loads(line) for line in logger.controller_calls_path.read_text(encoding="utf-8").splitlines()]
    assert [row["step_number"] for row in continuation] == [9, 10, 11]
    assert continuation[0]["execution_allowed"] is False
    assert continuation[0]["environment_observation"] is None
    assert continuation[1]["agent_call_id"] != continuation[0]["agent_call_id"]
    assert continuation[-1]["environment_observation"]
    assert continuation[-1]["task_completed"] is True
    assert continuation[-1]["termination_reason"] == "task_completed"
    assert calls[0]["step_number"] == continuation[0]["step_number"]
    assert calls[0]["controller_input"]["proposed_agent_output"] == "apis.supervisor.complete_task()"
