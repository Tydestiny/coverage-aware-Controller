from dataclasses import dataclass, field

from controller_agent.controller.smoke import DeterministicSmokeController
from controller_agent.integration.appworld_adapter import AppWorldInterceptionAdapter, ControlledAppWorldRunner, GUIDANCE_MARKER
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.state.schemas import ControllerConfig


@dataclass
class ScriptedExecutionIO:
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ScriptedUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    cost: float = 0.001


@dataclass
class ScriptedStatus:
    failed: bool = False
    message: str = ""


class ScriptedTask:
    task_id = "scripted-task"
    instruction = "Use tools, verify state, then complete the task."


class ScriptedWorld:
    def __init__(self) -> None:
        self.task_id = ScriptedTask.task_id
        self.task = ScriptedTask()
        self.executed_codes: list[str] = []
        self.completed_after_execution_index: int | None = None

    def batch_execute(self, codes: list[str]) -> list[str]:
        outputs = []
        for code in codes:
            self.executed_codes.append(code)
            if "complete_task" in code:
                self.completed_after_execution_index = len(self.executed_codes)
            outputs.append(f"output for: {code}")
        return outputs

    def task_completed(self) -> bool:
        return self.completed_after_execution_index is not None


class ScriptedAgent:
    def __init__(self) -> None:
        self.outputs = [
            "print('query current state')",
            "apis.supervisor.complete_task()",
            "print('verify target set')",
            "apis.supervisor.complete_task()",
        ]
        self.messages: list[dict[str, object]] = []
        self.max_steps = 8
        self.step_number = 0
        self.world = None

    def initialize(self, world: ScriptedWorld) -> None:
        self.world = world
        self.step_number = 0
        self.messages = [{"role": "user", "content": world.task.instruction}]

    def next_execution_inputs_usage_and_status(self, last_execution_outputs):
        if last_execution_outputs:
            self.messages.append({"role": "user", "content": last_execution_outputs[0].content})
        output = self.outputs.pop(0)
        self.messages.append({"role": "assistant", "content": output})
        return [ScriptedExecutionIO(output)], ScriptedUsage(), ScriptedStatus()


def test_deterministic_controller_full_runner_chain() -> None:
    world = ScriptedWorld()
    controlled = ControlledReActAgent(
        ScriptedAgent(),
        controller=DeterministicSmokeController(),
        config=ControllerConfig(
            enabled=True,
            max_controller_calls_per_task=2,
        ),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled)

    runner.run_initialized_world(world)

    assert world.executed_codes == [
        "print('query current state')",
        "print('verify target set')",
        "apis.supervisor.complete_task()",
    ]
    assert "" not in world.executed_codes
    assert world.completed_after_execution_index == 3

    events = runner.turn_events
    assert [event["generated_code"] for event in events] == [
        "print('query current state')",
        "apis.supervisor.complete_task()",
        "print('verify target set')",
        "apis.supervisor.complete_task()",
    ]
    assert events[0]["world_batch_execute_called"] is True
    assert events[1]["world_batch_execute_called"] is False
    assert events[1]["execution_allowed"] is False
    assert events[1]["controller_action"] == "VERIFY_TARGET_SET"
    assert events[1]["task_completed"] is False
    assert events[2]["world_batch_execute_called"] is True
    assert events[3]["execution_allowed"] is True
    assert events[3]["controller_action"] == "PROCEED"
    assert events[3]["task_completed"] is True

    guidance_messages = [
        message for message in controlled.agent.messages if str(message.get("content", "")).startswith(GUIDANCE_MARKER)
    ]
    assert len(guidance_messages) == 1
    assert "complete_task" in events[1]["generated_code"]

    assert controlled.runtime_state.total_calls == 2
    assert controlled.runtime_state.total_blocks == 1
    assert controlled.runtime_state.consecutive_blocks == 0
    assert controlled.runtime_state.agent_turn_count == 4
    assert controlled.runtime_state.environment_execution_count == 3


def test_runner_does_not_execute_empty_code_after_guidance() -> None:
    class EmptyAfterGuidanceAgent(ScriptedAgent):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = [
                "apis.supervisor.complete_task()",
                "I will verify the target set before finishing.",
                "apis.supervisor.complete_task()",
            ]

        def next_execution_inputs_usage_and_status(self, last_execution_outputs):
            output = self.outputs.pop(0)
            self.messages.append({"role": "assistant", "content": output})
            code = "" if output.startswith("I will verify") else output
            return [ScriptedExecutionIO(code)], ScriptedUsage(), ScriptedStatus()

    world = ScriptedWorld()
    controlled = ControlledReActAgent(
        EmptyAfterGuidanceAgent(),
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled)

    runner.run_initialized_world(world)

    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert runner.stop_reason == "task_completed"
    assert runner.turn_events[1]["world_batch_execute_called"] is False
    assert runner.turn_events[1]["generated_code"] == ""
    assert runner.turn_events[1]["guidance_injected"] is True
    assert controlled.runtime_state.environment_execution_count == 1


def test_runner_stops_after_one_empty_code_retry() -> None:
    class RepeatedEmptyAfterGuidanceAgent(ScriptedAgent):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = [
                "apis.supervisor.complete_task()",
                "I will verify.",
                "Still thinking in natural language.",
            ]

        def next_execution_inputs_usage_and_status(self, last_execution_outputs):
            output = self.outputs.pop(0)
            self.messages.append({"role": "assistant", "content": output})
            code = "" if not output.startswith("apis.") else output
            return [ScriptedExecutionIO(code)], ScriptedUsage(), ScriptedStatus()

    world = ScriptedWorld()
    controlled = ControlledReActAgent(
        RepeatedEmptyAfterGuidanceAgent(),
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled)

    runner.run_initialized_world(world)

    assert world.executed_codes == []
    assert runner.stop_reason == "model_format_failure"
    assert [event["guidance_injected"] for event in runner.turn_events] == [True, True, False]


def test_runner_allows_extra_turns_after_block_at_step_limit() -> None:
    class LateFinishAgent(ScriptedAgent):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = [
                "print('step one')",
                "apis.supervisor.complete_task()",
                "apis.supervisor.complete_task()",
            ]
            self.max_steps = 2

    world = ScriptedWorld()
    controlled = ControlledReActAgent(
        LateFinishAgent(),
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=2),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, max_extra_turns_after_first_block=3)

    runner.run_initialized_world(world)

    assert world.executed_codes == ["print('step one')", "apis.supervisor.complete_task()"]
    assert runner.stop_reason == "task_completed"
    assert controlled.runtime_state.agent_turn_count == 3
