from dataclasses import dataclass, field

from controller_agent.controller.smoke import DeterministicSmokeController
from controller_agent.integration.appworld_adapter import (
    AppWorldInterceptionAdapter,
    AppWorldInterceptionApplied,
    ControlledAppWorldRunner,
    GUIDANCE_MARKER,
)
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.state.schemas import ControllerConfig


@dataclass
class FakeIO:
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeUsage:
    value: int = 1


@dataclass
class FakeStatus:
    failed: bool = False
    message: str = ""


class FakeTask:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.instruction = "Finish only when all requirements are satisfied."


class FakeWorld:
    def __init__(self, task_id: str = "task-1") -> None:
        self.task_id = task_id
        self.task = FakeTask(task_id)
        self.executed_codes: list[str] = []
        self.completed = False

    def batch_execute(self, codes: list[str]) -> list[str]:
        self.executed_codes.extend(codes)
        if any("complete_task" in code for code in codes):
            self.completed = True
        return [f"executed: {code}" for code in codes]

    def task_completed(self) -> bool:
        return self.completed


class FakeBaseReActAgent:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.messages: list[dict[str, object]] = []
        self.step_number = 0
        self.max_steps = 10
        self.world = None

    def initialize(self, world: FakeWorld) -> None:
        self.world = world
        self.step_number = 0
        self.messages = [{"role": "user", "content": world.task.instruction}]

    def next_execution_inputs_usage_and_status(self, last_execution_outputs):
        if last_execution_outputs:
            self.messages.append({"role": "user", "content": last_execution_outputs[0].content})
        output = self.outputs.pop(0)
        self.messages.append({"role": "assistant", "content": output})
        return [FakeIO(output)], FakeUsage(), FakeStatus()


def make_controlled_agent(outputs: list[str], enabled: bool = True) -> ControlledReActAgent:
    return ControlledReActAgent(
        FakeBaseReActAgent(outputs),
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=enabled),
        integration_adapter=AppWorldInterceptionAdapter(),
    )


def test_first_complete_task_is_blocked_and_second_executes() -> None:
    world = FakeWorld()
    agent = make_controlled_agent(
        [
            "apis.supervisor.complete_task()",
            "print('continue work')",
            "apis.supervisor.complete_task()",
        ]
    )
    ControlledAppWorldRunner(agent).run_initialized_world(world)
    assert world.executed_codes == ["print('continue work')", "apis.supervisor.complete_task()"]
    assert world.completed is True


def test_guidance_enters_next_round_context_without_user_task_replacement() -> None:
    world = FakeWorld()
    agent = make_controlled_agent(
        [
            "apis.supervisor.complete_task()",
            "print('continue work')",
            "apis.supervisor.complete_task()",
        ]
    )
    ControlledAppWorldRunner(agent).run_initialized_world(world)
    guidance_messages = [
        message for message in agent.agent.messages if str(message.get("content", "")).startswith(GUIDANCE_MARKER)
    ]
    assert len(guidance_messages) == 1
    assert agent.agent.messages[0]["content"] == world.task.instruction
    assert agent.agent.messages[0]["role"] == "user"
    assert guidance_messages[0]["role"] == "user"


def test_blocked_complete_task_is_not_executed_and_no_empty_code_runs() -> None:
    world = FakeWorld()
    agent = make_controlled_agent(
        [
            "apis.supervisor.complete_task()",
            "apis.supervisor.complete_task()",
        ]
    )
    ControlledAppWorldRunner(agent).run_initialized_world(world)
    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert "" not in world.executed_codes


def test_appworld_adapter_returns_blocked_result_for_runner() -> None:
    agent = make_controlled_agent(["apis.supervisor.complete_task()"])
    world = FakeWorld()
    agent.initialize(world)
    agent.step_number = 1
    result = agent.next_execution_inputs_usage_and_status([])
    assert isinstance(result, AppWorldInterceptionApplied)
    assert result.guidance_injected is True
    assert result.interception.allow_execution is False


def test_appworld_adapter_does_not_duplicate_same_guidance() -> None:
    agent = make_controlled_agent(["apis.supervisor.complete_task()", "apis.supervisor.complete_task()"])
    world = FakeWorld()
    agent.initialize(world)
    agent.step_number = 1
    first = agent.next_execution_inputs_usage_and_status([])
    second = AppWorldInterceptionAdapter().apply(first.original_result, first.interception, agent.agent)
    assert isinstance(first, AppWorldInterceptionApplied)
    assert isinstance(second, AppWorldInterceptionApplied)
    assert first.guidance_injected is True
    assert second.guidance_injected is False


def test_runtime_state_resets_for_new_task() -> None:
    agent = make_controlled_agent(["apis.supervisor.complete_task()"])
    world = FakeWorld("task-1")
    agent.initialize(world)
    agent.step_number = 1
    agent.next_execution_inputs_usage_and_status([])
    assert agent.runtime_state.total_calls == 1
    assert agent.runtime_state.total_blocks == 1
    agent.reset_for_task("task-2")
    assert agent.runtime_state.total_calls == 0
    assert agent.runtime_state.total_blocks == 0
    assert agent.runtime_state.consecutive_blocks == 0
    assert agent.runtime_state.last_trigger_step is None
    assert agent.runtime_state.last_action is None


def test_controller_disabled_keeps_original_flow() -> None:
    world = FakeWorld()
    agent = make_controlled_agent(["apis.supervisor.complete_task()"], enabled=False)
    ControlledAppWorldRunner(agent).run_initialized_world(world)
    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert world.completed is True
