from controller_agent.controller.smoke import DeterministicSmokeController
from controller_agent.integration.appworld_adapter import AppWorldInterceptionAdapter, ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.state.schemas import ControllerConfig
from controller_agent.tests.test_end_to_end_smoke import ScriptedAgent, ScriptedWorld


def make_runner(outputs: list[str], budget: RunnerBudget) -> tuple[ControlledAppWorldRunner, ScriptedWorld, ControlledReActAgent]:
    """Create a runner with scripted outputs."""
    agent = ScriptedAgent()
    agent.outputs = outputs
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(
            enabled=True,
            max_controller_calls_per_task=budget.max_controller_calls,
            max_controller_blocks=budget.max_controller_blocks,
        ),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, budget=budget)
    world = ScriptedWorld()
    return runner, world, controlled


def test_blocked_complete_task_does_not_consume_environment_execution() -> None:
    runner, world, controlled = make_runner(
        ["apis.supervisor.complete_task()", "print('continue')"],
        RunnerBudget(max_agent_turns=3, max_environment_executions=1),
    )
    runner.run_initialized_world(world)
    assert controlled.runtime_state.agent_turn_count == 2
    assert controlled.runtime_state.environment_execution_count == 1
    assert controlled.runtime_state.total_calls == 1
    assert controlled.runtime_state.total_blocks == 1
    assert world.executed_codes == ["print('continue')"]


def test_environment_execution_budget_terminates_before_next_execute() -> None:
    runner, world, _ = make_runner(
        ["print('one')", "print('two')"],
        RunnerBudget(max_agent_turns=5, max_environment_executions=1),
    )
    runner.run_initialized_world(world)
    assert world.executed_codes == ["print('one')"]
    assert runner.stop_reason == "max_environment_executions_reached"
    assert runner.turn_events[-1]["termination_reason"] == "max_environment_executions_reached"
    assert len(runner.turn_events) == 1
    assert runner.turn_events[-1]["world_batch_execute_called"] is True


def test_agent_turn_budget_terminates_without_environment_limit() -> None:
    runner, world, controlled = make_runner(
        ["print('one')", "print('two')"],
        RunnerBudget(max_agent_turns=1, max_environment_executions=10),
    )
    runner.run_initialized_world(world)
    assert world.executed_codes == ["print('one')"]
    assert controlled.runtime_state.agent_turn_count == 1
    assert runner.stop_reason == "max_agent_turns_reached"


def test_blocked_extra_turn_budget_terminates_after_block() -> None:
    runner, world, _ = make_runner(
        ["apis.supervisor.complete_task()", "print('after block')", "print('too late')"],
        RunnerBudget(max_agent_turns=1, max_environment_executions=10, max_extra_agent_turns_after_block=1),
    )
    runner.run_initialized_world(world)
    assert world.executed_codes == ["print('after block')"]
    assert runner.stop_reason == "blocked_extra_turns_exhausted"


def test_controller_disabled_preserves_original_budget_flow() -> None:
    agent = ScriptedAgent()
    agent.outputs = ["apis.supervisor.complete_task()"]
    controlled = ControlledReActAgent(
        agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(enabled=False),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    runner = ControlledAppWorldRunner(controlled, budget=RunnerBudget())
    world = ScriptedWorld()
    runner.run_initialized_world(world)
    assert world.executed_codes == ["apis.supervisor.complete_task()"]
    assert controlled.runtime_state.total_calls == 0
    assert controlled.runtime_state.total_blocks == 0


def test_identical_code_with_identical_observation_stops_as_a_loop() -> None:
    """A genuine no-progress loop ends without reinstating a global turn cap."""
    runner, world, _ = make_runner(
        ["print('same')", "print('same')", "print('same')", "print('later')"],
        RunnerBudget(
            max_agent_turns=None,
            max_environment_executions=None,
            max_consecutive_identical_executions=3,
        ),
    )

    runner.run_initialized_world(world)

    assert world.executed_codes == ["print('same')", "print('same')", "print('same')"]
    assert runner.stop_reason == "repeated_action_loop"
    assert runner.turn_events[-1]["termination_reason"] == "repeated_action_loop"
