from dataclasses import dataclass, field

from controller_agent.controller.controller import Controller
from controller_agent.integration.base import OfflineInterceptionResult, OfflineMockAdapter
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import Action, ControllerConfig, ControllerDecision, ControllerResponse, TriggerContext


@dataclass
class FakeIO:
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeStatus:
    failed: bool = False


class FakeAgent:
    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: list[dict[str, object]] = [{"role": "assistant", "content": "created record id=abc"}]
        self.world = type("World", (), {"task": type("Task", (), {"instruction": "Do task", "task_id": "task-1"})()})()
        self.step_number = 3

    def next_execution_inputs_usage_and_status(self, last_execution_outputs):
        return [FakeIO(self.output)], {"usage": 1}, FakeStatus()


def wrap(output: str, backend, enabled: bool = True, config: ControllerConfig | None = None):
    logger = ExperimentLogger()
    adapter = OfflineMockAdapter()
    wrapped = ControlledReActAgent(
        FakeAgent(output),
        Controller(backend),
        logger=logger,
        config=config or ControllerConfig(enabled=enabled),
        integration_adapter=adapter,
    )
    return wrapped, logger, adapter


def test_disabled_wrapper_is_behaviorally_transparent() -> None:
    wrapped, logger, adapter = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {"action": "BLOCK_RISKY_ACTION", "reason": "x"},
        enabled=False,
    )
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert logger.events == []
    assert adapter.applied == []


def test_no_trigger_delegates_without_controller_call() -> None:
    called = []
    wrapped, logger, adapter = wrap("print('continue')", lambda _: called.append(True))
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content == "print('continue')"
    assert called == []
    assert logger.events == []
    assert adapter.applied == []


def test_proceed_allows_original_execution() -> None:
    wrapped, logger, adapter = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {"action": "PROCEED", "reason": "verified", "confidence": 1},
    )
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert adapter.applied[0].allow_execution is True
    assert logger.events[0]["blocked"] is False


def test_block_returns_explicit_interception_result() -> None:
    wrapped, logger, adapter = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {"action": "VERIFY_TARGET_SET", "reason": "check", "instruction": "Verify IDs.", "confidence": 0.8},
    )
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert isinstance(result, OfflineInterceptionResult)
    assert result.integration_status == "integration not configured"
    assert result.interception.allow_execution is False
    assert result.interception.block_reason == "check"
    assert "Verify IDs." in result.interception.guidance_message
    assert logger.events[0]["action"] == "VERIFY_TARGET_SET"
    assert adapter.applied[0].allow_execution is False


def test_block_does_not_clear_original_execution_code() -> None:
    wrapped, _, _ = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {"action": "REPLAN", "reason": "not done", "instruction": "Continue.", "confidence": 0.7},
    )
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert isinstance(result, OfflineInterceptionResult)
    assert result.original_result[0][0].content == "apis.supervisor.complete_task()"


def test_trigger_context_fields_are_complete() -> None:
    captured: list[TriggerContext] = []

    class CapturingTrigger:
        def detect(self, context: TriggerContext):
            captured.append(context)
            from controller_agent.state.schemas import TriggerResult

            return TriggerResult(False, "skip", {})

    wrapped, _, _ = wrap("print('continue')", lambda _: None)
    wrapped.trigger_detector = CapturingTrigger()
    wrapped.next_execution_inputs_usage_and_status(["last output"])
    context = captured[0]
    assert context.task_id == "task-1"
    assert context.step_index == 3
    assert context.task == "Do task"
    assert context.proposed_execution == "print('continue')"
    assert context.last_execution_outputs == ["last output"]
    assert context.recent_history == [{"role": "assistant", "content": "created record id=abc"}]
    assert "total_calls" in context.runtime_state


def test_legacy_controller_backend_mapping_is_compatible() -> None:
    response = Controller(lambda _: {"action": "PROCEED", "reason": "legacy", "confidence": 1}).decide(
        _minimal_controller_input()
    )
    assert isinstance(response, ControllerResponse)
    assert response.decision.action == Action.PROCEED
    assert response.model_name is None


def test_controller_response_usage_is_logged() -> None:
    wrapped, logger, _ = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {
            "decision": {"action": "PROCEED", "reason": "ok", "confidence": 1},
            "model_name": "mock-controller",
            "input_tokens": 12,
            "output_tokens": 5,
            "latency_seconds": 0.2,
            "cost_cny": 0.01,
            "raw_response": "{}",
        },
    )
    wrapped.next_execution_inputs_usage_and_status([])
    event = logger.events[0]
    assert event["model_name"] == "mock-controller"
    assert event["controller_input_tokens"] == 12
    assert event["controller_output_tokens"] == 5
    assert event["controller_latency_seconds"] == 0.2
    assert event["controller_cost_cny"] == 0.01


def test_invalid_json_and_exceptions_fail_open() -> None:
    wrapped, logger, adapter = wrap("apis.supervisor.complete_task()", lambda _: "not json")
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert adapter.applied[0].allow_execution is True
    assert logger.events[0]["action"] == Action.PROCEED.value
    assert "fallback" in logger.events[0]["reason"]

    def fail(_):
        raise RuntimeError("offline")

    wrapped, logger, adapter = wrap("apis.supervisor.complete_task()", fail)
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert adapter.applied[0].allow_execution is True
    assert logger.events[0]["blocked"] is False


def test_runtime_state_tracks_calls_blocks_and_limits() -> None:
    wrapped, logger, adapter = wrap(
        "apis.supervisor.complete_task()",
        lambda _: {"action": "BLOCK_RISKY_ACTION", "reason": "risk", "instruction": "Stop.", "confidence": 0.9},
        config=ControllerConfig(enabled=True, max_controller_calls_per_task=1, max_consecutive_blocks=1),
    )
    first = wrapped.next_execution_inputs_usage_and_status([])
    second = wrapped.next_execution_inputs_usage_and_status([])
    assert isinstance(first, OfflineInterceptionResult)
    assert not isinstance(second, OfflineInterceptionResult)
    assert wrapped.runtime_state.total_calls == 1
    assert wrapped.runtime_state.total_blocks == 1
    assert wrapped.runtime_state.consecutive_blocks == 1
    assert wrapped.runtime_state.last_action == Action.BLOCK_RISKY_ACTION.value
    assert adapter.applied[0].allow_execution is False
    assert logger.events[0]["total_controller_calls"] == 1
    assert logger.events[0]["total_controller_blocks"] == 1
    assert logger.events[0]["consecutive_blocks"] == 1


def test_unconfigured_adapter_does_not_execute_empty_code() -> None:
    wrapped, _, _ = wrap(
        "apis.supervisor.complete_task()",
        lambda _: ControllerDecision(Action.BLOCK_RISKY_ACTION, "unsafe", "Do more checks.", 0.9),
    )
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert isinstance(result, OfflineInterceptionResult)
    assert result.integration_status == "integration not configured"
    assert result.original_result[0][0].content == "apis.supervisor.complete_task()"


def _minimal_controller_input():
    from controller_agent.state.schemas import ControllerInput

    return ControllerInput(
        task="Do task",
        trigger_reason="before_finish",
        recent_history=[],
        current_agent_output="apis.supervisor.complete_task()",
        known_entities=[],
        completed_operations=[],
        recent_errors=[],
    )
