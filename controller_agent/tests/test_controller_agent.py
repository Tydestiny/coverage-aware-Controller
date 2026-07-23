from dataclasses import dataclass, field

from controller_agent.controller.controller import Controller
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import Action, ControllerDecision


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
        self.messages: list[dict[str, object]] = []
        self.world = type("World", (), {"task": type("Task", (), {"instruction": "Do task"})()})()

    def next_execution_inputs_usage_and_status(self, last_execution_outputs):
        return [FakeIO(self.output)], {"usage": 1}, FakeStatus()


def wrap(output: str, backend, enabled: bool = True):
    logger = ExperimentLogger()
    return ControlledReActAgent(FakeAgent(output), Controller(backend), logger=logger, enabled=enabled), logger


def test_no_trigger_delegates_without_controller_call() -> None:
    called = []
    wrapped, logger = wrap("print('continue')", lambda _: called.append(True))
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content == "print('continue')"
    assert called == []
    assert logger.events == []


def test_proceed_allows_completion() -> None:
    wrapped, logger = wrap("apis.supervisor.complete_task()", lambda _: {"action": "PROCEED", "reason": "verified", "confidence": 1})
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert logger.events[0]["blocked"] is False


def test_non_proceed_blocks_completion_and_adds_guidance() -> None:
    wrapped, logger = wrap("apis.supervisor.complete_task()", lambda _: {"action": "VERIFY_TARGET_SET", "reason": "check", "instruction": "Verify IDs.", "confidence": 0.8})
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content == ""
    assert result[0][0].metadata["controller_blocked"] is True
    assert "Verify IDs." in wrapped.agent.messages[-1]["content"]
    assert logger.events[0]["action"] == "VERIFY_TARGET_SET"


def test_invalid_json_fails_open_to_original_behavior() -> None:
    wrapped, logger = wrap("apis.supervisor.complete_task()", lambda _: "not json")
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert logger.events[0]["action"] == Action.PROCEED.value
    assert "fallback" in logger.events[0]["reason"]


def test_controller_call_failure_fails_open() -> None:
    def fail(_):
        raise RuntimeError("offline")

    wrapped, logger = wrap("apis.supervisor.complete_task()", fail)
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert logger.events[0]["blocked"] is False


def test_disabled_wrapper_is_behaviorally_transparent() -> None:
    wrapped, logger = wrap("apis.supervisor.complete_task()", lambda _: {"action": "BLOCK_RISKY_ACTION", "reason": "x"}, enabled=False)
    result = wrapped.next_execution_inputs_usage_and_status([])
    assert result[0][0].content.endswith("complete_task()")
    assert logger.events == []


def test_log_event_contains_required_fields() -> None:
    wrapped, logger = wrap("apis.supervisor.complete_task()", lambda _: {"action": "PROCEED", "reason": "verified", "confidence": 1})
    wrapped.next_execution_inputs_usage_and_status([])
    assert {
        "event_type",
        "trigger_reason",
        "should_call_controller",
        "task",
        "action",
        "reason",
        "instruction",
        "confidence",
        "blocked",
        "known_entities",
        "completed_operations",
        "recent_errors",
    } <= logger.events[0].keys()
