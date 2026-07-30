"""Runtime tests for complete causal Controller context capture."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from controller_agent.controller.controller import Controller
from controller_agent.integration.appworld_adapter import AppWorldInterceptionAdapter
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.state.schemas import Action, ControllerDecision, ControllerResponse


@dataclass
class _IO:
    content: str
    metadata: dict[str, object] | None = None


@dataclass
class _Status:
    failed: bool = False


class _LanguageModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[list[dict[str, object]]] = []

    def generate(self, *, messages, **_kwargs):
        self.calls.append(copy.deepcopy(messages))
        return next(self.outputs)


class _ModelCallingAgent:
    def __init__(self, outputs: list[str]) -> None:
        self.language_model = _LanguageModel(outputs)
        self.messages: list[dict[str, object]] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "early success observation: created item-1"},
            {"role": "user", "content": "current task"},
        ]
        self.world = type("World", (), {"task": type("Task", (), {"instruction": "current task", "task_id": "task-a"})()})()
        self.step_number = 0
        self.evaluator_result = {"ground_truth": "must never be read"}

    def next_execution_inputs_usage_and_status(self, last_execution_outputs):
        if last_execution_outputs:
            self.messages.append({"role": "user", "content": last_execution_outputs[0].content})
        output = self.language_model.generate(messages=self.messages)
        self.messages.append({"role": "assistant", "content": output})
        code = output.split("```python\n", 1)[-1].split("\n```", 1)[0]
        self.step_number += 1
        return [_IO(code)], {}, _Status()


def test_controller_receives_exact_agent_context_and_prior_guidance_result() -> None:
    """Captured model inputs retain early observations and prior intervention effects."""
    outputs = [
        "```python\napis.supervisor.complete_task(answer='first')\n```",
        "```python\nprint('verify')\n```",
        "```python\napis.supervisor.complete_task(answer='second')\n```",
    ]
    agent = _ModelCallingAgent(outputs)
    received = []

    def backend(controller_input):
        received.append(controller_input)
        action = Action.REPLAN if len(received) == 1 else Action.PROCEED
        return ControllerResponse(ControllerDecision(action, "check", "Verify the work."))

    wrapped = ControlledReActAgent(
        agent,
        Controller(backend),
        integration_adapter=AppWorldInterceptionAdapter(),
    )

    first = wrapped.next_execution_inputs_usage_and_status([])
    assert first.interception.allow_execution is False
    assert "[CONTROLLER_GUIDANCE]" in str(agent.messages[-1]["content"])

    second = wrapped.next_execution_inputs_usage_and_status([])
    assert second[0][0].content == "print('verify')"
    assert any("[CONTROLLER_GUIDANCE]" in str(message["content"]) for message in agent.language_model.calls[1])

    wrapped.next_execution_inputs_usage_and_status([_IO("verification succeeded")])
    assert len(received) == 2
    second_input = received[1]
    assert second_input.agent_context_messages == agent.language_model.calls[2]
    assert "early success observation: created item-1" in str(second_input.agent_context_messages)
    assert "Verify the work." in second_input.previous_controller_decisions[0]["guidance"]
    assert second_input.previous_controller_decisions[0]["execution_result"] == ["verification succeeded"]
    assert "ground_truth" not in repr(second_input).lower()
    assert second_input.context_complete is True


def test_context_complete_is_false_when_appworld_marks_history_trimmed() -> None:
    """AppWorld's own abbreviation markers are exposed rather than hidden."""
    agent = _ModelCallingAgent(["```python\napis.supervisor.complete_task()\n```"])
    agent.messages.insert(1, {"role": "user", "content": "[TRIMMED HISTORY]"})
    received = []
    wrapped = ControlledReActAgent(
        agent,
        Controller(lambda controller_input: received.append(controller_input) or {"action": "PROCEED", "reason": "ok"}),
        integration_adapter=AppWorldInterceptionAdapter(),
    )

    wrapped.next_execution_inputs_usage_and_status([])

    assert received[0].context_complete is False


def test_task_reset_does_not_carry_previous_context_or_decisions() -> None:
    """A new task starts with an isolated Controller context."""
    agent = _ModelCallingAgent(["```python\napis.supervisor.complete_task()\n```"])
    received = []
    wrapped = ControlledReActAgent(
        agent,
        Controller(lambda controller_input: received.append(controller_input) or {"action": "PROCEED", "reason": "ok"}),
        integration_adapter=AppWorldInterceptionAdapter(),
    )
    wrapped._previous_controller_decisions.append({"guidance": "old"})
    wrapped._current_task_id = "old-task"

    wrapped.next_execution_inputs_usage_and_status([])

    assert received[0].previous_controller_decisions == []
    assert received[0].agent_context_messages == agent.language_model.calls[0]


def test_controller_excludes_static_preamble_and_keeps_full_task_runtime_history() -> None:
    """The Agent retains its preamble while the Controller receives task runtime only."""
    agent = _ModelCallingAgent(
        [
            "```python\nprint('first current-task action')\n```",
            "```python\napis.supervisor.complete_task(answer='done')\n```",
        ]
    )
    agent.messages = [
        {"role": "system", "content": "static preamble unrelated to the runtime task"},
        {"role": "assistant", "content": "static demonstration action"},
        {"role": "user", "content": "current task instruction"},
    ]
    agent.num_instruction_messages = 3
    initial_messages = copy.deepcopy(agent.messages)
    received = []
    wrapped = ControlledReActAgent(
        agent,
        Controller(lambda controller_input: received.append(controller_input) or {"action": "PROCEED", "reason": "ok"}),
        integration_adapter=AppWorldInterceptionAdapter(),
    )

    wrapped.next_execution_inputs_usage_and_status([])
    wrapped.next_execution_inputs_usage_and_status([_IO("first environment observation")])

    assert agent.language_model.calls[0] == initial_messages
    controller_input = received[0]
    assert controller_input.agent_context_messages == [
        {"role": "user", "content": "current task instruction"},
        {"role": "assistant", "content": "```python\nprint('first current-task action')\n```"},
        {"role": "user", "content": "first environment observation"},
    ]
    assert "static preamble" not in str(controller_input.agent_context_messages)
    assert "static demonstration" not in str(controller_input.agent_context_messages)
    assert controller_input.context_complete is True


def test_static_marker_does_not_mark_runtime_context_incomplete() -> None:
    """Only runtime truncation affects context_complete after static filtering."""
    agent = _ModelCallingAgent(["```python\napis.supervisor.complete_task()\n```"])
    agent.messages = [
        {"role": "system", "content": "[TRIMMED HISTORY] static prompt marker"},
        {"role": "assistant", "content": "static demonstration"},
        {"role": "user", "content": "current task"},
    ]
    agent.num_instruction_messages = 3
    received = []
    wrapped = ControlledReActAgent(
        agent,
        Controller(lambda controller_input: received.append(controller_input) or {"action": "PROCEED", "reason": "ok"}),
        integration_adapter=AppWorldInterceptionAdapter(),
    )

    wrapped.next_execution_inputs_usage_and_status([])

    assert received[0].context_complete is True


def test_already_filtered_checkpoint_context_is_not_filtered_again() -> None:
    """Checkpoint audits containing runtime-only history retain their first task message."""
    agent = _ModelCallingAgent([])
    agent.messages = [
        {"role": "system", "content": "static preamble"},
        {"role": "assistant", "content": "static demonstration"},
        {"role": "user", "content": "current task"},
    ]
    agent.num_instruction_messages = 3
    wrapped = ControlledReActAgent(agent, integration_adapter=AppWorldInterceptionAdapter())
    checkpoint_runtime_messages = [
        {"role": "user", "content": "current task"},
        {"role": "assistant", "content": "runtime action"},
        {"role": "user", "content": "runtime observation"},
    ]

    assert wrapped._controller_runtime_messages(checkpoint_runtime_messages) == checkpoint_runtime_messages
