"""Tests for canonical TriggerContext construction."""

from __future__ import annotations

import pytest

from controller_agent.state.schemas import ControllerRuntimeState, TriggerContext
from controller_agent.state.trigger_context_builder import TriggerContextBuilder


def test_builds_expected_canonical_trigger_context() -> None:
    """Builder populates the public TriggerContext contract."""
    context = TriggerContextBuilder().build(
        task_id="task-1",
        step_index=3,
        task="Complete the task.",
        proposed_execution="print('work')",
        last_execution_outputs=["previous output"],
        recent_history=[{"role": "user", "content": "Task"}],
        runtime_state={"total_calls": 1},
    )

    assert isinstance(context, TriggerContext)
    assert context.task_id == "task-1"
    assert context.step_index == 3
    assert context.proposed_execution == "print('work')"
    assert context.last_execution_outputs == ["previous output"]


def test_converts_controller_runtime_state_to_dictionary() -> None:
    """Controller runtime state is stored as its public dictionary snapshot."""
    runtime_state = ControllerRuntimeState(total_calls=2, total_blocks=1, last_action="REPLAN")

    context = TriggerContextBuilder().build(
        task_id="task-1",
        step_index=1,
        task="Task",
        proposed_execution="print('work')",
        last_execution_outputs=[],
        recent_history=[],
        runtime_state=runtime_state,
    )

    assert context.runtime_state == runtime_state.to_dict()


def test_accepts_runtime_state_dictionary() -> None:
    """Existing runtime-state dictionaries are retained by value."""
    state = {"total_calls": 3, "nested": {"key": "value"}}

    context = TriggerContextBuilder().build(
        task_id="task-1",
        step_index=1,
        task="Task",
        proposed_execution="print('work')",
        last_execution_outputs=[],
        recent_history=[],
        runtime_state=state,
    )

    assert context.runtime_state == state


def test_rejects_negative_step_index() -> None:
    """Negative step indexes are invalid trigger positions."""
    with pytest.raises(ValueError, match="step_index"):
        TriggerContextBuilder().build(
            task_id="task-1",
            step_index=-1,
            task="Task",
            proposed_execution="print('work')",
            last_execution_outputs=[],
            recent_history=[],
            runtime_state={},
        )


def test_context_is_isolated_from_later_input_mutation() -> None:
    """Nested mutable inputs cannot alter a completed context."""
    outputs = [{"items": ["first"]}]
    history = [{"role": "user", "content": "Task", "metadata": {"count": 1}}]
    state = {"counts": {"calls": 1}}

    context = TriggerContextBuilder().build(
        task_id="task-1",
        step_index=1,
        task="Task",
        proposed_execution="print('work')",
        last_execution_outputs=outputs,
        recent_history=history,
        runtime_state=state,
    )
    outputs[0]["items"].append("later")
    history[0]["metadata"]["count"] = 2
    state["counts"]["calls"] = 2

    assert context.last_execution_outputs == [{"items": ["first"]}]
    assert context.recent_history[0]["metadata"] == {"count": 1}
    assert context.runtime_state == {"counts": {"calls": 1}}
