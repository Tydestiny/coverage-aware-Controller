"""Tests for ControllerInput construction from visible history."""

from __future__ import annotations

from copy import deepcopy

from controller_agent.state.builder import StateBuilder
from controller_agent.state.schemas import TriggerResult


def test_builds_controller_input_from_visible_history() -> None:
    """Visible message content becomes the Controller history."""
    history = [
        {"role": "user", "content": "Create record_id=abc-123."},
        {"role": "assistant", "content": "Created record_id=abc-123."},
    ]

    result = StateBuilder().build(
        task="Create a record.",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=history,
        current_agent_output="I will verify it.",
    )

    assert result.task == "Create a record."
    assert result.trigger_reason == "before_finish"
    assert result.recent_history == history
    assert result.current_agent_output == "I will verify it."
    assert result.known_entities == ["abc-123"]


def test_extracts_recent_errors_and_completed_operations() -> None:
    """Visible content supplies deterministic error and operation summaries."""
    result = StateBuilder().build(
        task="Task",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=[
            {"role": "user", "content": "Execution failed: 422 invalid target."},
            {"role": "assistant", "content": "Created record_id=abc-123."},
        ],
        current_agent_output="Completed the requested operation.",
    )

    assert "Execution failed: 422 invalid target." in result.recent_errors
    assert "Created record_id=abc-123." in result.completed_operations
    assert "Completed the requested operation." in result.completed_operations


def test_does_not_mutate_provided_history() -> None:
    """Builder copies visible history before retaining it."""
    history = [{"role": "user", "content": "Task", "metadata": {"attempt": 1}}]
    original = deepcopy(history)

    result = StateBuilder().build(
        task="Task",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=history,
        current_agent_output="Output",
    )
    history[0]["metadata"]["attempt"] = 2

    assert history != original
    assert result.recent_history[0].get("metadata") is None


def test_does_not_include_hidden_message_fields_or_unprovided_future_history() -> None:
    """Only visible fields from supplied history enter ControllerInput."""
    visible_history = [
        {
            "role": "user",
            "content": "Visible progress.",
            "hidden_state": "future answer",
            "evaluator_result": "secret",
        }
    ]
    future_history = [{"role": "assistant", "content": "Future action."}]

    result = StateBuilder().build(
        task="Task",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=visible_history,
        current_agent_output="Current output.",
    )

    assert result.recent_history == [{"role": "user", "content": "Visible progress."}]
    assert "future answer" not in str(result)
    assert "secret" not in str(result)
    assert "Future action." not in str(result)
