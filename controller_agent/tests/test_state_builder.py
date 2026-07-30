"""Tests for complete causal ControllerInput construction."""

from __future__ import annotations

from controller_agent.state.builder import StateBuilder
from controller_agent.state.schemas import TriggerResult


def _build(messages, decisions=None, *, complete=True):
    return StateBuilder().build(
        task="  Current task only.  ",
        trigger=TriggerResult(True, "before_finish", {"source": "test"}),
        agent_context_messages=messages,
        proposed_agent_output="I will finish.\n```python\napis.supervisor.complete_task()\n```",
        proposed_execution="apis.supervisor.complete_task()",
        previous_controller_decisions=decisions or [],
        context_complete=complete,
    )


def test_builder_preserves_early_success_observation_without_window_truncation() -> None:
    """The Controller gets every message provided to the Agent model."""
    messages = [{"role": "user", "content": "early observation: create succeeded"}]
    messages.extend({"role": "assistant", "content": f"later step {index}"} for index in range(20))

    result = _build(messages)

    assert len(result.agent_context_messages) == 21
    assert result.agent_context_messages[0]["content"] == "early observation: create succeeded"
    assert result.task == "Current task only."
    assert result.trigger_metadata == {"source": "test"}
    assert result.context_complete is True


def test_builder_deep_copies_context_and_previous_decisions() -> None:
    """Later caller mutations cannot alter the decision input snapshot."""
    messages = [{"role": "user", "content": {"value": "original"}}]
    decisions = [{"guidance": "verify first", "execution_result": ["ok"]}]

    result = _build(messages, decisions)
    messages[0]["content"]["value"] = "changed"
    decisions[0]["execution_result"].append("changed")

    assert result.agent_context_messages[0]["content"]["value"] == "original"
    assert result.previous_controller_decisions[0]["execution_result"] == ["ok"]


def test_builder_only_uses_explicit_causal_arguments() -> None:
    """Evaluator data and future state cannot enter absent from supplied input."""
    result = _build([{"role": "user", "content": "visible history"}])
    serialized = repr(result)

    assert "evaluator" not in serialized.lower()
    assert "ground_truth" not in serialized.lower()
    assert "future" not in serialized.lower()
    assert result.proposed_agent_output.startswith("I will finish.")
    assert result.proposed_execution == "apis.supervisor.complete_task()"


def test_builder_preserves_incomplete_context_signal() -> None:
    """Known upstream truncation is explicit rather than silently hidden."""
    result = _build([{"role": "user", "content": "[TRIMMED HISTORY]"}], complete=False)

    assert result.context_complete is False
