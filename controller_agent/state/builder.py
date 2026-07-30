"""Causal construction of complete Controller input."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from controller_agent.state.schemas import ControllerInput, TriggerResult


class StateBuilder:
    """Build a Controller view from the exact Agent-visible context."""

    def build(
        self,
        *,
        task: str,
        trigger: TriggerResult,
        agent_context_messages: Iterable[dict[str, Any]],
        proposed_agent_output: str,
        proposed_execution: str,
        previous_controller_decisions: Iterable[dict[str, Any]],
        context_complete: bool,
    ) -> ControllerInput:
        """Return a deep-copied, complete causal Controller input."""
        if not isinstance(task, str):
            raise TypeError("task must be a string.")
        if not isinstance(proposed_agent_output, str):
            raise TypeError("proposed_agent_output must be a string.")
        if not isinstance(proposed_execution, str):
            raise TypeError("proposed_execution must be a string.")
        if not isinstance(context_complete, bool):
            raise TypeError("context_complete must be a bool.")

        return ControllerInput(
            task=task.strip(),
            trigger_reason=str(trigger.reason),
            trigger_metadata=copy.deepcopy(trigger.metadata),
            agent_context_messages=_copy_messages(agent_context_messages),
            proposed_agent_output=proposed_agent_output,
            proposed_execution=proposed_execution,
            previous_controller_decisions=_copy_records(previous_controller_decisions),
            context_complete=context_complete,
        )


def _copy_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy model-visible messages without filtering or summarizing them."""
    copied: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("agent_context_messages must contain dictionaries.")
        copied.append(copy.deepcopy(message))
    return copied


def _copy_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy prior Controller records without deriving additional facts."""
    copied: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("previous_controller_decisions must contain dictionaries.")
        copied.append(copy.deepcopy(record))
    return copied
