"""Deterministic construction of Controller input from visible history."""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

from controller_agent.state.schemas import ControllerInput, TriggerResult


class StateBuilder:
    """Build a conservative Controller view without hidden state or future data."""

    def build(
        self,
        task: str,
        trigger: TriggerResult,
        recent_history: Iterable[dict[str, Any]],
        current_agent_output: str,
    ) -> ControllerInput:
        """Create a ControllerInput from task and visible messages."""
        history = [
            {
                key: copy.deepcopy(item[key])
                for key in ("role", "content")
                if key in item
            }
            for item in recent_history
        ]
        text = "\n".join(str(item.get("content", "")) for item in history)
        text += "\n" + (current_agent_output or "")
        entities = sorted(
            set(re.findall(r"\b(?:[A-Za-z]+[_ -])?(?:id|ID)[=: ]+['\"]?([A-Za-z0-9_-]+)", text))
        )
        errors = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"error|failed|exception|traceback|401|422", line, re.IGNORECASE)
        ][-10:]
        operations = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"created|updated|deleted|completed|added|removed", line, re.IGNORECASE)
        ][-20:]
        return ControllerInput(
            task=task,
            trigger_reason=trigger.reason,
            recent_history=history[-12:],
            current_agent_output=current_agent_output,
            known_entities=entities,
            completed_operations=operations,
            recent_errors=errors,
        )
