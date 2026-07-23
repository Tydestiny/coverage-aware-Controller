"""Structured Controller event logger."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from controller_agent.state.schemas import ControllerDecision, ControllerInput, TriggerResult


class ExperimentLogger:
    """Collect JSON-serializable Controller events for tests or JSONL sinks."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.sink = sink

    def log_controller_event(
        self,
        trigger: TriggerResult,
        controller_input: ControllerInput,
        decision: ControllerDecision,
        blocked: bool,
    ) -> None:
        """Record a complete Controller decision event without secrets."""
        event = {
            "event_type": "controller_decision",
            "trigger_reason": trigger.reason,
            "should_call_controller": trigger.should_call_controller,
            "task": controller_input.task,
            "action": decision.action.value,
            "reason": decision.reason,
            "instruction": decision.instruction,
            "confidence": decision.confidence,
            "blocked": blocked,
            "known_entities": copy.deepcopy(controller_input.known_entities),
            "completed_operations": copy.deepcopy(controller_input.completed_operations),
            "recent_errors": copy.deepcopy(controller_input.recent_errors),
        }
        json.dumps(event)
        self.events.append(event)
        if self.sink is not None:
            self.sink(copy.deepcopy(event))
