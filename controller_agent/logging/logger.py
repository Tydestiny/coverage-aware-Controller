"""Structured Controller event logger."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from controller_agent.state.schemas import (
    ControllerInput,
    ControllerResponse,
    ControllerRuntimeState,
    TriggerContext,
    TriggerResult,
)


class ExperimentLogger:
    """Collect JSON-serializable Controller events for tests or JSONL sinks."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.sink = sink

    def log_controller_event(
        self,
        context: TriggerContext,
        trigger: TriggerResult,
        controller_input: ControllerInput,
        response: ControllerResponse,
        runtime_state: ControllerRuntimeState,
        blocked: bool,
    ) -> None:
        """Record a complete Controller decision event without secrets."""
        decision = response.decision
        event = {
            "event_type": "controller_decision",
            "task_id": context.task_id,
            "step_index": context.step_index,
            "trigger_reason": trigger.reason,
            "should_call_controller": trigger.should_call_controller,
            "trigger_metadata": copy.deepcopy(trigger.metadata),
            "task": controller_input.task,
            "action": decision.action.value,
            "reason": decision.reason,
            "instruction": decision.instruction,
            "confidence": decision.confidence,
            "blocked": blocked,
            "model_name": response.model_name,
            "controller_input_tokens": response.input_tokens,
            "controller_output_tokens": response.output_tokens,
            "controller_latency_seconds": response.latency_seconds,
            "controller_cost_cny": response.cost_cny,
            "total_controller_calls": runtime_state.total_calls,
            "total_controller_blocks": runtime_state.total_blocks,
            "consecutive_blocks": runtime_state.consecutive_blocks,
            "known_entities": copy.deepcopy(controller_input.known_entities),
            "completed_operations": copy.deepcopy(controller_input.completed_operations),
            "recent_errors": copy.deepcopy(controller_input.recent_errors),
        }
        json.dumps(event)
        self.events.append(event)
        if self.sink is not None:
            self.sink(copy.deepcopy(event))
