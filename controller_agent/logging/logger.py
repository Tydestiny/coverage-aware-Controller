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
            "blocked": blocked,
            "model_name": response.model_name,
            "controller_input_tokens": response.input_tokens,
            "controller_output_tokens": response.output_tokens,
            "controller_latency_seconds": response.latency_seconds,
            "controller_cost_cny": response.cost_cny,
            "total_controller_calls": runtime_state.total_calls,
            "total_controller_blocks": runtime_state.total_blocks,
            "consecutive_blocks": runtime_state.consecutive_blocks,
            "trigger_metadata": copy.deepcopy(controller_input.trigger_metadata),
            "agent_context_messages": copy.deepcopy(controller_input.agent_context_messages),
            "proposed_agent_output": controller_input.proposed_agent_output,
            "proposed_execution": controller_input.proposed_execution,
            "previous_controller_decisions": copy.deepcopy(controller_input.previous_controller_decisions),
            "context_complete": controller_input.context_complete,
        }
        json.dumps(event)
        self.events.append(event)
        if self.sink is not None:
            self.sink(copy.deepcopy(event))

    def log_runtime_event(
        self,
        *,
        task_id: str,
        step_index: int,
        event_type: str,
        reason: str,
        trigger: TriggerResult | None,
        error: str | None,
        runtime_state: ControllerRuntimeState,
    ) -> None:
        """Record a fail-open runtime event without serializing model history."""
        event = {
            "event_type": event_type,
            "task_id": task_id,
            "step_index": step_index,
            "reason": reason,
            "trigger_reason": trigger.reason if trigger is not None else None,
            "trigger_metadata": copy.deepcopy(trigger.metadata) if trigger is not None else {},
            "error": error,
            "total_controller_calls": runtime_state.total_calls,
            "total_controller_blocks": runtime_state.total_blocks,
            "consecutive_blocks": runtime_state.consecutive_blocks,
        }
        self.events.append(event)
        if self.sink is not None:
            self.sink(copy.deepcopy(event))
