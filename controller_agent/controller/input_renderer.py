"""Render model-visible Controller input without embedding a system policy."""

from __future__ import annotations

import json

from controller_agent.state.schemas import ControllerInput


def render_controller_input(controller_input: ControllerInput) -> str:
    """Serialize the complete ControllerInput in a stable field order."""
    fields = (
        ("task", controller_input.task),
        ("trigger_reason", controller_input.trigger_reason),
        ("trigger_metadata", controller_input.trigger_metadata),
        ("context_complete", controller_input.context_complete),
        ("context_metadata", {
            key: controller_input.context_metadata[key]
            for key in ("scope", "step_number", "task_boundary_verified", "incomplete_reasons", "proposed_execution_status")
            if key in controller_input.context_metadata
        }),
        ("agent_context_messages", controller_input.agent_context_messages),
        ("previous_controller_decisions", controller_input.previous_controller_decisions),
        ("proposed_agent_output", controller_input.proposed_agent_output),
        ("proposed_execution", controller_input.proposed_execution),
    )
    return "\n\n".join(
        f"## {name}\n{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
        for name, value in fields
    )
