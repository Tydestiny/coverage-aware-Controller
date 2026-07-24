from __future__ import annotations

from copy import deepcopy
from typing import Any

from controller_agent.state.schemas import (
    ControllerRuntimeState,
    TriggerContext,
)


class TriggerContextBuilder:
    """Build the shared trigger context for online and offline use."""

    def build(
        self,
        *,
        task_id: str,
        step_index: int,
        task: str,
        proposed_execution: str,
        last_execution_outputs: list[Any],
        recent_history: list[dict[str, Any]],
        runtime_state: ControllerRuntimeState | dict[str, Any],
    ) -> TriggerContext:
        """Build one isolated canonical TriggerContext."""
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")
        state = (
            runtime_state.to_dict()
            if isinstance(runtime_state, ControllerRuntimeState)
            else deepcopy(runtime_state)
        )

        return TriggerContext(
            task_id=task_id,
            step_index=step_index,
            task=task,
            proposed_execution=proposed_execution,
            last_execution_outputs=deepcopy(last_execution_outputs),
            recent_history=deepcopy(recent_history),
            runtime_state=deepcopy(state),
        )
