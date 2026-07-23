"""Offline-testable wrapper around simplified_react_code_agent."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from controller_agent.controller.controller import Controller
from controller_agent.controller.guidance import GuidanceBuilder
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.builder import StateBuilder
from controller_agent.state.schemas import Action
from controller_agent.triggers.before_finish import BeforeFinishTrigger


class ControlledReActAgent:
    """Wrap an existing ReAct Agent without changing it when disabled.

    The wrapped object must expose ``next_execution_inputs_usage_and_status``
    and may expose ``messages``, ``world.task.instruction`` and ``step_number``
    as the official simplified Agent does.
    """

    def __init__(
        self,
        agent: Any,
        controller: Controller | None = None,
        trigger_detector: Any | None = None,
        state_builder: StateBuilder | None = None,
        guidance_builder: GuidanceBuilder | None = None,
        logger: ExperimentLogger | None = None,
        enabled: bool = True,
    ) -> None:
        self.agent = agent
        self.controller = controller or Controller()
        self.trigger_detector = trigger_detector or BeforeFinishTrigger()
        self.state_builder = state_builder or StateBuilder()
        self.guidance_builder = guidance_builder or GuidanceBuilder()
        self.logger = logger or ExperimentLogger()
        self.enabled = enabled

    def next_execution_inputs_usage_and_status(self, last_execution_outputs: Sequence[Any]) -> tuple[Sequence[Any], Any, Any]:
        """Delegate one Agent turn and intercept only before_finish."""
        result = self.agent.next_execution_inputs_usage_and_status(last_execution_outputs)
        if not self.enabled:
            return result
        execution_inputs, usage, status = result
        if getattr(status, "failed", False) or not execution_inputs:
            return result
        current_output = str(getattr(execution_inputs[0], "content", ""))
        trigger = self.trigger_detector.detect(current_output)
        if not trigger.should_call_controller:
            return result
        controller_input = self.state_builder.build(
            self._task_text(), trigger, getattr(self.agent, "messages", []), current_output
        )
        decision = self.controller.decide(controller_input)
        blocked = decision.action != Action.PROCEED
        guidance = self.guidance_builder.build(decision)
        if blocked and guidance:
            messages = getattr(self.agent, "messages", None)
            if isinstance(messages, list):
                messages.append({"role": "user", "content": guidance})
        self.logger.log_controller_event(trigger, controller_input, decision, blocked)
        if not blocked:
            return result
        blocked_inputs = [copy.deepcopy(item) for item in execution_inputs]
        blocked_inputs[0].content = ""
        metadata = getattr(blocked_inputs[0], "metadata", None)
        if isinstance(metadata, dict):
            metadata["controller_blocked"] = True
            metadata["controller_action"] = decision.action.value
        return blocked_inputs, usage, status

    def _task_text(self) -> str:
        """Read the task instruction from the wrapped Agent when available."""
        world = getattr(self.agent, "world", None)
        task = getattr(world, "task", None)
        return str(getattr(task, "instruction", getattr(self.agent, "task", "")))

    def __getattr__(self, name: str) -> Any:
        """Delegate non-controller attributes to the original Agent."""
        return getattr(self.agent, name)
