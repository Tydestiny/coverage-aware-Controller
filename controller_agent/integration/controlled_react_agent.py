"""Offline-testable wrapper around simplified_react_code_agent.

Core interface status: frozen for the Coverage-Aware Controller v0 core.
New triggers, state builders, controller backends, and runtime adapters should
plug into this wrapper through the existing constructor interfaces.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from controller_agent.controller.controller import Controller
from controller_agent.controller.guidance import GuidanceBuilder
from controller_agent.integration.base import ExecutionInterceptionAdapter, OfflineMockAdapter
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.builder import StateBuilder
from controller_agent.state.schemas import (
    Action,
    ControllerConfig,
    ControllerResponse,
    ControllerRuntimeState,
    InterceptionResult,
    TriggerContext,
    TriggerResult,
)
from controller_agent.triggers.composite import CompositeTriggerDetectorWithBudget
from controller_agent.triggers.config import TriggerBudgetConfig, TriggerConfig


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
        enabled: bool | None = None,
        config: ControllerConfig | None = None,
        runtime_state: ControllerRuntimeState | None = None,
        integration_adapter: ExecutionInterceptionAdapter | None = None,
    ) -> None:
        self.agent = agent
        self.controller = controller or Controller()
        # The runtime always dispatches through the composite interface.  The
        # initial experiment enables only before_finish by configuration.
        self.trigger_detector = trigger_detector or _before_finish_composite()
        self.state_builder = state_builder or StateBuilder()
        self.guidance_builder = guidance_builder or GuidanceBuilder()
        self.logger = logger or ExperimentLogger()
        self.config = config or ControllerConfig(enabled=True if enabled is None else enabled)
        self.runtime_state = runtime_state or ControllerRuntimeState()
        self.integration_adapter = integration_adapter or OfflineMockAdapter()
        self._current_task_id: str | None = None
        self.last_interception_result: InterceptionResult | None = None

    def next_execution_inputs_usage_and_status(self, last_execution_outputs: Sequence[Any]) -> Any:
        """Delegate one Agent turn and inspect only proposed execution."""
        self.last_interception_result = None
        result = self.agent.next_execution_inputs_usage_and_status(last_execution_outputs)
        if not self.config.enabled:
            return result
        execution_inputs, _, status = result
        if getattr(status, "failed", False) or not execution_inputs:
            return result
        context = self._build_trigger_context(execution_inputs, last_execution_outputs)
        self._reset_if_task_changed(context.task_id)
        context = self._build_trigger_context(execution_inputs, last_execution_outputs)
        interception = self.inspect_proposed_execution(context, result, execution_inputs)
        self.last_interception_result = interception
        if interception.trigger_result is None:
            return result
        return self.integration_adapter.apply(result, interception, self.agent)

    def inspect_proposed_execution(
        self,
        context: TriggerContext,
        original_result: Any,
        original_execution_inputs: Any,
    ) -> InterceptionResult:
        """Inspect proposed execution and return a runtime-neutral decision."""
        try:
            trigger = self.trigger_detector.detect(context)
        except Exception as exc:
            self._log_fail_open(context, "trigger_failure", exc)
            return InterceptionResult(True, original_result, original_execution_inputs)
        if not trigger.should_call_controller:
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=None)
        if self._controller_call_limit_reached() or self._consecutive_block_limit_reached():
            self._log_fail_open(context, "controller_budget_exhausted", None, trigger)
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=trigger)

        try:
            controller_input = self.state_builder.build(
                context.task,
                trigger,
                self._history_before_current_output(),
                self._latest_assistant_output(context.proposed_execution),
            )
            response = self.controller.decide(controller_input)
        except Exception as exc:
            self._log_fail_open(context, "controller_or_state_builder_failure", exc, trigger)
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=trigger)
        blocked = response.decision.action != Action.PROCEED
        if blocked and self._controller_block_limit_reached():
            blocked = False
        guidance = self.guidance_builder.build(response.decision) if blocked else None
        self._update_runtime_state(context, response, blocked)
        self.logger.log_controller_event(
            context=context,
            trigger=trigger,
            controller_input=controller_input,
            response=response,
            runtime_state=self.runtime_state,
            blocked=blocked,
        )
        return InterceptionResult(
            allow_execution=not blocked,
            original_result=original_result,
            original_execution_inputs=original_execution_inputs,
            guidance_message=guidance,
            trigger_result=trigger,
            controller_response=response,
            block_reason=response.decision.reason if blocked else None,
        )

    def reset_for_task(self, task_id: str) -> None:
        """Reset controller state for a new AppWorld task."""
        self.runtime_state.total_calls = 0
        self.runtime_state.total_blocks = 0
        self.runtime_state.consecutive_blocks = 0
        self.runtime_state.agent_turn_count = 0
        self.runtime_state.environment_execution_count = 0
        self.runtime_state.last_trigger_step = None
        self.runtime_state.last_action = None
        self._current_task_id = task_id
        reset = getattr(self.trigger_detector, "reset", None)
        if callable(reset):
            reset()
        self.logger.events.clear()

    def _build_trigger_context(self, execution_inputs: Sequence[Any], last_execution_outputs: Sequence[Any]) -> TriggerContext:
        """Build trigger context from visible wrapped-agent state."""
        return TriggerContext(
            task_id=self._task_id(),
            step_index=int(getattr(self.agent, "step_number", 0) or 0),
            task=self._task_text(),
            proposed_execution=str(getattr(execution_inputs[0], "content", "")),
            last_execution_outputs=list(last_execution_outputs),
            recent_history=copy.deepcopy(list(getattr(self.agent, "messages", []))[-12:]),
            runtime_state=self.runtime_state.to_dict(),
        )

    def _history_before_current_output(self) -> list[dict[str, Any]]:
        """Return visible history before the just-generated assistant output."""
        history = copy.deepcopy(list(getattr(self.agent, "messages", [])))
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
                del history[index]
                break
        return history[-12:]

    def _latest_assistant_output(self, fallback: str) -> str:
        """Return the complete latest assistant message, falling back to code."""
        for message in reversed(list(getattr(self.agent, "messages", []))):
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
        return fallback

    def _log_fail_open(
        self,
        context: TriggerContext,
        reason: str,
        error: Exception | None,
        trigger: TriggerResult | None = None,
    ) -> None:
        """Record an outer-loop failure without preventing original execution."""
        self.logger.log_runtime_event(
            task_id=context.task_id,
            step_index=context.step_index,
            event_type="controller_fail_open",
            reason=reason,
            trigger=trigger,
            error=None if error is None else f"{type(error).__name__}: {error}",
            runtime_state=self.runtime_state,
        )

    def _controller_call_limit_reached(self) -> bool:
        """Return whether the configured per-task call limit is exhausted."""
        limit = self.config.max_controller_calls_per_task
        return limit is not None and self.runtime_state.total_calls >= limit

    def _consecutive_block_limit_reached(self) -> bool:
        """Return whether the configured consecutive block limit is exhausted."""
        limit = self.config.max_consecutive_blocks
        return limit is not None and self.runtime_state.consecutive_blocks >= limit

    def _controller_block_limit_reached(self) -> bool:
        """Return whether the configured total block limit is exhausted."""
        limit = self.config.max_controller_blocks
        return limit is not None and self.runtime_state.total_blocks >= limit

    def _update_runtime_state(self, context: TriggerContext, response: ControllerResponse, blocked: bool) -> None:
        """Maintain simple Controller runtime counters."""
        self.runtime_state.total_calls += 1
        self.runtime_state.last_trigger_step = context.step_index
        self.runtime_state.last_action = response.decision.action.value
        if blocked:
            self.runtime_state.total_blocks += 1
            self.runtime_state.consecutive_blocks += 1
        else:
            self.runtime_state.consecutive_blocks = 0

    def _reset_if_task_changed(self, task_id: str) -> None:
        """Reset runtime state when the wrapped task id changes."""
        if task_id and task_id != self._current_task_id:
            self.reset_for_task(task_id)

    def _task_text(self) -> str:
        """Read the task instruction from the wrapped Agent when available."""
        task = self._task_obj()
        return str(getattr(task, "instruction", getattr(self.agent, "task", "")))

    def _task_id(self) -> str:
        """Read the task id from the wrapped Agent when available."""
        task = self._task_obj()
        return str(getattr(task, "task_id", getattr(task, "id", getattr(self.agent, "task_id", ""))))

    def _task_obj(self) -> Any:
        """Return the wrapped Agent task object if present."""
        world = getattr(self.agent, "world", None)
        return getattr(world, "task", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-controller attributes to the original Agent."""
        return getattr(self.agent, name)


def _before_finish_composite() -> CompositeTriggerDetectorWithBudget:
    """Create the first-version composite with only before-finish enabled."""
    config = TriggerConfig(
        budget=TriggerBudgetConfig(
            detector_order=("before_finish",),
            max_triggers_per_task=4,
            min_before_finish_triggers=1,
        )
    )
    return CompositeTriggerDetectorWithBudget(full_config=config)
