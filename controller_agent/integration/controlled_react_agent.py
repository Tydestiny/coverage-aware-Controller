"""Offline-testable wrapper around simplified_react_code_agent."""

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
        enabled: bool | None = None,
        config: ControllerConfig | None = None,
        runtime_state: ControllerRuntimeState | None = None,
        integration_adapter: ExecutionInterceptionAdapter | None = None,
    ) -> None:
        self.agent = agent
        self.controller = controller or Controller()
        self.trigger_detector = trigger_detector or BeforeFinishTrigger()
        self.state_builder = state_builder or StateBuilder()
        self.guidance_builder = guidance_builder or GuidanceBuilder()
        self.logger = logger or ExperimentLogger()
        self.config = config or ControllerConfig(enabled=True if enabled is None else enabled)
        self.runtime_state = runtime_state or ControllerRuntimeState()
        self.integration_adapter = integration_adapter or OfflineMockAdapter()

    def next_execution_inputs_usage_and_status(self, last_execution_outputs: Sequence[Any]) -> Any:
        """Delegate one Agent turn and inspect only proposed execution."""
        result = self.agent.next_execution_inputs_usage_and_status(last_execution_outputs)
        if not self.config.enabled:
            return result
        execution_inputs, _, status = result
        if getattr(status, "failed", False) or not execution_inputs:
            return result
        context = self._build_trigger_context(execution_inputs, last_execution_outputs)
        interception = self.inspect_proposed_execution(context, execution_inputs)
        if interception.trigger_result is None:
            return result
        return self.integration_adapter.apply(result, interception, self.agent)

    def inspect_proposed_execution(
        self,
        context: TriggerContext,
        original_execution_inputs: Any,
    ) -> InterceptionResult:
        """Inspect proposed execution and return a runtime-neutral decision."""
        trigger = self.trigger_detector.detect(context)
        if not trigger.should_call_controller:
            return InterceptionResult(True, original_execution_inputs, trigger_result=None)
        if self._controller_call_limit_reached() or self._consecutive_block_limit_reached():
            return InterceptionResult(True, original_execution_inputs, trigger_result=trigger)

        controller_input = self.state_builder.build(
            context.task,
            trigger,
            context.recent_history,
            context.proposed_execution,
        )
        response = self.controller.decide(controller_input)
        blocked = response.decision.action != Action.PROCEED
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
            original_execution_inputs=original_execution_inputs,
            guidance_message=guidance,
            trigger_result=trigger,
            controller_response=response,
            block_reason=response.decision.reason if blocked else None,
        )

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

    def _controller_call_limit_reached(self) -> bool:
        """Return whether the configured per-task call limit is exhausted."""
        limit = self.config.max_controller_calls_per_task
        return limit is not None and self.runtime_state.total_calls >= limit

    def _consecutive_block_limit_reached(self) -> bool:
        """Return whether the configured consecutive block limit is exhausted."""
        limit = self.config.max_consecutive_blocks
        return limit is not None and self.runtime_state.consecutive_blocks >= limit

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
