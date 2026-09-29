"""Offline-testable wrapper around simplified_react_code_agent.

Core interface status: frozen for the Coverage-Aware Controller v0 core.
New triggers, state builders, controller backends, and runtime adapters should
plug into this wrapper through the existing constructor interfaces.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

from controller_agent.controller.controller import Controller
from controller_agent.controller.guidance import GuidanceBuilder
from controller_agent.integration.base import ExecutionInterceptionAdapter, OfflineMockAdapter
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.builder import StateBuilder
from controller_agent.state.context_capture import capture_context
from controller_agent.state.schemas import (
    Action,
    ControllerConfig,
    ControllerInput,
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
        self.last_controller_input: ControllerInput | None = None
        self._agent_context_messages: list[dict[str, Any]] = []
        self._agent_context_complete = False
        self._previous_controller_decisions: list[dict[str, Any]] = []
        self._pending_execution: dict[str, Any] | None = None
        self._context_metadata: dict[str, Any] = {}
        self._generation_message_count: int | None = None
        self._trigger_history: list[dict[str, Any]] = []
        self._pending_trigger_action: str | None = None

    def next_execution_inputs_usage_and_status(self, last_execution_outputs: Sequence[Any]) -> Any:
        """Delegate one Agent turn and inspect only proposed execution."""
        self._reset_if_task_changed(self._task_id())
        self._record_previous_execution_result(last_execution_outputs)
        self._record_trigger_observation(last_execution_outputs)
        self.last_interception_result = None
        self.last_controller_input = None
        self._generation_message_count = len(getattr(self.agent, "messages", []))
        with self._capture_agent_context_messages():
            result = self.agent.next_execution_inputs_usage_and_status(last_execution_outputs)
        if not self.config.enabled:
            return result
        execution_inputs, _, status = result
        if getattr(status, "failed", False) or not execution_inputs:
            return result
        context = self._build_trigger_context(execution_inputs, last_execution_outputs)
        interception = self.inspect_proposed_execution(context, result, execution_inputs)
        self.last_interception_result = interception
        if interception.allow_execution:
            self._pending_trigger_action = context.proposed_execution
            self._pending_execution = {
                "step_number": context.step_index,
                "executed_code": [str(getattr(item, "content", "")) for item in execution_inputs],
                "decision_index": len(self._previous_controller_decisions) - 1,
            }
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
        budget_bypassed = trigger.should_call_controller and (
            self._controller_call_limit_reached()
            or self._consecutive_block_limit_reached()
            or self._controller_block_limit_reached()
        )
        self.logger.log_trigger_evaluation(context, trigger, controller_budget_bypassed=budget_bypassed)
        if not trigger.should_call_controller:
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=None)
        if budget_bypassed:
            self._log_fail_open(context, "controller_budget_exhausted", None, trigger)
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=trigger)

        try:
            controller_input = self.state_builder.build(
                task=context.task,
                trigger=trigger,
                agent_context_messages=self._agent_context_messages,
                proposed_agent_output=self._latest_assistant_output(context.proposed_execution),
                proposed_execution=context.proposed_execution,
                previous_controller_decisions=self._previous_controller_decisions,
                context_complete=self._agent_context_complete,
                context_metadata={**self._context_metadata, "task_id": context.task_id,
                                  "step_number": context.step_index},
            )
            self.last_controller_input = controller_input
            response = self.controller.decide(controller_input)
        except Exception as exc:
            self._log_fail_open(context, "controller_or_state_builder_failure", exc, trigger)
            return InterceptionResult(True, original_result, original_execution_inputs, trigger_result=trigger)
        blocked = response.decision.action != Action.PROCEED
        if blocked and self._controller_block_limit_reached():
            blocked = False
        guidance = self.guidance_builder.build(response.decision) if blocked else None
        self._update_runtime_state(context, response, blocked)
        self._append_controller_decision(context, trigger, response, guidance, blocked)
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
        self._agent_context_messages = []
        self._agent_context_complete = False
        self._previous_controller_decisions.clear()
        self._pending_execution = None
        self._context_metadata = {}
        self._generation_message_count = None
        self._trigger_history.clear()
        self._pending_trigger_action = None
        reset = getattr(self.trigger_detector, "reset", None)
        if callable(reset):
            reset()
        self.logger.events.clear()
        self.logger.trigger_events.clear()

    def _build_trigger_context(self, execution_inputs: Sequence[Any], last_execution_outputs: Sequence[Any]) -> TriggerContext:
        """Build trigger context from visible wrapped-agent state."""
        return TriggerContext(
            task_id=self._task_id(),
            step_index=int(getattr(self.agent, "step_number", 0) or 0),
            task=self._task_text(),
            proposed_execution=str(getattr(execution_inputs[0], "content", "")),
            last_execution_outputs=list(last_execution_outputs),
            recent_history=copy.deepcopy(self._trigger_history),
            runtime_state=self.runtime_state.to_dict(),
        )

    def _record_trigger_observation(self, outputs: Sequence[Any]) -> None:
        """Pair an executed candidate with its following environment output."""
        if self._pending_trigger_action is None:
            return
        self._trigger_history.append({
            "proposed_execution": self._pending_trigger_action,
            "last_execution_outputs": [getattr(output, "content", output) for output in outputs],
        })
        self._pending_trigger_action = None
        self._trigger_history = self._trigger_history[-20:]

    @contextmanager
    def _capture_agent_context_messages(self) -> Any:
        """Capture only the current task runtime view of the Agent model call."""
        self._agent_context_messages = []
        self._agent_context_complete = False
        self._context_metadata = {"source": "unavailable", "incomplete_reasons": ["model_input_not_captured"]}
        capture_count = 0
        language_model = getattr(self.agent, "language_model", None)
        original_generate = getattr(language_model, "generate", None)
        if not callable(original_generate):
            yield
            return

        def capture_generate(*args: Any, **kwargs: Any) -> Any:
            nonlocal capture_count
            capture_count += 1
            messages = kwargs.get("messages")
            if messages is None and args:
                messages = args[0]
            if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
                self._agent_context_messages, self._agent_context_complete, self._context_metadata = capture_context(
                    messages, getattr(self.agent, "messages", None),
                    getattr(self.agent, "num_instruction_messages", None),
                )
                self._context_metadata["capture_count"] = capture_count
                if capture_count > 1:
                    self._agent_context_complete = False
                    self._context_metadata["incomplete_reasons"].append("multiple_generations_in_one_turn")
            else:
                self._agent_context_messages = []
                self._agent_context_complete = False
                self._context_metadata = {"source": "unavailable", "incomplete_reasons": ["invalid_messages"]}
            return original_generate(*args, **kwargs)

        setattr(language_model, "generate", capture_generate)
        try:
            yield
        finally:
            setattr(language_model, "generate", original_generate)

    def _controller_runtime_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Exclude the Agent's static preamble while retaining the current task history."""
        boundary = getattr(self.agent, "num_instruction_messages", None)
        if not isinstance(boundary, int) or boundary < 1 or boundary > len(messages):
            return list(messages)
        agent_messages = getattr(self.agent, "messages", None)
        if not isinstance(agent_messages, list) or agent_messages[: boundary - 1] != messages[: boundary - 1]:
            return list(messages)
        return [messages[boundary - 1], *messages[boundary:]]

    def _record_previous_execution_result(self, outputs: Sequence[Any]) -> None:
        """Support wrappers that deliver observations on the next generation."""
        if outputs:
            self.record_environment_execution(outputs)
        else:
            self._pending_execution = None

    def record_environment_execution(self, outputs: Sequence[Any]) -> None:
        """Attach actual execution feedback once, without inferring guidance success."""
        pending = self._pending_execution
        self._pending_execution = None
        if pending is None or pending["decision_index"] < 0:
            return
        decision = self._previous_controller_decisions[pending["decision_index"]]
        observed = copy.deepcopy([getattr(output, "content", output) for output in outputs])
        execution = {
            "step_number": pending["step_number"],
            "executed_code": pending["executed_code"],
            "observation": observed,
        }
        if not decision["blocked"] and decision["step_number"] == pending["step_number"]:
            decision["original_execution_status"] = "executed"
            decision["execution_result"] = observed
        else:
            decision["subsequent_executions"].append(execution)

    def _append_controller_decision(
        self,
        context: TriggerContext,
        trigger: TriggerResult,
        response: ControllerResponse,
        guidance: str | None,
        blocked: bool,
    ) -> None:
        """Persist a causal decision record for subsequent Controller calls."""
        self._previous_controller_decisions.append(
            {
                "step_index": context.step_index,
                "step_number": context.step_index,
                "proposed_execution": context.proposed_execution,
                "trigger_reason": trigger.reason,
                "trigger_metadata": copy.deepcopy(trigger.metadata),
                "action": response.decision.action.value,
                "reason": response.decision.reason,
                "instruction": response.decision.instruction,
                "guidance": guidance,
                "blocked": blocked,
                "execution_result": None,
                "original_execution_status": "blocked" if blocked else "allowed_pending_execution",
                "subsequent_executions": [],
            }
        )

    def _latest_assistant_output(self, fallback: str) -> str:
        """Return the complete latest assistant message, falling back to code."""
        messages = list(getattr(self.agent, "messages", []))
        if self._generation_message_count is not None:
            messages = messages[self._generation_message_count:]
        for message in reversed(messages):
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
