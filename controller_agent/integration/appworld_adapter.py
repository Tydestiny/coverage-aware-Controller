"""Minimal AppWorld integration adapter for Controller interception."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

from controller_agent.integration.base import ExecutionInterceptionAdapter
from controller_agent.state.schemas import InterceptionResult


GUIDANCE_MARKER = "[CONTROLLER_GUIDANCE]"
FORMAT_REMINDER = (
    "Your previous response did not include an executable Python code block. "
    "In the next turn, output exactly one executable Python code block. "
    "If the task is verified complete, call apis.supervisor.complete_task(...) in that code block."
)


@dataclass(frozen=True)
class AppWorldInterceptionApplied:
    """Result consumed by the minimal AppWorld runner wrapper."""

    original_result: Any
    interception: InterceptionResult
    guidance_injected: bool


@dataclass(frozen=True)
class RunnerBudget:
    """Independent runtime budgets for controlled AppWorld execution."""

    max_agent_turns: int = 20
    max_environment_executions: int = 15
    max_controller_calls: int = 2
    max_controller_blocks: int = 1
    max_extra_agent_turns_after_block: int = 3
    max_cost_cny: float = 1.0


class AppWorldInterceptionAdapter:
    """Apply Controller interception to a simplified AppWorld ReAct agent."""

    def apply(self, original_result: Any, interception: InterceptionResult, agent: Any) -> Any:
        """Return original execution or a blocked result for a runner wrapper."""
        if interception.allow_execution:
            return original_result
        injected = self.inject_guidance(agent, interception.guidance_message)
        return AppWorldInterceptionApplied(
            original_result=original_result,
            interception=interception,
            guidance_injected=injected,
        )

    def inject_guidance(self, agent: Any, guidance_message: str | None) -> bool:
        """Inject marked guidance into the agent's visible message history."""
        if not guidance_message:
            return False
        messages = getattr(agent, "messages", None)
        if not isinstance(messages, list):
            raise TypeError("AppWorld agent must expose a list-like messages attribute.")
        content = f"{GUIDANCE_MARKER}\n{guidance_message.strip()}\n"
        injected_set = getattr(agent, "_controller_guidance_injected", None)
        if not isinstance(injected_set, set):
            injected_set = set()
            setattr(agent, "_controller_guidance_injected", injected_set)
        if content in injected_set:
            return False
        if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == content:
            injected_set.add(content)
            return False
        messages.append({"role": "user", "content": content})
        injected_set.add(content)
        return True


class ControlledAppWorldRunner:
    """Small solve loop wrapper that can skip blocked finish execution."""

    def __init__(
        self,
        agent: Any,
        adapter: ExecutionInterceptionAdapter | None = None,
        budget: RunnerBudget | None = None,
        max_extra_turns_after_first_block: int | None = None,
    ) -> None:
        self.agent = agent
        self.adapter = adapter or AppWorldInterceptionAdapter()
        if budget is None:
            budget = RunnerBudget(
                max_extra_agent_turns_after_block=(
                    3 if max_extra_turns_after_first_block is None else max_extra_turns_after_first_block
                )
            )
        self.budget = budget
        self.turn_events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.first_block_turn: int | None = None
        self.empty_code_retry_used = False
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        if hasattr(agent, "integration_adapter"):
            agent.integration_adapter = self.adapter

    def run_initialized_world(self, world: Any) -> None:
        """Run one already constructed world without modifying AppWorld source."""
        task_id = str(getattr(world, "task_id", ""))
        if hasattr(self.agent, "initialize"):
            self.agent.initialize(world)
        if hasattr(self.agent, "reset_for_task"):
            self.agent.reset_for_task(task_id)
        execution_outputs: Sequence[Any] = []
        while self._should_continue():
            turn_start = time.perf_counter()
            self._increment_step_number()
            self._runtime_state().agent_turn_count += 1
            result = self.agent.next_execution_inputs_usage_and_status(execution_outputs)
            if isinstance(result, AppWorldInterceptionApplied):
                self._record_usage_from_result(task_id, result.original_result)
                if self.first_block_turn is None:
                    self.first_block_turn = self._runtime_state().agent_turn_count
                self.turn_events.append(self._blocked_event(result, world, turn_start))
                if self._extra_turn_limit_reached():
                    self.stop_reason = "blocked_extra_turns_exhausted"
                    self.turn_events[-1]["termination_reason"] = self.stop_reason
                    break
                if self._cost_limit_reached(task_id):
                    self.stop_reason = "cost_limit_reached"
                    self.turn_events[-1]["termination_reason"] = self.stop_reason
                    break
                execution_outputs = []
                continue
            execution_inputs, usage, status = result
            self._record_usage_from_result(task_id, result)
            if getattr(status, "failed", False):
                self.stop_reason = "runner_or_api_error"
                self._log_termination(getattr(status, "message", ""))
                break
            if self._all_execution_inputs_empty(execution_inputs):
                reminder_injected = False
                if self.first_block_turn is not None and not self.empty_code_retry_used:
                    reminder_injected = self._inject_format_reminder()
                    self.empty_code_retry_used = True
                    self.turn_events.append(
                        self._empty_code_event(result, execution_inputs, world, turn_start, reminder_injected)
                    )
                    if self._extra_turn_limit_reached():
                        self.stop_reason = "blocked_extra_turns_exhausted"
                        self.turn_events[-1]["termination_reason"] = self.stop_reason
                        break
                    if self._cost_limit_reached(task_id):
                        self.stop_reason = "cost_limit_reached"
                        self.turn_events[-1]["termination_reason"] = self.stop_reason
                        break
                    execution_outputs = []
                    continue
                self.turn_events.append(
                    self._empty_code_event(result, execution_inputs, world, turn_start, reminder_injected)
                )
                self.stop_reason = "model_format_failure"
                self.turn_events[-1]["termination_reason"] = self.stop_reason
                break
            if self._environment_execution_limit_reached():
                self.stop_reason = "max_environment_executions_reached"
                self.turn_events.append(
                    self._budget_stop_event(result, execution_inputs, world, turn_start, self.stop_reason)
                )
                break
            outputs = world.batch_execute([item.content for item in execution_inputs])
            self._runtime_state().environment_execution_count += len(execution_inputs)
            execution_outputs = [
                type(item)(content=output, metadata=item.metadata)
                for item, output in zip(execution_inputs, outputs, strict=True)
            ]
            task_completed = bool(world.task_completed())
            self.turn_events.append(
                self._executed_event(result, execution_inputs, world, task_completed, turn_start)
            )
            if task_completed:
                self.stop_reason = "task_completed"
                self.turn_events[-1]["termination_reason"] = self.stop_reason
                break
            if self._environment_execution_limit_reached():
                self.stop_reason = "max_environment_executions_reached"
                self.turn_events[-1]["termination_reason"] = self.stop_reason
                break
            if self._cost_limit_reached(task_id):
                self.stop_reason = "cost_limit_reached"
                self.turn_events[-1]["termination_reason"] = self.stop_reason
                break
            if self._extra_turn_limit_reached():
                self.stop_reason = "blocked_extra_turns_exhausted"
                self.turn_events[-1]["termination_reason"] = self.stop_reason
                break
        if self.stop_reason is None:
            self.stop_reason = "max_agent_turns_reached"
            if self.turn_events:
                self.turn_events[-1]["termination_reason"] = self.stop_reason

    def _log_termination(self, message: str) -> None:
        """Log termination if the wrapped agent exposes an AppWorld logger."""
        logger = getattr(self.agent, "logger", None)
        if logger is not None and hasattr(logger, "show_message"):
            logger.show_message(role="termination", content=message)

    def _record_usage(self, task_id: str, usage: Any) -> None:
        """Record usage if the wrapped agent exposes an AppWorld usage tracker."""
        tracker = getattr(self.agent, "usage_tracker", None)
        if tracker is not None and hasattr(tracker, "add"):
            tracker.add(task_id, usage)
        if hasattr(self.agent, "log_usage"):
            self.agent.log_usage()

    def _record_usage_from_result(self, task_id: str, result: Any) -> None:
        """Record model usage even when execution is blocked or skipped."""
        try:
            usage = result[1]
        except (TypeError, IndexError):
            return
        self.cumulative_input_tokens += int(self._usage_tokens(usage, "input_tokens") or 0)
        self.cumulative_output_tokens += int(self._usage_tokens(usage, "output_tokens") or 0)
        self._record_usage(task_id, usage)

    def _usage_exceeded(self, task_id: str) -> bool:
        """Return whether the wrapped usage tracker has exceeded limits."""
        tracker = getattr(self.agent, "usage_tracker", None)
        if tracker is None or not hasattr(tracker, "exceeded"):
            return False
        return bool(tracker.exceeded(task_id))

    def _cost_limit_reached(self, task_id: str) -> bool:
        """Return whether configured cost limit is reached."""
        if self._usage_exceeded(task_id):
            return True
        cost = self._total_cost()
        return cost is not None and float(cost) >= self.budget.max_cost_cny

    def _environment_execution_limit_reached(self) -> bool:
        """Return whether environment execution budget is exhausted."""
        return self._runtime_state().environment_execution_count >= self.budget.max_environment_executions

    def _increment_step_number(self) -> None:
        """Increment the wrapped AppWorld agent step number."""
        target = getattr(self.agent, "agent", self.agent)
        target.step_number = int(getattr(target, "step_number", 0)) + 1

    def _extra_turn_limit_reached(self) -> bool:
        """Return whether post-block extra generation budget is exhausted."""
        if self.first_block_turn is None:
            return False
        return (
            self._runtime_state().agent_turn_count - self.first_block_turn
            >= self.budget.max_extra_agent_turns_after_block
        )

    def _should_continue(self) -> bool:
        """Return whether the runner can take another generation turn."""
        turn_count = self._runtime_state().agent_turn_count
        if turn_count < self.budget.max_agent_turns:
            return True
        if self.first_block_turn is None:
            return False
        return turn_count - self.first_block_turn < self.budget.max_extra_agent_turns_after_block

    def _runtime_state(self) -> Any:
        """Return the controller runtime state if available."""
        return getattr(self.agent, "runtime_state")

    def _blocked_event(self, result: AppWorldInterceptionApplied, world: Any, turn_start: float) -> dict[str, Any]:
        """Create a turn event for a blocked execution."""
        response = result.interception.controller_response
        decision = response.decision if response is not None else None
        return self._event_base(world, turn_start) | {
            "generated_code": self._first_code(result.interception.original_execution_inputs),
            "trigger_detected": result.interception.trigger_result is not None,
            "controller_action": decision.action.value if decision is not None else None,
            "execution_allowed": False,
            "guidance_injected": result.guidance_injected,
            "world_batch_execute_called": False,
            "task_completed": bool(world.task_completed()),
            "agent_input_tokens": None,
            "agent_output_tokens": None,
            "controller_input_tokens": response.input_tokens if response is not None else None,
            "controller_output_tokens": response.output_tokens if response is not None else None,
        }

    def _executed_event(
        self,
        result: Any,
        execution_inputs: Sequence[Any],
        world: Any,
        task_completed: bool,
        turn_start: float,
    ) -> dict[str, Any]:
        """Create a turn event for environment execution."""
        interception = getattr(self.agent, "last_interception_result", None)
        response = getattr(interception, "controller_response", None)
        decision = response.decision if response is not None else None
        return self._event_base(world, turn_start) | {
            "generated_code": self._first_code(execution_inputs),
            "trigger_detected": getattr(interception, "trigger_result", None) is not None,
            "controller_action": decision.action.value if decision is not None else None,
            "execution_allowed": True,
            "guidance_injected": False,
            "world_batch_execute_called": True,
            "task_completed": task_completed,
            "agent_input_tokens": self._usage_value(result, "input_tokens"),
            "agent_output_tokens": self._usage_value(result, "output_tokens"),
            "controller_input_tokens": response.input_tokens if response is not None else None,
            "controller_output_tokens": response.output_tokens if response is not None else None,
        }

    def _empty_code_event(
        self,
        result: Any,
        execution_inputs: Sequence[Any],
        world: Any,
        turn_start: float,
        guidance_injected: bool,
    ) -> dict[str, Any]:
        """Create an event when the agent produced no executable code."""
        return self._event_base(world, turn_start) | {
            "generated_code": self._first_code(execution_inputs),
            "trigger_detected": False,
            "controller_action": None,
            "execution_allowed": False,
            "guidance_injected": guidance_injected,
            "world_batch_execute_called": False,
            "task_completed": bool(world.task_completed()),
            "agent_input_tokens": self._usage_value(result, "input_tokens"),
            "agent_output_tokens": self._usage_value(result, "output_tokens"),
            "controller_input_tokens": None,
            "controller_output_tokens": None,
            "stop_reason": "model_format_failure",
        }

    def _budget_stop_event(
        self,
        result: Any,
        execution_inputs: Sequence[Any],
        world: Any,
        turn_start: float,
        reason: str,
    ) -> dict[str, Any]:
        """Create an event for a budget stop before environment execution."""
        return self._event_base(world, turn_start) | {
            "generated_code": self._first_code(execution_inputs),
            "trigger_detected": False,
            "controller_action": None,
            "execution_allowed": False,
            "guidance_injected": False,
            "world_batch_execute_called": False,
            "task_completed": bool(world.task_completed()),
            "agent_input_tokens": self._usage_value(result, "input_tokens"),
            "agent_output_tokens": self._usage_value(result, "output_tokens"),
            "controller_input_tokens": None,
            "controller_output_tokens": None,
            "termination_reason": reason,
        }

    def _event_base(self, world: Any, turn_start: float) -> dict[str, Any]:
        """Return common event fields for every runner turn."""
        state = self._runtime_state()
        return {
            "agent_turn_index": state.agent_turn_count,
            "environment_execution_index": state.environment_execution_count,
            "controller_call_count": state.total_calls,
            "controller_block_count": state.total_blocks,
            "is_extra_turn_after_block": self.first_block_turn is not None and state.agent_turn_count > self.first_block_turn,
            "termination_reason": None,
            "cumulative_tokens": {
                "input": self.cumulative_input_tokens,
                "output": self.cumulative_output_tokens,
                "total": self.cumulative_input_tokens + self.cumulative_output_tokens,
            },
            "cumulative_cost": self._total_cost(),
            "total_cost": self._total_cost(),
            "wall_time": time.perf_counter() - turn_start,
        }

    def _inject_format_reminder(self) -> bool:
        """Inject one short code-block format reminder."""
        adapter = self.adapter
        if hasattr(adapter, "inject_guidance"):
            return bool(adapter.inject_guidance(getattr(self.agent, "agent", self.agent), FORMAT_REMINDER))
        return False

    def _first_code(self, execution_inputs: Any) -> str:
        """Return the first generated code string."""
        if not execution_inputs:
            return ""
        return str(getattr(execution_inputs[0], "content", ""))

    def _all_execution_inputs_empty(self, execution_inputs: Sequence[Any]) -> bool:
        """Return whether no execution input contains code."""
        return not execution_inputs or all(not str(getattr(item, "content", "")).strip() for item in execution_inputs)

    def _usage_value(self, result: Any, name: str) -> Any:
        """Best-effort extraction from AppWorld Usage objects."""
        try:
            usage = result[1]
        except (TypeError, IndexError):
            return None
        tokens = getattr(usage, "tokens", None)
        if name == "input_tokens" and tokens is not None:
            return getattr(tokens, "input_cache_hit_or_miss", None)
        if name == "output_tokens" and tokens is not None:
            return getattr(tokens, "output", None)
        return getattr(usage, name, None)

    def _usage_tokens(self, usage: Any, name: str) -> Any:
        """Best-effort extraction directly from a Usage object."""
        tokens = getattr(usage, "tokens", None)
        if name == "input_tokens" and tokens is not None:
            return getattr(tokens, "input_cache_hit_or_miss", None)
        if name == "output_tokens" and tokens is not None:
            return getattr(tokens, "output", None)
        return getattr(usage, name, None)

    def _total_cost(self) -> Any:
        """Best-effort extraction of accumulated usage cost."""
        tracker = getattr(self.agent, "usage_tracker", None)
        usage = getattr(tracker, "overall_usage", None)
        cost = getattr(usage, "cost", None)
        return getattr(cost, "total", None)
