"""Stable, model-agnostic data contracts for the Controller layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    """Actions the Controller may return."""

    PROCEED = "PROCEED"
    VERIFY_TARGET_SET = "VERIFY_TARGET_SET"
    LOOKUP_API_SPEC = "LOOKUP_API_SPEC"
    REPLAN = "REPLAN"
    BLOCK_RISKY_ACTION = "BLOCK_RISKY_ACTION"


@dataclass(frozen=True)
class TriggerResult:
    """Result of a trigger detector."""

    should_call_controller: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriggerContext:
    """Runtime context available to trigger detectors."""

    task_id: str
    step_index: int
    task: str
    proposed_execution: str
    last_execution_outputs: list[Any]
    recent_history: list[dict[str, Any]]
    runtime_state: dict[str, Any]


@dataclass(frozen=True)
class ControllerInput:
    """Information visible to a Controller decision."""

    task: str
    trigger_reason: str
    recent_history: list[dict[str, Any]]
    current_agent_output: str
    known_entities: list[str]
    completed_operations: list[str]
    recent_errors: list[str]


@dataclass(frozen=True)
class ControllerDecision:
    """Validated Controller action and optional guidance."""

    action: Action
    reason: str
    instruction: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ControllerResponse:
    """Controller decision plus optional usage and raw response metadata."""

    decision: ControllerDecision
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None
    cost_cny: float | None = None
    raw_response: str | None = None


@dataclass
class ControllerRuntimeState:
    """Mutable per-task Controller accounting state."""

    total_calls: int = 0
    total_blocks: int = 0
    consecutive_blocks: int = 0
    agent_turn_count: int = 0
    environment_execution_count: int = 0
    last_trigger_step: int | None = None
    last_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        return {
            "total_calls": self.total_calls,
            "total_blocks": self.total_blocks,
            "consecutive_blocks": self.consecutive_blocks,
            "agent_turn_count": self.agent_turn_count,
            "environment_execution_count": self.environment_execution_count,
            "last_trigger_step": self.last_trigger_step,
            "last_action": self.last_action,
        }


@dataclass(frozen=True)
class ControllerConfig:
    """Static Controller runtime limits."""

    enabled: bool = True
    max_controller_calls_per_task: int | None = None
    max_controller_blocks: int | None = None
    max_consecutive_blocks: int | None = None
    fail_open_on_controller_error: bool = True


@dataclass(frozen=True)
class InterceptionResult:
    """Core decision about whether proposed execution may continue."""

    allow_execution: bool
    original_result: Any
    original_execution_inputs: Any
    guidance_message: str | None = None
    trigger_result: TriggerResult | None = None
    controller_response: ControllerResponse | None = None
    block_reason: str | None = None


def decision_from_mapping(value: dict[str, Any]) -> ControllerDecision:
    """Validate an untrusted mapping returned by a Controller backend."""
    try:
        action = Action(str(value["action"]).upper())
        reason = str(value["reason"])
        instruction = str(value.get("instruction", ""))
        confidence = float(value.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Controller decision: {value!r}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Controller confidence must be between 0 and 1.")
    return ControllerDecision(action, reason, instruction, confidence)


def response_from_mapping(value: dict[str, Any]) -> ControllerResponse:
    """Validate a Controller response mapping, including legacy decision maps."""
    decision_value = value.get("decision")
    decision = decision_from_mapping(decision_value if isinstance(decision_value, dict) else value)
    return ControllerResponse(
        decision=decision,
        model_name=_optional_str(value.get("model_name")),
        input_tokens=_optional_int(value.get("input_tokens")),
        output_tokens=_optional_int(value.get("output_tokens")),
        latency_seconds=_optional_float(value.get("latency_seconds")),
        cost_cny=_optional_float(value.get("cost_cny")),
        raw_response=_optional_str(value.get("raw_response")),
    )


def _optional_str(value: Any) -> str | None:
    """Convert optional metadata to string."""
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    """Convert optional metadata to int."""
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    """Convert optional metadata to float."""
    return None if value is None else float(value)
