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
