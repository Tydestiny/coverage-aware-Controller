"""Controller backend adapter with safe parsing and fallback behavior."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from controller_agent.state.schemas import Action, ControllerDecision, ControllerInput, decision_from_mapping


class Controller:
    """Convert a backend response into a validated ControllerDecision."""

    def __init__(self, backend: Callable[[ControllerInput], Any] | None = None) -> None:
        self.backend = backend

    def decide(self, controller_input: ControllerInput) -> ControllerDecision:
        """Return a decision, failing open to original Agent behavior."""
        if self.backend is None:
            return ControllerDecision(Action.PROCEED, "controller_disabled", confidence=1.0)
        try:
            response = self.backend(controller_input)
            if isinstance(response, ControllerDecision):
                return response
            if isinstance(response, str):
                response = json.loads(response)
            if not isinstance(response, dict):
                raise ValueError("Controller backend must return a mapping or JSON object.")
            return decision_from_mapping(response)
        except Exception as exc:
            return ControllerDecision(
                Action.PROCEED,
                "controller_failure_fallback",
                f"Controller failure; preserve original Agent behavior: {type(exc).__name__}",
                0.0,
            )
