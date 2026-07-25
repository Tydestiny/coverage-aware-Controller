"""Controller backend adapter with safe parsing and fallback behavior."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from controller_agent.state.schemas import (
    Action,
    ControllerDecision,
    ControllerInput,
    ControllerResponse,
    decision_from_mapping,
    response_from_mapping,
)


class Controller:
    """Convert a backend response into a validated ControllerResponse."""

    def __init__(self, backend: Callable[[ControllerInput], Any] | None = None) -> None:
        self.backend = backend
        self.last_response: ControllerResponse | None = None

    def decide(self, controller_input: ControllerInput) -> ControllerResponse:
        """Return a response, failing open to original Agent behavior."""
        if self.backend is None:
            self.last_response = ControllerResponse(ControllerDecision(Action.PROCEED, "controller_disabled"))
            return self.last_response
        try:
            response = self.backend(controller_input)
            if isinstance(response, ControllerResponse):
                self.last_response = response
                return response
            if isinstance(response, ControllerDecision):
                self.last_response = ControllerResponse(response)
                return self.last_response
            if isinstance(response, str):
                response = json.loads(response)
            if not isinstance(response, dict):
                raise ValueError("Controller backend must return a mapping or JSON object.")
            if "decision" in response:
                self.last_response = response_from_mapping(response)
                return self.last_response
            self.last_response = ControllerResponse(decision_from_mapping(response))
            return self.last_response
        except Exception as exc:
            self.last_response = ControllerResponse(
                ControllerDecision(
                    Action.PROCEED,
                    "controller_failure_fallback",
                    f"Controller failure; preserve original Agent behavior: {type(exc).__name__}",
                )
            )
            return self.last_response
