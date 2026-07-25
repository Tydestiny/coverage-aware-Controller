"""Composite trigger dispatcher with per-task budget tracking.

Wraps several trigger detectors and exposes a single ``detect(context)``
method that:

1. Returns ``budget_exhausted`` once the per-task trigger count reaches
   ``config.budget.max_triggers_per_task`` (unless the
   ``min_before_finish_triggers`` floor is not yet met, in which case the
   before_finish detector is forced).
2. Iterates the detectors in ``config.budget.detector_order`` and returns
   the first positive result.

The default config drives the historical behaviour of
``scripts/test_all_triggers_optimized.py``.
"""

from __future__ import annotations

from typing import Any

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.before_finish import BeforeFinishTrigger
from controller_agent.triggers.before_write import BeforeWriteTrigger
from controller_agent.triggers.config import (
    DEFAULT_CONFIG,
    TriggerBudgetConfig,
    TriggerConfig,
)
from controller_agent.triggers.no_progress import NoProgressTrigger


_DEFAULT_DETECTORS: dict[str, Any] = {
    "before_finish": BeforeFinishTrigger,
    "before_write": BeforeWriteTrigger,
    "no_progress": NoProgressTrigger,
}


def _build_detector_registry(
    config: TriggerConfig,
) -> list[tuple[str, Any]]:
    """Instantiate detectors in the order specified by ``config.budget``."""
    registry: list[tuple[str, Any]] = []
    for name in config.budget.detector_order:
        cls = _DEFAULT_DETECTORS.get(name)
        if cls is None:
            raise ValueError(f"Unknown trigger detector: {name!r}")
        # Each detector class accepts a typed subconfig; pass the matching
        # config slice so all overrides from ``TriggerConfig`` propagate.
        if name == "before_finish":
            # BeforeFinishTrigger currently owns a fixed public constructor;
            # it has no configurable state to receive here.
            instance = cls()
        elif name == "before_write":
            instance = cls(config.before_write)
        elif name == "no_progress":
            instance = cls(config.no_progress)
        else:
            instance = cls()
        registry.append((name, instance))
    return registry


class CompositeTriggerDetectorWithBudget:
    """Run multiple trigger detectors with per-task budget tracking."""

    def __init__(
        self,
        config: TriggerBudgetConfig | None = None,
        full_config: TriggerConfig | None = None,
    ) -> None:
        if full_config is None:
            full_config = DEFAULT_CONFIG
        self._config = config or full_config.budget
        self._detectors = _build_detector_registry(full_config)
        self._trigger_count = 0
        self._before_finish_count = 0

    def reset(self) -> None:
        """Reset per-task counters (call between tasks)."""
        self._trigger_count = 0
        self._before_finish_count = 0

    def detect(self, context: TriggerContext) -> TriggerResult:
        if self._trigger_count >= self._config.max_triggers_per_task:
            if self._before_finish_count < self._config.min_before_finish_triggers:
                before_finish_detector = self._detectors[0][1]
                result = before_finish_detector.detect(context)
                if result.should_call_controller:
                    self._before_finish_count += 1
                    return result
            return TriggerResult(False, "budget_exhausted")

        for name, detector in self._detectors:
            result = detector.detect(context)
            if result.should_call_controller:
                self._trigger_count += 1
                if name == "before_finish":
                    self._before_finish_count += 1
                return result

        return TriggerResult(False, "no_trigger")

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def before_finish_count(self) -> int:
        return self._before_finish_count
