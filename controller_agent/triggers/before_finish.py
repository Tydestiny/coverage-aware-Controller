"""Detection of imminent AppWorld task completion."""

from __future__ import annotations

import re

from controller_agent.state.schemas import TriggerContext, TriggerResult


class BeforeFinishTrigger:
    """Trigger once when the Agent emits a supervisor complete_task call."""

    _pattern = re.compile(r"\b(?:apis\.)?supervisor\.complete_task\s*\(", re.IGNORECASE)

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Return a trigger result for a complete_task call."""
        match = self._pattern.search(context.proposed_execution or "")
        if match is None:
            return TriggerResult(False, "before_finish_not_detected")
        return TriggerResult(
            True,
            "before_finish",
            {"matched_text": match.group(0), "offset": match.start(), "task_id": context.task_id},
        )
