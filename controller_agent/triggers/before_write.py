"""Detection of high-risk API calls before they are executed.

The keyword set, params preview cap, and other tunables live in
:mod:`controller_agent.triggers.config`.  When the trigger is instantiated
without an explicit config, the package defaults are used so behaviour
matches the historically hard-coded values.
"""

from __future__ import annotations

import re

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.config import DEFAULT_CONFIG, BeforeWriteConfig

# Matches an identifier (possibly dot/underscore chained) followed by "(".
# Greedy so that "apis.orders.delete(" captures the whole chain as group(1).
_API_CALL_RE = re.compile(r"([a-z_][a-z0-9_]*(?:[._][a-z_][a-z0-9_]*)*)\s*\(", re.IGNORECASE)


def _extract_api_name(code: str, keywords: frozenset[str]) -> str | None:
    """Return the first high-risk action name in ``code``, or None.

    Splits the matched identifier on dots and underscores, then checks each
    full token (not partial substrings) against the keyword set.
    """
    m = _API_CALL_RE.search(code)
    if m is None:
        return None
    raw = m.group(1)
    for part in re.split(r"[._]", raw, flags=re.IGNORECASE):
        if part.lower() in keywords:
            return part.lower()
    return None


def _params_preview(code: str, api_name: str, limit: int) -> str:
    """Extract a short preview of the parameters passed to ``api_name``."""
    api_lower = api_name.lower()
    code_lower = code.lower()

    api_pos = code_lower.find(api_lower)
    if api_pos == -1:
        return ""

    paren_pos = code.find("(", api_pos)
    if paren_pos == -1:
        return ""

    depth = 0
    for i in range(paren_pos, len(code)):
        c = code[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                preview = code[paren_pos : i + 1]
                if len(preview) > limit:
                    return preview[:limit] + "..."
                return preview

    preview = code[paren_pos:]
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


class BeforeWriteTrigger:
    """Trigger when the Agent is about to invoke a high-risk write API."""

    def __init__(self, config: BeforeWriteConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG.before_write

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Return a trigger result for a high-risk API call in proposed execution."""
        code = context.proposed_execution or ""
        if not code.strip():
            return TriggerResult(False, "before_write_empty_execution")

        api_name = _extract_api_name(code, self._config.high_risk_keywords)
        if api_name is None:
            return TriggerResult(False, "before_write_not_triggered")

        return TriggerResult(
            True,
            "before_write",
            {
                "trigger": "before_write",
                "api_name": api_name,
                "params_preview": _params_preview(
                    code, api_name, self._config.params_preview_limit
                ),
                "task_id": context.task_id,
            },
        )