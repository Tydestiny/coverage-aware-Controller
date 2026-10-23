"""Detection of high-risk API calls before they are executed.

The keyword set, params preview cap, and other tunables live in
:mod:`controller_agent.triggers.config`.  When the trigger is instantiated
without an explicit config, the current package defaults are used.
"""

from __future__ import annotations

import ast

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.code import python_spans
from controller_agent.triggers.config import DEFAULT_CONFIG, BeforeWriteConfig

def _extract_api_call(
    code: str, keywords: frozenset[str], read_only_api_names: frozenset[str],
) -> tuple[str, ast.Call] | None:
    """Find the first high-risk AppWorld API call in parsed Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in sorted(
        (item for item in ast.walk(tree) if isinstance(item, ast.Call)),
        key=lambda item: (item.lineno, item.col_offset),
    ):
        parts: list[str] = []
        target = node.func
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        parts.reverse()
        if len(parts) < 3 or parts[0] != "apis":
            continue
        if ".".join(parts[1:]).lower() in read_only_api_names:
            continue
        for part in parts[-1].split("_"):
            if part.lower() in keywords:
                return part.lower(), node
    return None


def _params_preview(code: str, call: ast.Call, limit: int) -> str:
    """Return arguments for the matched call, not an earlier text occurrence."""
    segment = ast.get_source_segment(code, call) or ""
    preview = segment[segment.find("("):] if "(" in segment else ""
    return preview[:limit] + "..." if len(preview) > limit else preview


class BeforeWriteTrigger:
    """Trigger when the Agent is about to invoke a high-risk write API."""

    def __init__(self, config: BeforeWriteConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG.before_write

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Return a trigger result for a high-risk API call in proposed execution."""
        code = context.proposed_execution or ""
        if not code.strip():
            return TriggerResult(False, "before_write_empty_execution")

        for source, _ in python_spans(code):
            matched = _extract_api_call(
                source, self._config.high_risk_keywords, self._config.read_only_api_names,
            )
            if matched is None:
                continue
            api_name, call = matched
            return TriggerResult(True, "before_write", {
                "trigger": "before_write",
                "api_name": api_name,
                "params_preview": _params_preview(source, call, self._config.params_preview_limit),
                "task_id": context.task_id,
            })
        return TriggerResult(False, "before_write_not_triggered")
