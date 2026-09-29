"""Detection of imminent AppWorld task completion."""

from __future__ import annotations

import ast

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.code import python_spans


class BeforeFinishTrigger:
    """Trigger once when the Agent emits a supervisor complete_task call."""

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Return a trigger result for a complete_task call."""
        for code, base_offset in python_spans(context.proposed_execution or ""):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            lines = code.splitlines(keepends=True)
            for call in sorted((node for node in ast.walk(tree) if isinstance(node, ast.Call)),
                               key=lambda node: (node.lineno, node.col_offset)):
                target = call.func
                if not isinstance(target, ast.Attribute) or target.attr != "complete_task":
                    continue
                supervisor = target.value
                if isinstance(supervisor, ast.Name) and supervisor.id == "supervisor":
                    pass
                elif (isinstance(supervisor, ast.Attribute) and supervisor.attr == "supervisor"
                      and isinstance(supervisor.value, ast.Name) and supervisor.value.id == "apis"):
                    pass
                else:
                    continue
                return TriggerResult(True, "before_finish", {
                    "matched_text": (ast.get_source_segment(code, target) or "supervisor.complete_task") + "(",
                    "offset": base_offset + sum(len(line) for line in lines[:call.lineno - 1]) + call.col_offset,
                    "task_id": context.task_id,
                })
        return TriggerResult(False, "before_finish_not_detected")
