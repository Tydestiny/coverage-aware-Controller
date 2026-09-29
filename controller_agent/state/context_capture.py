"""Capture and qualify the current-task view at the Agent generate boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def capture_context(
    messages: list[dict[str, Any]],
    history: Any,
    boundary: Any,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Copy visible messages and report verifiable completeness evidence."""
    request = copy.deepcopy(messages)
    valid_history = isinstance(history, list) and all(isinstance(m, dict) for m in history)
    known_boundary = (
        valid_history and type(boundary) is int and 1 <= boundary <= len(request)
        and boundary <= len(history)
        and request[:boundary - 1] == history[:boundary - 1]
        and request[boundary - 1].get("role") == history[boundary - 1].get("role") == "user"
    )
    start = boundary - 1 if known_boundary else 0
    runtime = request[start:]
    reasons: list[str] = []
    if not known_boundary:
        reasons.append("task_boundary_unverified")
    matches = valid_history and runtime == history[start:]
    if not matches:
        reasons.append("request_differs_from_agent_history")
    if any(marker in str(m.get("content", "")) for m in runtime
           for marker in ("[TRIMMED HISTORY]", "[NOT SHOWN FOR BREVITY]")):
        reasons.append("history_truncation_marker")
    metadata = {
        "source": "agent.language_model.generate.messages",
        "scope": "current_task" if known_boundary else "unfiltered_request",
        "task_boundary_verified": bool(known_boundary),
        "excluded_prefix_messages": start,
        "request_message_count": len(request),
        "context_message_count": len(runtime),
        "request_sha256": _digest(request),
        "context_sha256": _digest(runtime),
        "matches_agent_history": bool(matches),
        "incomplete_reasons": reasons,
        "provider_payload_verified": False,
        "proposed_execution_status": "not_executed",
    }
    return runtime, not reasons, metadata


def _digest(messages: list[dict[str, Any]]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
