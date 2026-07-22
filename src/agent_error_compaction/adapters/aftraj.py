"""Defensive adapter for inspected AFTraj JSON or JSONL exports."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseTrajectoryAdapter
from ..schemas import CanonicalStep, CanonicalTrajectory, TrajectoryLabels


def _as_text(value: Any, field: str) -> str | None:
    """Convert common scalar/object text forms without discarding source data."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("content", "text", "message", "value", "name"):
            if isinstance(value.get(key), str):
                return value[key]
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"{field} has unsupported type {type(value).__name__}")


def _extract_actor(turn: Any) -> str | None:
    """Extract an actor from common AFTraj turn representations."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    return _as_text(next((turn[key] for key in ("actor", "agent", "name", "sender", "role") if key in turn), None), "actor")


def _extract_content(turn: Any) -> str | None:
    """Extract message content from a source turn."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    return _as_text(next((turn[key] for key in ("content", "message", "text") if key in turn), None), "content")


def _extract_thought(turn: Any) -> str | None:
    """Extract reasoning text when a source field explicitly provides it."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    return _as_text(next((turn[key] for key in ("thought", "reasoning", "analysis") if key in turn), None), "thought")


def _extract_action(turn: Any) -> Any:
    """Return the source action unchanged for string, object, null, and scalar forms."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    return next((turn[key] for key in ("action", "tool_call", "function_call") if key in turn), None)


def _extract_observation(turn: Any) -> Any:
    """Return an explicit source observation unchanged when present."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    explicit = next((turn[key] for key in ("observation", "tool_result", "result", "output") if key in turn and _has_value(turn[key])), None)
    if explicit is not None:
        return explicit
    if turn.get("role") == "environment":
        return turn.get("content")
    return None


def _has_value(value: Any) -> bool:
    """Return whether a source field contains a non-empty value."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _infer_event_type(turn: Any) -> str:
    """Classify a turn only from explicitly present fields."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be an object, got {type(turn).__name__}")
    explicit = turn.get("event_type") or turn.get("type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if any(key in turn and _has_value(turn[key]) for key in ("action", "tool_call", "function_call")):
        return "action"
    if any(key in turn and _has_value(turn[key]) for key in ("observation", "tool_result", "result", "output")) or turn.get("role") == "environment":
        return "observation"
    return "message"


class AFTrajAdapter(BaseTrajectoryAdapter):
    """Convert validated AFTraj JSON/JSONL exports without label inference."""

    def __init__(self, mistake_step_base: int = 0, source_dataset: str = "AFTraj-2K") -> None:
        """Configure the manually confirmed source indexing convention."""
        if mistake_step_base != 0:
            raise ValueError("AFTraj-2K mistake_step_base is fixed at 0")
        self.mistake_step_base = mistake_step_base
        self.source_dataset = source_dataset

    def convert_record(self, record: dict[str, Any]) -> CanonicalTrajectory:
        """Convert one AFTraj record while preserving every source turn verbatim."""
        trajectory_id = _as_text(record.get("conv_id"), "conv_id")
        if not trajectory_id:
            raise ValueError("conv_id must be a non-empty string-like value")
        turns = record.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"trajectory {trajectory_id}: turns must be a list, got {type(turns).__name__}")
        steps: list[CanonicalStep] = []
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"trajectory {trajectory_id}: turns[{index}] must be an object")
            source_index = turn.get("step_index", turn.get("turn_index", turn.get("index", index)))
            if not isinstance(source_index, int):
                raise ValueError(f"trajectory {trajectory_id}: turns[{index}] source index must be an integer")
            source_id = _as_text(next((turn[key] for key in ("step_id", "turn_id", "id") if key in turn), None), "source_step_id")
            token_count = turn.get("token_count")
            if token_count is not None and (not isinstance(token_count, int) or isinstance(token_count, bool)):
                raise ValueError(f"trajectory {trajectory_id}: turns[{index}].token_count must be int or null")
            steps.append(CanonicalStep(index, source_index, source_id, _extract_actor(turn), _as_text(turn.get("role"), "role"), _infer_event_type(turn), _extract_content(turn), _extract_thought(turn), _extract_action(turn), _extract_observation(turn), turn, token_count))

        source_onset = record.get("mistake_step")
        if source_onset is not None and (not isinstance(source_onset, int) or isinstance(source_onset, bool)):
            raise ValueError(f"trajectory {trajectory_id}: mistake_step must be int or null")
        explicit_outcome = record.get("outcome")
        if explicit_outcome is not None and explicit_outcome not in {"success", "failure"}:
            raise ValueError(f"trajectory {trajectory_id}: outcome must be success or failure")
        is_failure = bool(source_onset is not None) if explicit_outcome is None else explicit_outcome == "failure"
        if not is_failure and source_onset is not None:
            raise ValueError(f"trajectory {trajectory_id}: success outcome conflicts with mistake_step")
        onset = source_onset - self.mistake_step_base if source_onset is not None else None
        unsafe_source = _as_text(record.get("unsafe_source"), "unsafe_source") or ("safe" if not is_failure else None)
        labels = TrajectoryLabels(is_failure, onset, source_onset, _as_text(record.get("mistake_agent"), "mistake_agent"), _as_text(record.get("mistake_type"), "mistake_type"), None, _as_text(record.get("mistake_reason"), "mistake_reason"), unsafe_source, unsafe_source)
        
        # 核心修复点：防防御式提取 family_id，无则自动回退为 trajectory_id
        family_id = _as_text(
            record.get("trajectory_family_id") or record.get("family_id") or record.get("task_id"),
            "family_id"
        ) or trajectory_id

        return CanonicalTrajectory(
            trajectory_id=trajectory_id,
            trajectory_family_id=family_id,  # 赋值为计算后的 family_id
            source_dataset=self.source_dataset,
            source_split=_as_text(record.get("split"), "split"),
            domain=_as_text(record.get("domain"), "domain"),
            task_id=_as_text(record.get("task_id"), "task_id"),
            task=record.get("task"),
            gold_answer=record.get("gold_answer"),
            outcome="failure" if is_failure else "success",
            tools=record.get("tools"),
            steps=steps,
            labels=labels,
            metadata={
                "num_steps": len(steps),
                "source_turn_count": len(turns),
                "mistake_step_base": self.mistake_step_base,
                "source_record": {key: record[key] for key in ("trajectory_family_id", "parent_trajectory_id", "parent_id", "related_trajectory_id", "family_id") if key in record},
            },
        )