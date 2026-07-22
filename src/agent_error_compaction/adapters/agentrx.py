"""Defensive adapter for AgentRx JSON/JSONL dataset records (Magentic-One & Tau-Retail)."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseTrajectoryAdapter
from ..schemas import CanonicalStep, CanonicalTrajectory, TrajectoryLabels


def _as_text(value: Any, field: str) -> str | None:
    """Convert scalar/object values to text defensively."""
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


def _extract_actor(turn: dict[str, Any]) -> str | None:
    """Extract actor identifier from turn or substep dict."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    for key in ("actor", "failed_agent", "agent", "name", "sender", "role"):
        if key in turn and turn[key] is not None:
            val = _as_text(turn[key], key)
            if val and val.strip():
                return val.strip()
    return None


def _extract_content(turn: dict[str, Any]) -> str | None:
    """Extract message content."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    for key in ("content", "message", "text"):
        if key in turn and turn[key] is not None:
            return _as_text(turn[key], key)
    return None


def _extract_thought(turn: dict[str, Any]) -> str | None:
    """Extract thought or reasoning text."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    for key in ("thought", "reasoning", "analysis"):
        if key in turn and turn[key] is not None:
            return _as_text(turn[key], key)
    return None


def _extract_action(turn: dict[str, Any]) -> Any:
    """Extract raw action / tool call field."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    for key in ("action", "tool_call", "function_call", "tool_calls"):
        if key in turn:
            return turn[key]
    return None


def _extract_observation(turn: dict[str, Any]) -> Any:
    """Extract raw observation / tool result field."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    for key in ("observation", "tool_result", "result", "output"):
        if key in turn and turn[key] is not None:
            return turn[key]
    if turn.get("role") in ("environment", "tool"):
        return turn.get("content")
    return None


def _has_value(value: Any) -> bool:
    """Return whether a field contains a non-empty value."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _infer_event_type(turn: dict[str, Any]) -> str:
    """Classify turn into action, observation, or message."""
    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
    explicit = turn.get("event_type") or turn.get("type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if any(key in turn and _has_value(turn[key]) for key in ("action", "tool_call", "function_call", "tool_calls")):
        return "action"
    if any(key in turn and _has_value(turn[key]) for key in ("observation", "tool_result", "result", "output")) or turn.get("role") in ("environment", "tool"):
        return "observation"
    return "message"


class AgentRxAdapter(BaseTrajectoryAdapter):
    """Adapter for AgentRx records (Magentic-One & Tau-Retail) into Canonical Schema v1.1."""

    def __init__(self, source_dataset: str = "AgentRx") -> None:
        self.source_dataset = source_dataset

    def convert_record(self, record: dict[str, Any]) -> CanonicalTrajectory:
        """Convert merged AgentRx record into CanonicalTrajectory."""
        trajectory_id = _as_text(record.get("trajectory_id"), "trajectory_id")
        if not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")

        # 原代码是：raw_steps = record.get("steps")
        # 修改为：兼容 steps, trajectory, turns 三种不同命名
        raw_steps = record.get("steps") or record.get("trajectory") or record.get("turns")
        if not isinstance(raw_steps, list):
            raise ValueError(f"trajectory {trajectory_id}: steps must be a list")
        if not isinstance(raw_steps, list):
            raise ValueError(f"trajectory {trajectory_id}: steps must be a list")

        canonical_steps: list[CanonicalStep] = []
        canonical_idx = 0

        # 1. 平铺 Steps 与 Substeps，构建连续 0-based index
        for step_dict in raw_steps:
            if not isinstance(step_dict, dict):
                raise ValueError(f"trajectory {trajectory_id}: step must be dict")

            source_step_index = step_dict.get("index")
            if source_step_index is None or not isinstance(source_step_index, int):
                source_step_index = canonical_idx + 1

            substeps = step_dict.get("substeps")
            if isinstance(substeps, list) and len(substeps) > 0:
                for sub_dict in substeps:
                    if not isinstance(sub_dict, dict):
                        continue
                    sub_idx = sub_dict.get("sub_index")
                    source_step_id = f"{source_step_index}_{sub_idx}" if sub_idx is not None else str(source_step_index)

                    canonical_steps.append(
                        CanonicalStep(
                            step_index=canonical_idx,
                            source_step_index=source_step_index,
                            source_step_id=source_step_id,
                            actor=_extract_actor(sub_dict),
                            role=_as_text(sub_dict.get("role"), "role"),
                            event_type=_infer_event_type(sub_dict),
                            content=_extract_content(sub_dict),
                            thought=_extract_thought(sub_dict),
                            action=_extract_action(sub_dict),
                            observation=_extract_observation(sub_dict),
                            raw_step={"parent_step_index": source_step_index, **sub_dict},
                            token_count=sub_dict.get("token_count"),
                        )
                    )
                    canonical_idx += 1
            else:
                source_step_id = _as_text(step_dict.get("step_id"), "step_id") or str(source_step_index)
                canonical_steps.append(
                    CanonicalStep(
                        step_index=canonical_idx,
                        source_step_index=source_step_index,
                        source_step_id=source_step_id,
                        actor=_extract_actor(step_dict),
                        role=_as_text(step_dict.get("role"), "role"),
                        event_type=_infer_event_type(step_dict),
                        content=_extract_content(step_dict),
                        thought=_extract_thought(step_dict),
                        action=_extract_action(step_dict),
                        observation=_extract_observation(step_dict),
                        raw_step=step_dict,
                        token_count=step_dict.get("token_count"),
                    )
                )
                canonical_idx += 1

        # 2. 解析失败与根因标注
        annotation_data = record.get("annotation") or record
        failures = annotation_data.get("failures") or []
        root_cause = annotation_data.get("root_cause") or {}
        root_cause_id = str(root_cause.get("failure_id") or annotation_data.get("root_cause_failure_id") or "")

        is_failure = bool(failures) or bool(root_cause)
        onset_step = None
        source_onset_step = None
        mistake_agent = None
        mistake_type = None
        mistake_reason = None

        if is_failure and failures:
            root_cause_failure = next((f for f in failures if str(f.get("failure_id")) == root_cause_id), failures[0])
            source_onset_step = root_cause_failure.get("step_number")

            # 寻找对应的 0-based canonical step_index
            if source_onset_step is not None:
                for cs in canonical_steps:
                    if cs.source_step_index == source_onset_step:
                        onset_step = cs.step_index
                        break

            mistake_agent = _as_text(root_cause_failure.get("failed_agent"), "failed_agent")
            mistake_type = _as_text(root_cause_failure.get("failure_category"), "failure_category")
            mistake_reason = _as_text(root_cause_failure.get("step_reason") or root_cause.get("reason_for_root_cause"), "reason")

        # 3. 封装 TrajectoryLabels 与 CanonicalTrajectory
        labels = TrajectoryLabels(
            is_failure=is_failure,
            error_onset_step=onset_step,
            source_error_onset_step=source_onset_step,
            responsible_actor=mistake_agent,
            error_type_original=mistake_type,
            error_type_unified=None,
            error_reason=mistake_reason,
            label_source="dataset_author" if is_failure else "safe",
            unsafe_source=_as_text(record.get("unsafe_source"), "unsafe_source") or ("post_hoc_diagnosis" if is_failure else "safe"),
        )

        domain = _as_text(record.get("domain"), "domain")
        if not domain:
            domain = "retail" if "tau_retail" in trajectory_id else "agentic"

        family_id = _as_text(record.get("trajectory_family_id") or record.get("family_id"), "family_id") or trajectory_id

        return CanonicalTrajectory(
            trajectory_id=trajectory_id,
            trajectory_family_id=family_id,
            source_dataset=self.source_dataset,
            source_split=_as_text(record.get("split"), "split"),
            domain=domain,
            task_id=_as_text(record.get("task_id"), "task_id"),
            task={"instruction": _as_text(record.get("instruction"), "instruction")},
            gold_answer=record.get("gold_answer"),
            outcome="failure" if is_failure else "success",
            tools=record.get("tools"),
            steps=canonical_steps,
            labels=labels,
            metadata={
                "num_steps": len(canonical_steps),
                "source_turn_count": len(raw_steps),
                "failure_summary": _as_text(annotation_data.get("failure_summary"), "failure_summary"),
                "raw_failures_count": len(failures),
            },
        )