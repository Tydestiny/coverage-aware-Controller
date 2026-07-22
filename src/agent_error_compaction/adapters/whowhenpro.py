"""Defensive adapter for Who&When Pro dataset exports into Canonical Schema v1.1."""

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


def _extract_actor(turn: dict[str, Any]) -> str | None:
    """Extract actor/agent identifier from turn dict."""
    if not isinstance(turn, dict):
        return None
    for key in ("agent", "actor", "name", "sender", "author"):
        if key in turn and turn[key] is not None:
            val = _as_text(turn[key], key)
            if val and val.strip():
                return val.strip()
    
    # 若无显式 agent 字段，尝试从 kind 中提取非通用 Role 的标识符
    kind = turn.get("kind") or turn.get("role")
    if isinstance(kind, str) and kind.strip():
        kind_str = kind.strip()
        if kind_str.lower() not in ("system", "user", "assistant", "tool", "environment", "observation"):
            return kind_str
    return None


def _infer_role(turn: dict[str, Any]) -> str | None:
    """Infer message role."""
    if not isinstance(turn, dict):
        return None
    kind_or_role = turn.get("role") or turn.get("kind")
    if isinstance(kind_or_role, str):
        k = kind_or_role.lower()
        if k in ("system", "user", "assistant", "tool", "environment", "observation"):
            return k
    return "assistant"


def _infer_event_type(turn: dict[str, Any]) -> str:
    """Classify turn into message, action, or observation."""
    if not isinstance(turn, dict):
        return "message"
    explicit = turn.get("event_type") or turn.get("type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    
    kind = str(turn.get("kind") or turn.get("role") or "").lower()
    if kind in ("tool", "environment", "observation") or "observation" in turn or "tool_result" in turn:
        return "observation"
    if any(key in turn for key in ("action", "tool_call", "function_call")):
        return "action"
    return "message"


class WhoWhenProAdapter(BaseTrajectoryAdapter):
    """Convert Who&When Pro dataset records into CanonicalTrajectory Schema v1.1."""

    def __init__(self, source_dataset: str = "WhoAndWhen-Pro") -> None:
        self.source_dataset = source_dataset

    def convert_record(self, record: dict[str, Any]) -> CanonicalTrajectory:
        """Convert a single Who&When Pro JSON dict into CanonicalTrajectory."""
        trajectory_id = _as_text(record.get("id"), "id")
        if not trajectory_id:
            raise ValueError("Who&When Pro record must contain a non-empty 'id'")

        raw_trajectory = record.get("trajectory")
        if not isinstance(raw_trajectory, list):
            raise ValueError(f"trajectory {trajectory_id}: 'trajectory' must be a list")

        canonical_steps: list[CanonicalStep] = []
        for index, turn in enumerate(raw_trajectory):
            if not isinstance(turn, dict):
                turn = {"role": "unknown", "content": str(turn)}

            source_index = turn.get("step", turn.get("index", index))
            if not isinstance(source_index, int):
                source_index = index

            canonical_steps.append(
                CanonicalStep(
                    step_index=index,
                    source_step_index=source_index,
                    source_step_id=f"{trajectory_id}_{index}",
                    actor=_extract_actor(turn),
                    role=_infer_role(turn),
                    event_type=_infer_event_type(turn),
                    content=_as_text(turn.get("content") or turn.get("message"), "content"),
                    thought=_as_text(turn.get("thought") or turn.get("reasoning"), "thought"),
                    action=turn.get("action") or turn.get("tool_call"),
                    observation=turn.get("observation") or turn.get("tool_result"),
                    raw_step=turn,
                    token_count=turn.get("token_count"),
                )
            )

        # 提取 Ground Truth
        gt = record.get("ground_truth") if isinstance(record.get("ground_truth"), dict) else {}
        source_onset_step = gt.get("step")
        failed_agent = _as_text(gt.get("agent"), "failed_agent")
        error_mode = _as_text(gt.get("mode"), "mode")

        is_failure = bool(source_onset_step is not None or failed_agent or error_mode)

        # 计算对应的 0-based canonical step_index
        onset_step = None
        if source_onset_step is not None and isinstance(source_onset_step, int):
            for cs in canonical_steps:
                if cs.source_step_index == source_onset_step:
                    onset_step = cs.step_index
                    break
            if onset_step is None:
                if 0 <= source_onset_step < len(canonical_steps):
                    onset_step = source_onset_step
                elif 0 <= source_onset_step - 1 < len(canonical_steps):
                    onset_step = source_onset_step - 1

        labels = TrajectoryLabels(
            is_failure=is_failure,
            error_onset_step=onset_step,
            source_error_onset_step=source_onset_step if isinstance(source_onset_step, int) else None,
            responsible_actor=failed_agent,
            error_type_original=error_mode,
            error_type_unified=None,
            error_reason=f"Injected/Attributed failure in mode: {error_mode}" if error_mode else None,
            label_source="dataset_author" if is_failure else "safe",
            unsafe_source="controlled_injection" if is_failure else "safe",
        )

        # 1. 明确提取 benchmark 与 framework 变量
        benchmark = _as_text(record.get("benchmark"), "benchmark") or "unknown"
        framework = _as_text(record.get("framework"), "framework") or "unknown"

        # 2. 优化 family_id：优先提取 family/task 标识，无则回退到 trajectory_id，确保细粒度划分
        family_id = _as_text(
            record.get("trajectory_family_id") or record.get("family_id") or record.get("task_id"), 
            "family_id"
        ) or trajectory_id

        task_info = record.get("task") if isinstance(record.get("task"), dict) else {}
        instruction = _as_text(task_info.get("query"), "query") or _as_text(record.get("instruction"), "instruction")
        gold_answer = task_info.get("answer")

        return CanonicalTrajectory(
            trajectory_id=trajectory_id,
            trajectory_family_id=family_id,
            source_dataset=self.source_dataset,
            source_split=_as_text(record.get("split"), "split"),
            domain=benchmark if benchmark != "unknown" else (_as_text(record.get("modality"), "modality") or "agentic"),
            task_id=trajectory_id,
            task={"instruction": instruction, "query": instruction},
            gold_answer=gold_answer,
            outcome="failure" if is_failure else "success",
            tools=record.get("tools"),
            steps=canonical_steps,
            labels=labels,
            metadata={
                "num_steps": len(canonical_steps),
                "framework": framework,
                "benchmark": benchmark,
                "modality": _as_text(record.get("modality"), "modality"),
            },
        )