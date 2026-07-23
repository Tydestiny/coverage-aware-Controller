"""Load replayable trajectory steps from AppWorld output logs."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from controller_agent.branching.schemas import StaticTrajectory, TrajectoryStep


_CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(content: str) -> str | None:
    """Extract the first Python code block from an Agent message."""
    match = _CODE_BLOCK.search(content or "")
    if match is None:
        return None
    return match.group(1).strip()


def load_appworld_logger_jsonl(task_dir: Path) -> StaticTrajectory:
    """Load a StaticTrajectory from an AppWorld task output directory."""
    logger_path = task_dir / "logs" / "logger.jsonl"
    if not logger_path.exists():
        raise FileNotFoundError(f"Missing AppWorld logger file: {logger_path}")
    task_id = task_dir.name
    records = _read_jsonl(logger_path)
    steps: list[TrajectoryStep] = []
    visible_messages: list[dict[str, Any]] = []
    pending_agent: dict[str, Any] | None = None
    pending_before: list[dict[str, Any]] = []
    for record in records:
        role = record.get("role")
        if role == "agent":
            pending_agent = record
            pending_before = copy.deepcopy(visible_messages)
            visible_messages.append(_agent_message(record))
            continue
        if role == "environment" and pending_agent is not None:
            output = str(record.get("content", ""))
            visible_messages.append(_environment_message(output))
            step = TrajectoryStep(
                step_index=len(steps),
                execution_code=extract_python_code(str(pending_agent.get("content", ""))),
                execution_output=output,
                messages_before_step=pending_before,
                messages_after_step=copy.deepcopy(visible_messages),
                metadata={
                    "agent_step_number": pending_agent.get("step_number"),
                    "environment_step_number": record.get("step_number"),
                    "complete_task": "complete_task" in str(pending_agent.get("content", "")),
                },
            )
            steps.append(step)
            pending_agent = None
            pending_before = []
    metadata = _load_versions(task_dir)
    metadata["has_unpaired_final_agent_message"] = pending_agent is not None
    return StaticTrajectory(task_id=task_id, steps=steps, source_path=str(task_dir), metadata=metadata)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records."""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
    return records


def _agent_message(record: dict[str, Any]) -> dict[str, Any]:
    """Convert AppWorld logger agent record to visible message."""
    raw = record.get("raw_message")
    if isinstance(raw, dict) and raw.get("role"):
        return copy.deepcopy(raw)
    return {"role": "assistant", "content": str(record.get("content", ""))}


def _environment_message(output: str) -> dict[str, Any]:
    """Convert AppWorld environment output to visible user message."""
    return {"role": "user", "content": f"Output:\n```\n{output}\n```\n\n"}


def _load_versions(task_dir: Path) -> dict[str, Any]:
    """Load AppWorld code/data version files when present."""
    metadata: dict[str, Any] = {}
    code_path = task_dir / "version" / "code.txt"
    data_path = task_dir / "version" / "data.txt"
    if code_path.exists():
        metadata["appworld_code_version"] = code_path.read_text(encoding="utf-8").strip()
    if data_path.exists():
        metadata["appworld_data_version"] = data_path.read_text(encoding="utf-8").strip()
    metadata["has_final_db_snapshot"] = (task_dir / "dbs").exists()
    metadata["has_lm_calls"] = (task_dir / "logs" / "lm_calls.jsonl").exists()
    metadata["has_api_calls"] = (task_dir / "logs" / "api_calls.jsonl").exists()
    return metadata

