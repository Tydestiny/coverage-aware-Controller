"""Run and record AppWorld's official evaluator in a separate process."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def official_evaluator_command(appworld_root: Path, experiment_name: str, task_id: str) -> list[str]:
    """Select the installed AppWorld CLI rather than an unrelated uv environment."""
    if os.name == "nt":
        executable = appworld_root / ".venv" / "Scripts" / "appworld.exe"
        if executable.is_file():
            return [str(executable), "evaluate", experiment_name, "--task-id", task_id]
    return ["uv", "run", "appworld", "evaluate", experiment_name, "--task-id", task_id]


def run_official_evaluation(
    appworld_root: Path,
    experiment_name: str,
    task_id: str,
) -> dict[str, Any]:
    """Run the official evaluator and return the task-level result."""
    command = official_evaluator_command(appworld_root, experiment_name, task_id)
    completed = subprocess.run(
        command,
        cwd=appworld_root,
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "available": False,
    }
    if completed.returncode != 0:
        return result

    evaluation_directory = appworld_root / "experiments" / "outputs" / experiment_name / "evaluations"
    matches = sorted(evaluation_directory.glob(f"*_{task_id}.json"), key=lambda path: path.stat().st_mtime)
    if not matches:
        result["error"] = f"Official evaluator completed but no result JSON found for {task_id}."
        return result

    report_path = matches[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    individual = report.get("individual", {}).get(task_id)
    result.update(
        {
            "available": isinstance(individual, dict),
            "report_path": str(report_path),
            "aggregate": report.get("aggregate"),
            "task_result": individual,
        }
    )
    return result


def update_run_summary(
    summary_path: Path,
    official_result: dict[str, Any],
    lifecycle_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge official evaluation and lifecycle metadata into a run summary."""
    current: dict[str, Any] = {}
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    # ``official`` is the canonical evaluator_result.json field consumed by
    # experiment analysis. Retain the older name for existing smoke reports.
    current["official"] = official_result
    current["official_evaluator"] = official_result
    if lifecycle_error is not None:
        current["known_lifecycle_error"] = lifecycle_error
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return current
