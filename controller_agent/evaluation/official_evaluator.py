"""Run and record AppWorld's official evaluator in a separate process."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_official_evaluation(
    appworld_root: Path,
    experiment_name: str,
    task_id: str,
) -> dict[str, Any]:
    """Run the official evaluator and return the task-level result."""
    command = ["uv", "run", "appworld", "evaluate", experiment_name, "--task-id", task_id]
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
    current["official_evaluator"] = official_result
    if lifecycle_error is not None:
        current["known_lifecycle_error"] = lifecycle_error
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return current
