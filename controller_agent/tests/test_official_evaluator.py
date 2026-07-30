"""Tests for separate-process official evaluation summary handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from controller_agent.evaluation.official_evaluator import (
    official_evaluator_command,
    run_official_evaluation,
    update_run_summary,
)


def test_official_evaluator_prefers_local_windows_cli(tmp_path: Path, monkeypatch: object) -> None:
    """A local AppWorld virtual environment is the evaluator runtime on Windows."""
    executable = tmp_path / ".venv" / "Scripts" / "appworld.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr("controller_agent.evaluation.official_evaluator.os.name", "nt")  # type: ignore[attr-defined]

    assert official_evaluator_command(tmp_path, "smoke", "task_1") == [
        str(executable), "evaluate", "smoke", "--task-id", "task_1"
    ]


def test_run_official_evaluation_reads_task_result(tmp_path: Path) -> None:
    """The wrapper reads the official task result from evaluations output."""
    report_dir = tmp_path / "experiments" / "outputs" / "smoke" / "evaluations"
    report_dir.mkdir(parents=True)
    report_dir.joinpath("on_only_task_1.json").write_text(
        json.dumps({"aggregate": {"task_goal_completion": 1.0}, "individual": {"task_1": {"success": True}}}),
        encoding="utf-8",
    )
    with patch("controller_agent.evaluation.official_evaluator.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "ok"
        run.return_value.stderr = ""
        result = run_official_evaluation(tmp_path, "smoke", "task_1")

    assert result["available"] is True
    assert result["task_result"] == {"success": True}


def test_update_run_summary_preserves_lifecycle_error(tmp_path: Path) -> None:
    """Known close failures remain visible beside official evaluation."""
    summary = tmp_path / "evaluator_result.json"
    summary.write_text(json.dumps({"in_process": {"available": False}}), encoding="utf-8")

    merged = update_run_summary(
        summary,
        {"available": True, "task_result": {"success": True}},
        {"known": True, "stage": "world.close", "error": "AttributeError"},
    )

    assert merged["in_process"] == {"available": False}
    assert merged["official"]["task_result"]["success"] is True
    assert merged["official_evaluator"]["task_result"]["success"] is True
    assert merged["known_lifecycle_error"]["stage"] == "world.close"
