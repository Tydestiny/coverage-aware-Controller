"""Tests for paired checkpoint replay summaries."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.branching.checkpoint_comparison import write_checkpoint_comparison


def test_checkpoint_comparison_writes_required_rows_and_summary(tmp_path: Path) -> None:
    """Controller and no-guidance outcomes are paired by the selected checkpoint."""
    baseline = tmp_path / "baseline"
    _write(baseline / "evaluator_result.json", {"official": {"task_result": {"success": False}}})
    root = tmp_path / "experiment"
    _branch(root, "task-a", "single_intervention__step_0003__before_write", True, 11, "REPLAN")
    _branch(root, "task-a", "no_guidance_replay__step_0003__before_write", False, None, None)
    _branch(root, "task-b", "single_intervention__step_0004__no_progress", False, 7, "VERIFY_TARGET_SET")
    _branch(root, "task-b", "no_guidance_replay__step_0004__no_progress", True, None, None)
    second = tmp_path / "baseline-b"
    _write(second / "evaluator_result.json", {"success": False})
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(json.dumps(row) for row in (
            {"task_id": "task-a", "checkpoint_step": 3, "trigger_type": "before_write", "baseline_run_path": str(baseline)},
            {"task_id": "task-b", "checkpoint_step": 4, "trigger_type": "no_progress", "baseline_run_path": str(second)},
        )) + "\n",
        encoding="utf-8",
    )

    summary = write_checkpoint_comparison(root, candidates)

    assert summary["controller_only_success"] == 1
    assert summary["no_guidance_only_success"] == 1
    assert summary["net_recovery"] == 0
    assert summary["total_controller_tokens"] == 18
    csv_text = (root / "checkpoint_comparison.csv").read_text(encoding="utf-8")
    assert "controller_only_success" in csv_text
    assert "no_guidance_only_success" in csv_text


def _branch(root: Path, task_id: str, name: str, success: bool, tokens: int | None, action: str | None) -> None:
    """Write a minimal completed branch fixture."""
    path = root / "tasks" / task_id / name
    _write(path / "result.json", {"success": success, "tokens": {"controller_total": tokens}})
    calls = "" if action is None else json.dumps({"action": action}) + "\n"
    (path / "controller_calls.jsonl").write_text(calls, encoding="utf-8")


def _write(path: Path, value: dict[str, object]) -> None:
    """Write one fixture JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
