"""Tests for the minimal durable checkpoint-branch artifact set."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.branching.artifact_compaction import compact_completed_branch


def test_compaction_preserves_audit_summaries_and_removes_duplicate_debug_files(tmp_path: Path) -> None:
    """A completed branch retains one authoritative copy of each audit fact."""
    branch = tmp_path / "branch"
    raw = branch / "raw_logs"
    raw.mkdir(parents=True)
    _write(branch / "manifest.json", {"task_id": "task-1"})
    _write(branch / "checkpoint_info.json", {"checkpoint_id": "task-1_post_0003", "checkpoint_step": 3})
    _write(branch / "result.json", {"success": True})
    _write(branch / "restore_result.json", {"success": True, "mode": "replay"})
    _write(branch / "lifecycle_errors.json", {"errors": ["known lifecycle issue"]})
    _write(branch / "runtime_error.json", {"type": "RuntimeError"})
    for name in ("config.json", "turn_events.jsonl", "sanitized_messages.jsonl"):
        (branch / name).write_text("debug\n", encoding="utf-8")
    (branch / "controller_events.jsonl").write_text("controller event\n", encoding="utf-8")
    for name in ("logger.jsonl", "lm_calls.jsonl", "api_calls.jsonl", "environment_io.md", "logger.log", "usage.json"):
        (raw / name).write_text("raw\n", encoding="utf-8")
    for name in ("continuation.jsonl", "controller_calls.jsonl", "token_usage.jsonl", "evaluator_result.json"):
        (branch / name).write_text("{}\n", encoding="utf-8")

    compact_completed_branch(branch)

    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((branch / "result.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["checkpoint_step"] == 3
    assert result["restore"]["mode"] == "replay"
    assert result["lifecycle_errors"] == ["known lifecycle issue"]
    assert result["runtime_error"]["type"] == "RuntimeError"
    assert (raw / "logger.jsonl").is_file()
    assert (raw / "lm_calls.jsonl").is_file()
    assert (raw / "api_calls.jsonl").is_file()
    assert (raw / "logger.log").is_file()
    assert (branch / "continuation.jsonl").is_file()
    assert (branch / "controller_calls.jsonl").is_file()
    assert (branch / "controller_events.jsonl").is_file()
    assert (branch / "token_usage.jsonl").is_file()
    assert (branch / "evaluator_result.json").is_file()
    assert not (raw / "environment_io.md").exists()
    assert not (raw / "usage.json").exists()
    assert not (branch / "turn_events.jsonl").exists()
    assert not (branch / "checkpoint_info.json").exists()
    assert not (branch / "restore_result.json").exists()


def _write(path: Path, value: dict[str, object]) -> None:
    """Write one fixture JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
