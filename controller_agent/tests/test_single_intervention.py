"""Tests for single-intervention candidate discovery and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.branching.single_intervention import (
    SingleInterventionCandidate,
    branch_directory,
    scan_failed_baselines,
    scan_successful_baselines,
    summarize_single_interventions,
)
from scripts.run_live_controller_checkpoint_smoke import FirstActionCheckpointTrigger
from scripts.run_single_intervention_branches import _execution_pairs
from controller_agent.state.schemas import TriggerContext, TriggerResult


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_scan_only_failed_baselines_and_records_trigger_metadata(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    failed = root / "tasks" / "task_failed" / "task_failed__baseline"
    passed = root / "tasks" / "task_passed" / "task_passed__baseline"
    _write_json(failed / "manifest.json", {"task_id": "task_failed"})
    _write_json(
        failed / "evaluator_result.json",
        {"official": {"task_result": {"success": False}}},
    )
    _write_jsonl(
        failed / "continuation.jsonl",
        [
            {"step_number": 7, "agent_call_id": "agent_call_0007", "generated_code": "print(apis.spotify.show_library())", "environment_observation": ["[]"]},
            {"step_number": 8, "agent_call_id": "agent_call_0008", "generated_code": "apis.supervisor.complete_task(answer='x')", "environment_observation": None},
        ],
    )
    _write_json(passed / "manifest.json", {"task_id": "task_passed"})
    _write_json(passed / "evaluator_result.json", {"official": {"task_result": {"success": True}}})
    _write_jsonl(passed / "continuation.jsonl", [{"step_number": 1, "generated_code": "apis.supervisor.complete_task()"}])

    candidates, summary = scan_failed_baselines(root)

    assert summary["failed_baseline_count"] == 1
    assert summary["checkpoint_count"] == 1
    candidate = candidates[0]
    assert candidate.task_id == "task_failed"
    assert candidate.checkpoint_step == 8
    assert candidate.trigger_type == "before_finish"
    assert candidate.trigger_metadata["trigger"] if "trigger" in candidate.trigger_metadata else candidate.trigger_metadata["matched_text"]
    assert candidate.proposed_execution.startswith("apis.supervisor.complete_task")


def test_scan_successful_baselines_excludes_failed_runs(tmp_path: Path) -> None:
    """Successful-baseline scans select only evaluator-confirmed successes."""
    root = tmp_path / "experiment"
    failed = root / "tasks" / "failed" / "failed__baseline"
    passed = root / "tasks" / "passed" / "passed__baseline"
    _write_json(failed / "manifest.json", {"task_id": "failed"})
    _write_json(failed / "evaluator_result.json", {"official": {"task_result": {"success": False}}})
    _write_jsonl(failed / "continuation.jsonl", [{"step_number": 1, "generated_code": "apis.supervisor.complete_task()"}])
    _write_json(passed / "manifest.json", {"task_id": "passed"})
    _write_json(passed / "evaluator_result.json", {"official": {"task_result": {"success": True}}})
    _write_jsonl(passed / "continuation.jsonl", [{"step_number": 3, "generated_code": "apis.supervisor.complete_task()"}])

    candidates, summary = scan_successful_baselines(root)

    assert summary["successful_baseline_count"] == 1
    assert [candidate.task_id for candidate in candidates] == ["passed"]


def test_candidate_branch_directories_do_not_overlap_existing_branches(tmp_path: Path) -> None:
    candidate = SingleInterventionCandidate(
        task_id="a",
        checkpoint_step=12,
        checkpoint_id="a_pre_action_0012",
        trigger_type="before_write",
        trigger_metadata={},
        proposed_execution="apis.gmail.delete_thread(thread_id=1)",
        agent_call_id="agent_call_0012",
        baseline_event_index=11,
        baseline_run_path="baseline",
        source_appworld_task_path=None,
    )
    path = branch_directory(tmp_path / "a", candidate)
    assert path.name == "single_intervention__step_0012__before_write"
    assert "baseline" not in path.name
    assert "controller" not in path.name


def test_summary_reads_each_independent_branch_result(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    first = root / "tasks" / "a" / "single_intervention__step_0002__before_finish" / "result.json"
    second = root / "tasks" / "a" / "single_intervention__step_0003__before_write" / "result.json"
    _write_json(first, {"task_id": "a", "trigger_type": "before_finish", "status": "completed", "success": True, "restore_success": True, "tokens": {"agent_total": 20, "controller_total": 5}})
    _write_json(second, {"task_id": "a", "trigger_type": "before_write", "status": "completed", "success": False, "restore_success": True, "tokens": {"agent_total": 30, "controller_total": 5}})

    summary = summarize_single_interventions(root)

    assert summary["branch_count"] == 2
    assert summary["single_intervention_success_rate"] == 0.5
    assert summary["tasks_with_at_least_one_restored_checkpoint"] == 1
    assert summary["trigger_results"]["before_finish"]["success_rate"] == 1.0


def test_controller_once_trigger_never_dispatches_a_follow_up_controller_call() -> None:
    class Fallback:
        def detect(self, context: TriggerContext) -> TriggerResult:
            return TriggerResult(True, "before_finish")

    context = TriggerContext("task", 1, "task", "code", [], [], {})
    trigger = FirstActionCheckpointTrigger(Fallback(), "before_write", disable_after_first=True)

    assert trigger.detect(context).should_call_controller is True
    second = trigger.detect(context)
    assert second.should_call_controller is False
    assert second.reason == "single_intervention_complete"


def test_raw_logger_pairs_keep_a_final_unpaired_complete_task() -> None:
    records = [
        {"role": "agent", "step_number": 1, "content": "```python\nprint('a')\n```"},
        {"role": "environment", "content": "a"},
        {"role": "agent", "step_number": 2, "content": "```python\napis.supervisor.complete_task()\n```"},
    ]

    pairs = _execution_pairs(records)

    assert [pair["step_number"] for pair in pairs] == [1, 2]
    assert pairs[0]["code"] == "print('a')"
    assert pairs[1]["observation"] is None
