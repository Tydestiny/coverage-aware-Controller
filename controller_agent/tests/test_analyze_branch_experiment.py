"""Tests for read-only baseline/controller experiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_branch_experiment import analyze


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _branch(root: Path, task: str, branch: str, *, split: str = "normal", run_id: str | None = "r1", seed: str | None = None, success: bool | None = True, score: int | None = 100, passes: int | None = 3, total: int | None = 4, agent_tokens: int | None = 10, calls: list[dict[str, object]] | None = None, location: str = "") -> None:
    path = root / "runs" / f"{task}_{branch}_{run_id or seed or 'default'}{location}"
    manifest = {"task_id": task, "branch": branch, "split": split, "run_id": run_id}
    if seed is not None:
        manifest["seed"] = seed
    _write_json(path / "manifest.json", manifest)
    _write_json(path / "result.json", {"task_id": task, "branch": branch})
    official: dict[str, object] = {"task_result": {"num_tests": total, "passes": [{}] * (passes or 0)}}
    if score is not None:
        official["aggregate"] = {"task_goal_completion": score}
    if success is not None:
        official["success"] = success
    _write_json(path / "evaluator_result.json", {"official": official})
    if agent_tokens is not None:
        (path / "token_usage.jsonl").write_text(json.dumps({"source": "agent", "total_tokens": agent_tokens}) + "\n", encoding="utf-8")
    (path / "controller_calls.jsonl").write_text("" if calls is None else "".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")


def test_analyze_all_outcome_categories_and_token_totals(tmp_path: Path) -> None:
    """The analyzer pairs four outcomes and separates Agent/Controller tokens."""
    root = tmp_path / "experiment"
    _write_json(root / "manifest.json", {"agent_model": "agent", "controller_model": "controller"})
    cases = [("a", False, True), ("b", True, False), ("c", True, True), ("d", False, False)]
    for task, baseline, controller in cases:
        _branch(root, task, "baseline", success=baseline, score=40 if not baseline else 100, agent_tokens=10)
        _branch(root, task, "controller", success=controller, score=80 if controller else 20, agent_tokens=12, calls=[{"fail_open": False, "record_origin": "runtime_direct", "usage": {"input_tokens": 3, "output_tokens": 2}}])
    payload = analyze(root, tmp_path / "analysis")
    summary = payload["summary"]["overall"]
    assert summary["outcome_categories"] == {"fail_to_success": 1, "success_to_fail": 1, "success_to_success": 1, "fail_to_fail": 1}
    assert summary["baseline_total_tokens"] == 40
    assert summary["controller_agent_total_tokens"] == 48
    assert summary["controller_total_tokens"] == 20
    assert summary["controller_branch_total_tokens"] == 68


def test_fail_to_fail_can_improve_score_and_untriggered_controller_is_false(tmp_path: Path) -> None:
    """Evaluator score changes are retained even when both runs fail."""
    root = tmp_path / "experiment"
    _branch(root, "task", "baseline", success=False, score=20, calls=[])
    _branch(
        root,
        "task",
        "controller",
        success=False,
        score=60,
        calls=[
            {"fail_open": False, "usage": {"total_tokens": 1}},
            {"fail_open": False, "usage": {"total_tokens": 2}},
        ],
    )
    row = analyze(root, tmp_path / "analysis")["rows"][0]
    assert row["outcome_category"] == "fail_to_fail"
    assert row["score_delta"] == 40
    assert row["controller_triggered"] is True
    assert row["controller_call_count"] == 2
    assert row["controller_tokens"] == 3


def test_groups_missing_fields_duplicates_and_seed_pairing(tmp_path: Path) -> None:
    """Quality output excludes bad pairs and summary exposes normal/challenge groups."""
    root = tmp_path / "experiment"
    _branch(root, "normal_task", "baseline", split="normal", run_id="seed-1")
    _branch(root, "normal_task", "controller", split="normal", run_id="seed-1", calls=[{"fail_open": False, "usage": {"total_tokens": 2}}])
    _branch(root, "challenge_task", "baseline", split="challenge", run_id="seed-2", score=None, success=None, passes=None, total=None)
    _branch(root, "challenge_task", "controller", split="challenge", run_id="seed-2", score=None, success=None, passes=None, total=None, calls=[])
    _branch(root, "missing", "baseline")
    _branch(root, "duplicate", "baseline", run_id="x")
    _branch(root, "duplicate", "baseline", run_id="x", score=50, location="_second")
    _branch(root, "duplicate", "controller", run_id="x", calls=[])
    _branch(root, "seed_task", "baseline", run_id=None, seed="42")
    _branch(root, "seed_task", "controller", run_id=None, seed="42", calls=[])
    payload = analyze(root, tmp_path / "analysis")
    assert set(payload["summary"]) == {"overall", "normal", "challenge"}
    assert payload["summary"]["challenge"]["baseline_mean_score"] is None
    assert len(payload["data_quality"]["missing_controller"]) == 1
    assert len(payload["data_quality"]["duplicate_branches"]) == 1
    assert any(row["task_id"] == "seed_task" and row["seed"] == "42" for row in payload["rows"])
    assert any(row["task_id"] == "challenge_task" and row["controller_triggered"] is False for row in payload["rows"])


def test_referenced_baseline_is_paired_read_only(tmp_path: Path) -> None:
    """Prompt iterations reuse an explicit baseline without copying or changing it."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    _branch(old, "task", "baseline", run_id=None, seed="42", success=False)
    source = old / "runs" / "task_baseline_42"
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    _write_json(new / "manifest.json", {"baseline_references": [{"task_id": "task", "run_path": str(source)}]})
    _branch(new, "task", "controller", run_id=None, seed="42", calls=[{"usage": {"total_tokens": 4}}])
    result = analyze(new, tmp_path / "analysis")
    assert result["summary"]["overall"]["task_count"] == 1
    assert result["rows"][0]["outcome_category"] == "fail_to_success"
    assert result["rows"][0]["baseline_run_path"] == str(source.resolve())
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_referenced_baseline_identity_must_match(tmp_path: Path) -> None:
    """A reference cannot silently substitute another task or a Controller run."""
    old, new = tmp_path / "old", tmp_path / "new"
    _branch(old, "wrong_task", "baseline")
    _write_json(new / "manifest.json", {"baseline_references": [{
        "task_id": "task", "run_path": str(old / "runs" / "wrong_task_baseline_r1"),
    }]})
    with pytest.raises(ValueError, match="identity mismatch"):
        analyze(new, tmp_path / "analysis")


def test_duplicate_baseline_references_are_not_counted(tmp_path: Path) -> None:
    """Duplicate references are surfaced by the existing pairing quality checks."""
    old, new = tmp_path / "old", tmp_path / "new"
    _branch(old, "task", "baseline")
    ref = {"task_id": "task", "run_path": str(old / "runs" / "task_baseline_r1")}
    _write_json(new / "manifest.json", {"baseline_references": [ref, ref]})
    _branch(new, "task", "controller", calls=[])
    result = analyze(new, tmp_path / "analysis")
    assert result["summary"]["overall"]["task_count"] == 0
    assert len(result["data_quality"]["duplicate_branches"]) == 1
