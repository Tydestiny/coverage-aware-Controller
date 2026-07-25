"""Tests for checkpoint branch token-usage ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _lm(model: str, prompt: int | None, completion: int | None) -> dict[str, object]:
    usage: dict[str, object] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if prompt is not None and completion is not None:
        usage["total_tokens"] = prompt + completion
    return {"id": f"provider-{prompt}-{completion}", "input": {"model": model}, "output": {"usage": usage}}


def _make_experiment(root: Path) -> tuple[Path, Path, Path]:
    """Create one canonical-layout synthetic experiment with two branches."""
    experiment = root / "exp"
    task = experiment / "tasks" / "task-a"
    checkpoint = task / "checkpoints" / "task-a_post_0002"
    baseline_source = root / "baseline" / "logs"
    _write_jsonl(
        baseline_source / "lm_calls.jsonl",
        [_lm("agent-model", 10, 2), _lm("agent-model", 12, 3), _lm("agent-model", 99, 9)],
    )
    _write_jsonl(
        checkpoint / "prefix_trajectory.jsonl",
        [
            {"role": "agent", "step_number": 1},
            {"role": "environment", "step_number": 1},
            {"role": "agent", "step_number": 2},
        ],
    )
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {"checkpoint_id": "task-a_post_0002", "last_step_number": 2, "prefix_logs_path": str(baseline_source)},
    )

    controller = task / "task-a__task-a_post_0002__controller"
    _write_jsonl(
        controller / "continuation.jsonl",
        [
            {"step_number": 3, "agent_call_id": "agent_call_0003", "generated_code": "print(1)"},
            {"step_number": 4, "agent_call_id": "agent_call_0004", "generated_code": "complete()"},
        ],
    )
    _write_jsonl(controller / "raw_logs" / "lm_calls.jsonl", [_lm("agent-model", 15, 4), _lm("agent-model", 18, 5)])
    _write_jsonl(
        controller / "controller_calls.jsonl",
        [
            {
                "controller_call_id": "controller_call_0001",
                "step_number": 4,
                "model": "controller-model",
                "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
            }
        ],
    )
    _write_json(controller / "result.json", {"task_id": "task-a", "checkpoint_id": "task-a_post_0002", "branch": "controller"})

    baseline = task / "task-a__task-a_post_0002__baseline"
    _write_jsonl(
        baseline / "continuation.jsonl",
        [{"step_number": 3, "agent_call_id": "agent_call_0003", "generated_code": "print(1)"}],
    )
    _write_jsonl(baseline / "raw_logs" / "lm_calls.jsonl", [_lm("agent-model", 20, 4)])
    _write_jsonl(baseline / "controller_calls.jsonl", [])
    _write_json(baseline / "result.json", {"task_id": "task-a", "checkpoint_id": "task-a_post_0002", "branch": "baseline"})
    return experiment, controller, baseline


def test_builds_prefix_and_branch_ledgers_then_updates_totals(tmp_path: Path) -> None:
    """Token totals are derived from per-call ledgers rather than raw summaries."""
    from controller_agent.branching.token_usage import build_experiment_token_usage

    experiment, controller, baseline = _make_experiment(tmp_path)
    report = build_experiment_token_usage(experiment)

    prefix_rows = [json.loads(line) for line in (experiment / "tasks/task-a/checkpoints/task-a_post_0002/prefix_token_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    controller_rows = [json.loads(line) for line in (controller / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    controller_result = json.loads((controller / "result.json").read_text(encoding="utf-8"))
    baseline_result = json.loads((baseline / "result.json").read_text(encoding="utf-8"))

    assert report.errors == []
    assert [row["call_id"] for row in prefix_rows] == ["agent_call_0001", "agent_call_0002"]
    assert [row["step"] for row in controller_rows] == [3, 4, 4]
    assert controller_result["prefix_total_tokens"] == 27
    assert controller_result["continuation_agent_total_tokens"] == 42
    assert controller_result["continuation_controller_total_tokens"] == 9
    assert controller_result["continuation_combined_total_tokens"] == 51
    assert controller_result["effective_total_tokens"] == 78
    assert controller_result["token_usage_status"] == "complete"
    assert baseline_result["continuation_controller_total_tokens"] == 0


def test_missing_raw_usage_stays_null_and_is_not_treated_as_zero(tmp_path: Path) -> None:
    """Unavailable raw Agent usage yields null totals and an unavailable status."""
    from controller_agent.branching.token_usage import build_experiment_token_usage

    experiment, controller, _ = _make_experiment(tmp_path)
    (controller / "raw_logs" / "lm_calls.jsonl").unlink()
    build_experiment_token_usage(experiment)

    rows = [json.loads(line) for line in (controller / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    result = json.loads((controller / "result.json").read_text(encoding="utf-8"))
    assert all(row["available"] is False for row in rows if row["source"] == "agent")
    assert all(row["total_tokens"] is None for row in rows if row["source"] == "agent")
    assert result["continuation_agent_total_tokens"] is None
    assert result["token_usage_status"] == "partial"


def test_rejects_duplicate_call_ids_and_prefix_boundary_violations(tmp_path: Path) -> None:
    """The ledger refuses ambiguous call identity or prefix leakage."""
    from controller_agent.branching.token_usage import build_experiment_token_usage

    experiment, controller, _ = _make_experiment(tmp_path)
    _write_jsonl(
        controller / "continuation.jsonl",
        [
            {"step_number": 2, "agent_call_id": "agent_call_0003"},
            {"step_number": 3, "agent_call_id": "agent_call_0003"},
        ],
    )
    with pytest.raises(ValueError, match="checkpoint|Duplicate"):
        build_experiment_token_usage(experiment)


def test_repeated_build_is_idempotent_and_regenerates_indices(tmp_path: Path) -> None:
    """A repeat run produces byte-identical ledgers and root indices."""
    from controller_agent.branching.token_usage import build_experiment_token_usage

    experiment, controller, _ = _make_experiment(tmp_path)
    build_experiment_token_usage(experiment)
    first = {
        name: (experiment / name).read_bytes()
        for name in ("branch_results.jsonl", "comparison_summary.jsonl")
    } | {"ledger": (controller / "token_usage.jsonl").read_bytes()}
    build_experiment_token_usage(experiment)
    second = {
        name: (experiment / name).read_bytes()
        for name in ("branch_results.jsonl", "comparison_summary.jsonl")
    } | {"ledger": (controller / "token_usage.jsonl").read_bytes()}
    assert second == first
