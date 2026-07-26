"""Offline tests for checkpoint branch prompt-optimizer inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts import optimize_controller_prompt as optimizer


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Write one small JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL fixture rows."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _branch(root: Path, branch: str, success: bool, action: str | None = None) -> None:
    """Create one branch fixture using the runtime logging contract."""
    path = root / "tasks" / "task_1" / f"task_1__task_1_post_0008__{branch}"
    path.mkdir(parents=True)
    _write_json(
        path / "manifest.json",
        {"task_id": "task_1", "checkpoint_id": "task_1_post_0008", "branch": branch},
    )
    _write_json(
        path / "result.json",
        {
            "task_id": "task_1",
            "checkpoint_id": "task_1_post_0008",
            "branch": branch,
            "evaluator_success": success,
            "evaluator_score": 1.0 if success else 0.0,
            "continuation_combined_total_tokens": 20,
        },
    )
    _write_jsonl(
        path / "continuation.jsonl",
        [
            {
                "step_number": 9,
                "agent_call_id": f"agent_call_{branch}",
                "agent_output": "```python\nprint('next')\n```",
                "generated_code": "print('next')",
                "controller_triggered": action is not None,
                "controller_call_id": "controller_call_0001" if action else None,
                "execution_allowed": action is None,
                "environment_observation": "ok" if action is None else None,
                "task_completed": success,
                "termination_reason": "task_completed" if success else "failed",
            }
        ],
    )
    _write_jsonl(
        path / "controller_calls.jsonl",
        []
        if action is None
        else [
            {
                "controller_call_id": "controller_call_0001",
                "step_number": 9,
                "controller_input": {"task": "fixture"},
                "controller_response": {"decision": {"action": action}},
                "action": action,
                "reason": "fixture reason",
                "guidance": "fixture guidance",
                "blocked": True,
                "fail_open": False,
            }
        ],
    )


def test_build_pairs_uses_current_branching_fields(tmp_path: Path) -> None:
    """Current result and runtime identifiers survive contrastive grouping."""
    _branch(tmp_path, "baseline", False)
    _branch(tmp_path, "controller", True, "REPLAN")

    pairs = optimizer.build_pairs(optimizer.discover_branches(tmp_path), 8, 8, 1200, 0)

    assert pairs[0]["contrast_label"] == "controller_helpful"
    controller = pairs[0]["branches"]["controller"]
    assert controller["continuation"][0]["step_number"] == 9
    assert controller["continuation"][0]["agent_call_id"] == "agent_call_controller"
    assert controller["controller_calls"][0]["controller_call_id"] == "controller_call_0001"


def test_build_user_prompt_contains_contrastive_samples(tmp_path: Path) -> None:
    """Prompt input serializes selected branch evidence deterministically."""
    _branch(tmp_path, "baseline", False)
    _branch(tmp_path, "controller", True, "REPLAN")
    pairs = optimizer.build_pairs(optimizer.discover_branches(tmp_path), 8, 8, 1200, 0)

    payload = json.loads(optimizer.build_user_prompt("current prompt", pairs, None))

    assert payload["current_system_prompt"] == "current prompt"
    assert payload["contrastive_branch_samples"][0]["pair_id"] == "task_1::task_1_post_0008"


def test_dry_run_writes_input_without_openai_or_network(tmp_path: Path, monkeypatch: object) -> None:
    """Dry-run writes evidence and never constructs an OpenAI client."""
    _branch(tmp_path / "experiment", "baseline", False)
    _branch(tmp_path / "experiment", "controller", True, "REPLAN")
    prompt = tmp_path / "current_prompt.txt"
    prompt.write_text("current prompt", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_controller_prompt.py",
            "--experiment-root", str(tmp_path / "experiment"),
            "--current-prompt", str(prompt),
            "--output-dir", str(output),
            "--dry-run",
        ],
    )

    assert optimizer.main() == 0
    assert (output / "optimizer_input.json").is_file()
    assert not (output / "optimizer_output.json").exists()
