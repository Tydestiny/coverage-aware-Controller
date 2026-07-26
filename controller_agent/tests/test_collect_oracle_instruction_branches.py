"""Offline tests for oracle-instruction batch branch collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import collect_oracle_instruction_branches as collector


def _annotation() -> dict[str, str]:
    """Return one valid oracle annotation record."""
    return {
        "task_id": "task_1",
        "checkpoint_id": "task_1_post_0008",
        "checkpoint_phase": "post_observation",
        "pre_action_record_id": "task_1_pre_0009",
        "trigger_type": "before_finish",
        "oracle_action": "REPLAN",
        "oracle_instruction": "Review the evidence before completing the task.",
    }


def _write_annotation(path: Path) -> None:
    """Write one valid JSONL annotation fixture."""
    path.write_text(json.dumps(_annotation()) + "\n", encoding="utf-8")


def test_read_oracle_annotations_requires_current_schema(tmp_path: Path) -> None:
    """The batch accepts the required oracle schema and rejects other phases."""
    path = tmp_path / "oracle_instruction.jsonl"
    _write_annotation(path)
    assert collector.read_oracle_annotations(path) == [_annotation()]

    invalid = _annotation() | {"checkpoint_phase": "pre_action"}
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    try:
        collector.read_oracle_annotations(path)
    except ValueError as exc:
        assert "unsupported checkpoint_phase" in str(exc)
    else:
        raise AssertionError("Expected unsupported checkpoint phase failure")


def test_build_branch_command_isolates_oracle_and_controller_modes(tmp_path: Path) -> None:
    """Oracle is an external intervention while only controller runs the Controller."""
    args = argparse.Namespace(
        source_root=tmp_path / "source",
        appworld_root=tmp_path / "appworld",
        agent_model="qwen3-coder-flash-2025-07-28",
        max_agent_turns=10_000,
        max_environment_executions=10_000,
        max_controller_calls=10_000,
        max_controller_blocks=10_000,
        max_extra_agent_turns_after_block=10_000,
        max_cost_cny=float("inf"),
    )
    annotation = _annotation()

    baseline = collector.build_branch_command(args, annotation, "baseline", tmp_path / "baseline", "run-baseline")
    controller = collector.build_branch_command(args, annotation, "controller", tmp_path / "controller", "run-controller")
    oracle = collector.build_branch_command(args, annotation, "oracle_instruction", tmp_path / "oracle", "run-oracle")

    assert "--disable-controller" in baseline
    assert "--disable-controller" not in controller
    assert "--disable-controller" in oracle
    assert oracle[oracle.index("--external-action") + 1] == "REPLAN"
    assert oracle[oracle.index("--external-guidance") + 1] == annotation["oracle_instruction"]


def test_dry_run_materializes_prefix_and_skips_without_subprocess(tmp_path: Path, monkeypatch: object) -> None:
    """A non-execute batch writes shared checkpoint metadata but calls no API runner."""
    source_root = tmp_path / "source"
    task_dir = source_root / "task_1"
    (task_dir / "logs").mkdir(parents=True)
    (task_dir / "logs" / "logger.jsonl").write_text('{"step_number": 8}\n', encoding="utf-8")
    checkpoint = {
        "task_id": "task_1",
        "checkpoint_id": "task_1_post_0008",
        "agent_turn_index": 8,
        "appworld_state_id": "state_8",
    }
    checkpoint_path = task_dir / "checkpoints" / "post_observation" / "task_1_post_0008.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    state_dir = task_dir / "checkpoints" / "state_8"
    state_dir.mkdir()
    (state_dir / "spotify.jsonl").write_text("{}\n", encoding="utf-8")
    input_path = tmp_path / "oracle_instruction.jsonl"
    _write_annotation(input_path)
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path / "repo")
    args = argparse.Namespace(
        input=input_path,
        experiment_id="batch_dry_run",
        branches="baseline,controller,oracle_instruction",
        resume=False,
        execute=False,
        source_root=source_root,
        appworld_root=tmp_path / "appworld",
        agent_model="qwen3-coder-flash-2025-07-28",
        max_agent_turns=10_000,
        max_environment_executions=10_000,
        max_controller_calls=10_000,
        max_controller_blocks=10_000,
        max_extra_agent_turns_after_block=10_000,
        max_cost_cny=float("inf"),
    )

    summary = collector.execute_batch(args)

    root = tmp_path / "repo" / "results" / "branching_experiment" / "batch_dry_run"
    assert summary == {"success": 0, "failed": 0, "skipped": 3, "failures": []}
    assert (root / "tasks" / "task_1" / "checkpoints" / "task_1_post_0008" / "checkpoint_manifest.json").is_file()
    assert not list(root.rglob("continuation.jsonl"))


def test_oracle_instruction_is_injected_once_after_restore() -> None:
    """External oracle guidance is visible to the Agent but is not a Controller call."""
    from scripts.run_live_controller_checkpoint_smoke import _inject_external_guidance

    class Agent:
        messages: list[dict[str, str]] = []

    agent = Agent()
    _inject_external_guidance(agent, "REPLAN", "Review the available evidence.")
    _inject_external_guidance(agent, "REPLAN", "Review the available evidence.")

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "user"
    assert "[ORACLE_INSTRUCTION]" in agent.messages[0]["content"]
    assert "Action: REPLAN" in agent.messages[0]["content"]
