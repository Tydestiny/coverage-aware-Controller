"""Tests for the end-to-end baseline/controller batch command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import collect_test_baseline_controller as collector


def test_batch_passes_four_intervention_limits_to_each_runner(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Every child run receives the fixed per-task intervention cap."""
    dataset_root = tmp_path / "appworld" / "data" / "datasets"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test_normal.txt").write_text("normal_1\n", encoding="utf-8")
    (dataset_root / "test_challenge.txt").write_text("challenge_1\n", encoding="utf-8")
    monkeypatch.setattr(collector, "APPWORLD_ROOT", tmp_path / "appworld")
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path / "repo")

    commands: list[list[str]] = []

    class Completed:
        returncode = 0

    def fake_run(command: list[str], **_: Any) -> Completed:
        commands.append(command)
        return Completed()

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    summary = collector.run(
        argparse.Namespace(
            experiment_id="test_batch",
            normal_count=1,
            challenge_count=1,
            seed=20260722,
            agent_model="agent",
            controller_model="controller",
            controller_base_url="https://example.test/v1",
            controller_api_key_env="TEST_API_KEY",
            execute=True,
        )
    )

    assert summary["completed"] == 4
    assert len(commands) == 4
    for command in commands:
        assert command[command.index("--max-controller-calls") + 1] == "4"
        assert command[command.index("--max-controller-blocks") + 1] == "4"


def test_resume_reuses_selection_skips_completed_and_archives_partial(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Resume never resamples tasks or appends new logs to a partial branch."""
    root = tmp_path / "repo" / "results" / "end_to_end" / "resumed"
    root.mkdir(parents=True)
    selection = {"seed": 1, "tasks": [{"task_id": "task_1", "dataset_name": "test_normal", "split": "normal"}]}
    (root / "task_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    completed = root / "tasks" / "task_1" / "task_1__baseline"
    completed.mkdir(parents=True)
    (completed / "result.json").write_text("{}", encoding="utf-8")
    (completed / "run_summary.json").write_text('{"status":"completed"}', encoding="utf-8")
    partial = root / "tasks" / "task_1" / "task_1__controller"
    partial.mkdir(parents=True)
    (partial / "continuation.jsonl").write_text("partial\n", encoding="utf-8")
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(collector, "APPWORLD_ROOT", tmp_path / "appworld")

    commands: list[list[str]] = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(collector.subprocess, "run", lambda command, **_: commands.append(command) or Completed())
    summary = collector.run(
        argparse.Namespace(
            experiment_id="resumed", normal_count=99, challenge_count=99, seed=2,
            agent_model="agent", controller_model="controller", controller_base_url="https://example.test/v1",
            controller_api_key_env="TEST_API_KEY", execute=True, resume=True,
        )
    )

    assert summary["resumed_completed"] == 1
    assert len(commands) == 1
    assert not partial.exists()
    archived = [Path(path) for path in summary["restarted_incomplete"]]
    assert len(archived) == 1 and (archived[0] / "continuation.jsonl").is_file()


def test_batch_resolves_prompt_path_for_appworld_child_process(tmp_path: Path, monkeypatch: Any) -> None:
    """The child process receives an absolute prompt path despite its AppWorld cwd."""
    dataset_root = tmp_path / "appworld" / "data" / "datasets"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test_normal.txt").write_text("normal_1\n", encoding="utf-8")
    (dataset_root / "test_challenge.txt").write_text("challenge_1\n", encoding="utf-8")
    prompt = tmp_path / "repo" / "g1_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("G1", encoding="utf-8")
    monkeypatch.setattr(collector, "APPWORLD_ROOT", tmp_path / "appworld")
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path / "repo")
    commands: list[list[str]] = []
    monkeypatch.setattr(collector.subprocess, "run", lambda command, **_: commands.append(command) or type("Done", (), {"returncode": 0})())

    collector.run(argparse.Namespace(experiment_id="prompt_path", normal_count=1, challenge_count=1, seed=1, agent_model="agent", controller_model="controller", controller_base_url="https://example.test/v1", controller_api_key_env="TEST_API_KEY", controller_system_prompt=str(prompt), execute=True, resume=False, task_selection=None, rerun_controller=False))

    for command in commands:
        value = command[command.index("--controller-system-prompt") + 1]
        assert Path(value).is_absolute()
