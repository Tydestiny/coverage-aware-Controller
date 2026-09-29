"""Offline guard tests for the human-facing experiment commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Prepare minimal official task lists without contacting AppWorld."""
    appworld = tmp_path / "appworld"
    datasets = appworld / "data" / "datasets"
    datasets.mkdir(parents=True)
    (appworld / "pyproject.toml").write_text("", encoding="utf-8")
    (datasets / "dev.txt").write_text("dev_task_1\n", encoding="utf-8")
    (datasets / "train.txt").write_text("train_task_1\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("APPWORLD_ROOT", str(appworld))
    monkeypatch.setenv("CONTROLLER_ARTIFACT_ROOT", str(artifacts))
    return appworld, artifacts


def test_dev_baseline_dry_run_never_creates_results(roots: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run prints the command but writes nothing."""
    _, artifacts = roots
    assert experiment.main(["run", "--split", "dev", "--task-id", "dev_task_1", "--experiment-id", "fresh_v1", "--branch", "baseline", "--agent-model", "model"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["mode"] == "dry-run"
    assert planned["command"][planned["command"].index("--trigger-profile") + 1] == "all"
    assert not artifacts.exists()


def test_test_split_is_not_available(roots: tuple[Path, Path]) -> None:
    """The development CLI cannot select AppWorld test tasks."""
    with pytest.raises(SystemExit):
        experiment.main(["run", "--split", "test", "--task-id", "x", "--experiment-id", "fresh_v1", "--branch", "baseline", "--agent-model", "model"])


def test_controller_requires_new_explicit_prompt(roots: tuple[Path, Path]) -> None:
    """No G0/G1/G2 fallback can reach a live model call."""
    with pytest.raises(ValueError, match="require an existing --prompt"):
        experiment.main(["run", "--split", "dev", "--task-id", "dev_task_1", "--experiment-id", "fresh_v1", "--branch", "controller", "--agent-model", "model"])


def test_archived_prompt_is_rejected(roots: tuple[Path, Path], tmp_path: Path) -> None:
    """A historical prompt cannot be accidentally supplied to the new CLI."""
    old_prompt = tmp_path / "archive" / "g2_system.txt"
    old_prompt.parent.mkdir()
    old_prompt.write_text("historical policy", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts/train_dev"):
        experiment.main(["run", "--split", "dev", "--task-id", "dev_task_1", "--experiment-id", "fresh_v1", "--branch", "controller", "--agent-model", "model", "--prompt", str(old_prompt)])


def test_execute_flag_reaches_guarded_runner(roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """A real call can only be dispatched after explicit opt-in."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    assert experiment.main(["run", "--split", "dev", "--task-id", "dev_task_1", "--experiment-id", "fresh_v1", "--branch", "baseline", "--agent-model", "model", "--execute"]) == 0
    assert len(commands) == 1
    assert commands[0][-1] == "--execute"


def test_bad_identifier_cannot_escape_artifact_root(tmp_path: Path) -> None:
    """User-supplied experiment IDs cannot traverse into other directories."""
    with pytest.raises(ValueError):
        experiment.run_path("../archive", "task", "baseline", tmp_path)
    with pytest.raises(ValueError):
        experiment.run_path("..", "task", "baseline", tmp_path)


def test_official_task_ids_reject_test_and_preserve_dev(roots: tuple[Path, Path]) -> None:
    """Direct and wrapped runs share one official train/dev membership rule."""
    from controller_agent.paths import official_task_ids

    appworld, _ = roots
    assert official_task_ids(appworld, "dev") == {"dev_task_1"}
    with pytest.raises(ValueError, match="train/dev"):
        official_task_ids(appworld, "test_normal")


def test_official_evaluator_wrapper_writes_adjacent_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The analysis reader gets the offline evaluator artifact it expects."""
    from scripts import run_official_evaluation

    summary = tmp_path / "run_summary.json"
    summary.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_official_evaluation, "parse_args", lambda: type("Args", (), {
        "experiment_name": "dev_run", "task_id": "dev_task_1", "appworld_root": str(tmp_path),
        "summary": str(summary), "lifecycle_error": None,
    })())
    monkeypatch.setattr(run_official_evaluation, "run_official_evaluation", lambda *args: {"available": True, "task_result": {"success": True}})
    assert run_official_evaluation.main() == 0
    assert json.loads((tmp_path / "evaluator_result.json").read_text(encoding="utf-8"))["task_result"]["success"] is True


def test_checkpoint_branch_dispatch_uses_appworld_python_and_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The preserved checkpoint runner does not fall back to an implicit prompt or wrong Python."""
    import argparse

    from scripts import run_checkpoint_dual_branch_smoke as checkpoint_runner

    captured: list[str] = []
    monkeypatch.setattr(checkpoint_runner, "appworld_python", lambda root: tmp_path / "python.exe")
    monkeypatch.setattr(checkpoint_runner.subprocess, "run", lambda command, **kwargs: captured.extend(command) or object())
    args = argparse.Namespace(
        task_id="task_1", checkpoint_id="task_1_post_0002", max_agent_turns=20,
        max_environment_executions=20, max_controller_calls=2, max_controller_blocks=2,
        max_cost_cny=float("inf"), controller_system_prompt=tmp_path / "prompt.txt",
    )
    checkpoint_runner._run_branch(args, "controller", tmp_path / "branch", "run_1", tmp_path, tmp_path)
    assert captured[0] == str(tmp_path / "python.exe")
    assert "--execute" in captured
    assert "--controller-system-prompt" in captured


def test_checkpoint_pair_dry_run_keeps_sources_untouched(
    roots: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public CLI validates checkpoint identity and plans both branches."""
    appworld, artifacts = roots
    source = appworld / "experiments" / "outputs" / "dev_base__dev_task_1__baseline" / "tasks" / "dev_task_1"
    checkpoint_dir = source / "checkpoints" / "post_observation"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "dev_task_1_post_0002.json"
    checkpoint.write_text(json.dumps({"task_id": "dev_task_1", "checkpoint_id": "dev_task_1_post_0002"}), encoding="utf-8")
    manifest = artifacts / "dev_base" / "tasks" / "dev_task_1" / "baseline" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"split": "dev", "appworld_experiment": "dev_base__dev_task_1__baseline"}), encoding="utf-8")
    monkeypatch.setattr(experiment, "ROOT", tmp_path / "project")
    prompt = experiment.ROOT / "prompts" / "train_dev" / "v1.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("new development policy", encoding="utf-8")
    argv = ["run", "checkpoint", "--split", "dev", "--task-id", "dev_task_1",
            "--experiment-id", "checkpoint_v1", "--agent-model", "model",
            "--checkpoint-id", "dev_task_1_post_0002", "--source-task-dir", str(source),
            "--prompt", str(prompt), "--controller-model", "controller"]
    assert experiment.main(argv) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["mode"] == "dry-run"
    assert "run_checkpoint_dual_branch_smoke.py" in planned["command"][1]
    assert planned["output"] == str(artifacts / "checkpoint_v1")
    assert not (artifacts / "checkpoint_v1").exists()
    checkpoint.write_text(json.dumps({"task_id": "other", "checkpoint_id": "dev_task_1_post_0002"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint identity"):
        experiment.main(argv)
    checkpoint.write_text(json.dumps({"task_id": "dev_task_1", "checkpoint_id": "dev_task_1_post_0002"}), encoding="utf-8")
    manifest.unlink()
    with pytest.raises(ValueError, match="active train/dev baseline manifest"):
        experiment.main(argv)
