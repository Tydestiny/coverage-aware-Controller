"""Diagnose AppWorld evaluation behavior before and after ``load_state``.

This is a standalone diagnostic.  It deliberately does not import the
Controller runtime or modify AppWorld/freezegun behavior.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


CASES = (
    "new_evaluate",
    "save_load_evaluate",
    "execute_evaluate",
    "execute_save_load_evaluate",
    "checkpoint_replay_load_evaluate",
)


def versions() -> dict[str, str]:
    """Return exact interpreter, package versions, and import locations."""
    import appworld
    import freezegun

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "appworld_version": importlib.metadata.version("appworld"),
        "appworld_path": str(Path(appworld.__file__).resolve()),
        "freezegun_version": importlib.metadata.version("freezegun"),
        "freezegun_path": str(Path(freezegun.__file__).resolve()),
    }


def run_case(
    case: str,
    task_id: str,
    experiment_name: str,
    source_task_dir: Path | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run one minimal lifecycle case and always attempt ``world.close``."""
    from appworld import AppWorld

    report: dict[str, Any] = {"case": case, "task_id": task_id, "experiment_name": experiment_name, "versions": versions()}
    world: Any | None = None
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=experiment_name,
            random_seed=100,
            raise_on_extra_parameters=True,
        ):
            world = AppWorld(task_id=task_id)
            if case in {"execute_evaluate", "execute_save_load_evaluate"}:
                report["execute_output"] = str(world.execute("print('diagnostic no-op')"))
            if case in {"save_load_evaluate", "execute_save_load_evaluate"}:
                state_id = f"diagnostic-{uuid.uuid4().hex}"
                report["state_id"] = world.save_state(state_id)
                world.load_state(state_id)
            if case == "checkpoint_replay_load_evaluate":
                if source_task_dir is None or checkpoint_id is None:
                    raise ValueError("checkpoint replay case requires --source-task-dir and --checkpoint-id")
                checkpoint_path = source_task_dir / "checkpoints" / "post_observation" / f"{checkpoint_id}.json"
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                state_id = str(checkpoint["appworld_state_id"])
                source_state = source_task_dir / "checkpoints" / state_id
                target_state = Path(world.output_checkpoints_directory) / state_id
                shutil.copytree(source_state, target_state, dirs_exist_ok=True)
                for code in checkpoint.get("executed_code_prefix", []):
                    world.execute(str(code))
                _reset_restore_only_logs(world)
                world.load_state(state_id)
                report["checkpoint_replay"] = {"checkpoint_path": str(checkpoint_path), "state_id": state_id}
            report["evaluation"] = _tracker_to_dict(world.evaluate(suppress_errors=True))
            world.save()
            report["status"] = "success"
    except Exception:
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
    finally:
        if world is not None:
            try:
                world.close()
                report["close_status"] = "success"
            except Exception:
                report["close_status"] = "error"
                report["close_traceback"] = traceback.format_exc()
    return report


def run_offline_evaluation(task_id: str, experiment_name: str) -> dict[str, Any]:
    """Save a task result, close it, then evaluate through the official CLI."""
    preparation = run_case("new_evaluate", task_id, experiment_name)
    cli = Path(sys.executable).with_name("appworld.exe")
    command = [str(cli), "evaluate", experiment_name, "--task-id", task_id]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd(), check=False)
    return {
        "preparation": preparation,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "success": completed.returncode == 0,
    }


def _tracker_to_dict(tracker: Any) -> Any:
    """Serialize an evaluator tracker without assuming a concrete model type."""
    if hasattr(tracker, "to_dict"):
        return tracker.to_dict()
    if hasattr(tracker, "model_dump"):
        return tracker.model_dump()
    return str(tracker)


def _child_command(
    case: str,
    task_id: str,
    experiment_name: str,
    source_task_dir: Path | None,
    checkpoint_id: str | None,
) -> list[str]:
    """Build an isolated child-process command for one lifecycle case."""
    command = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--task-id", task_id, "--experiment-name", experiment_name]
    if source_task_dir is not None:
        command.extend(["--source-task-dir", str(source_task_dir)])
    if checkpoint_id is not None:
        command.extend(["--checkpoint-id", checkpoint_id])
    return command


def run_all(task_id: str, experiment_name: str, source_task_dir: Path, checkpoint_id: str) -> dict[str, Any]:
    """Run all lifecycle cases in separate processes and then offline evaluation."""
    reports: list[dict[str, Any]] = []
    for case in CASES:
        completed = subprocess.run(
            _child_command(case, task_id, experiment_name, source_task_dir, checkpoint_id),
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            check=False,
        )
        reports.append(
            {
                "case": case,
                "process_returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": _extract_result(completed.stdout),
            }
        )
    return {"versions": versions(), "cases": reports, "offline_evaluation": run_offline_evaluation(task_id, experiment_name)}


def _reset_restore_only_logs(world: Any) -> None:
    """Mirror checkpoint namespace replay cleanup without changing runtime state."""
    world.environment_io = []
    world.num_interactions = 0
    world.num_sub_interactions = 0
    world.requester.reset_requests()
    world.save_logs()


def _extract_result(stdout: str) -> dict[str, Any] | None:
    """Extract the final JSON marker emitted by a child process."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("DIAGNOSTIC_RESULT="):
            return json.loads(line.removeprefix("DIAGNOSTIC_RESULT="))
    return None


def main() -> int:
    """Run one case or the isolated full diagnosis suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--task-id", default="287e338_1")
    parser.add_argument("--experiment-name", default="diagnose_evaluate_after_load_state")
    parser.add_argument(
        "--source-task-dir",
        default=str(
            Path("D:/TY/SJTU/CIS/appworld/experiments/outputs/controlled_react_checkpoint_collection/tasks/287e338_1")
        ),
    )
    parser.add_argument("--checkpoint-id", default="287e338_1_post_0008")
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.case) == bool(args.run_all):
        parser.error("Pass exactly one of --case or --run-all.")

    source_task_dir = Path(args.source_task_dir).resolve()
    report = (
        run_all(args.task_id, args.experiment_name, source_task_dir, args.checkpoint_id)
        if args.run_all
        else run_case(args.case, args.task_id, args.experiment_name, source_task_dir, args.checkpoint_id)
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_RESULT=" + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
