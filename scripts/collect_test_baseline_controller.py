"""Collect matched baseline and Controller trajectories from AppWorld test splits."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_ROOT = REPO_ROOT.parent / "appworld"
SEED = 20260722
MAX_CONTROLLER_INTERVENTIONS = 4


def parse_args() -> argparse.Namespace:
    """Parse deterministic test-collection options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="test_gpt4o_gpt56luna_v1")
    parser.add_argument("--normal-count", type=int, default=15)
    parser.add_argument("--challenge-count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--agent-model", default="openrouter/gpt-4o")
    parser.add_argument("--controller-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--controller-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--controller-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--controller-system-prompt", default=None)
    parser.add_argument("--task-selection", default=None)
    parser.add_argument("--rerun-controller", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def select_tasks(dataset_root: Path, normal_count: int, challenge_count: int, seed: int) -> list[dict[str, str]]:
    """Select fixed, non-overlapping task IDs from both official test files."""
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for dataset_name, split, count in (("test_normal", "normal", normal_count), ("test_challenge", "challenge", challenge_count)):
        task_ids = [line.strip() for line in (dataset_root / f"{dataset_name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        if count > len(task_ids):
            raise ValueError(f"Requested {count} tasks from {dataset_name}, which has only {len(task_ids)}.")
        selected.extend({"task_id": task_id, "dataset_name": dataset_name, "split": split} for task_id in rng.sample(task_ids, count))
    return selected


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_or_select_tasks(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    """Reuse the fixed task list when resuming an existing experiment."""
    supplied_selection = getattr(args, "task_selection", None)
    if supplied_selection:
        payload = json.loads(Path(supplied_selection).read_text(encoding="utf-8"))
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise ValueError(f"Invalid supplied task selection: {supplied_selection}")
        return [dict(task) for task in tasks]
    selection_path = root / "task_selection.json"
    if getattr(args, "resume", False) and selection_path.is_file():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise ValueError(f"Invalid persisted task selection: {selection_path}")
        return [dict(task) for task in tasks]
    return select_tasks(APPWORLD_ROOT / "data" / "datasets", args.normal_count, args.challenge_count, args.seed)


def _is_completed_branch(output: Path) -> bool:
    """Return whether a branch has a completed durable result."""
    result_path = output / "result.json"
    summary_path = output / "run_summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(summary, dict) and summary.get("status") == "completed"


def _archive_incomplete_branch(root: Path, output: Path) -> Path:
    """Preserve partial output before rerunning a branch from the initial state."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "interrupted_attempts" / f"{output.name}__{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output), str(destination))
    return destination


def _archive_invalid_controller_branch(root: Path, output: Path) -> Path:
    """Preserve an invalid Controller attempt outside the active result tree."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "invalid_prompt_path_attempts" / f"{output.name}__{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output), str(destination))
    return destination


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize the task list and run baseline/controller pairs sequentially."""
    root = REPO_ROOT / "results" / "end_to_end" / args.experiment_id
    resume = bool(getattr(args, "resume", False))
    if root.exists() and not resume:
        raise FileExistsError(f"Experiment already exists: {root}")
    tasks = _load_or_select_tasks(root, args)
    prompt_path = getattr(args, "controller_system_prompt", None)
    resolved_prompt_path = str(Path(prompt_path).resolve()) if prompt_path else None
    if not resume:
        _write_json(root / "manifest.json", {"experiment_id": args.experiment_id, "seed": args.seed, "agent_model": args.agent_model, "controller_model": args.controller_model, "controller_base_url": args.controller_base_url, "controller_system_prompt": resolved_prompt_path, "enabled_triggers": ["before_finish", "before_write", "no_progress"], "max_controller_calls": MAX_CONTROLLER_INTERVENTIONS, "max_controller_blocks": MAX_CONTROLLER_INTERVENTIONS, "tasks": len(tasks)})
        _write_json(root / "task_selection.json", {"seed": args.seed, "tasks": tasks})
    summary: dict[str, Any] = {"completed": 0, "failed": 0, "skipped": 0, "resumed_completed": 0, "restarted_incomplete": [], "failures": []}
    for task in tasks:
        for branch in ("baseline", "controller"):
            if getattr(args, "rerun_controller", False) and branch == "baseline":
                continue
            output = root / "tasks" / task["task_id"] / f"{task['task_id']}__{branch}"
            force_controller = bool(getattr(args, "rerun_controller", False) and branch == "controller")
            if resume and _is_completed_branch(output) and not force_controller:
                summary["resumed_completed"] += 1
                continue
            if resume and output.exists():
                archive = _archive_invalid_controller_branch(root, output) if force_controller else _archive_incomplete_branch(root, output)
                summary["restarted_incomplete"].append(str(archive))
            command = [
                "uv", "run", "python", str(REPO_ROOT / "scripts" / "run_live_controller_initial_task.py"),
                "--task-id", task["task_id"], "--dataset-name", task["dataset_name"], "--split", task["split"],
                "--branch", branch, "--agent-model", args.agent_model, "--controller-model", args.controller_model,
                "--controller-base-url", args.controller_base_url, "--controller-api-key-env", args.controller_api_key_env,
                "--seed", str(args.seed), "--experiment-name", f"{args.experiment_id}__{task['split']}__{task['task_id']}__{branch}",
                "--output-dir", str(output),
                "--max-controller-calls", str(MAX_CONTROLLER_INTERVENTIONS),
                "--max-controller-blocks", str(MAX_CONTROLLER_INTERVENTIONS),
            ]
            if resolved_prompt_path:
                command.extend(["--controller-system-prompt", resolved_prompt_path])
            if branch == "baseline":
                command.append("--disable-controller")
            if not args.execute:
                summary["skipped"] += 1
                continue
            completed = subprocess.run(command, cwd=APPWORLD_ROOT, check=False)
            if completed.returncode == 0:
                summary["completed"] += 1
            else:
                summary["failed"] += 1
                summary["failures"].append({"task_id": task["task_id"], "branch": branch, "returncode": completed.returncode})
    _write_json(root / "batch_summary.json", summary)
    return summary


def main() -> int:
    """Run the batch collector."""
    args = parse_args()
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
