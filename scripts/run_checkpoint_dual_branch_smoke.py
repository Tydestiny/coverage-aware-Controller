"""Run one isolated baseline/controller checkpoint smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.token_usage import build_experiment_token_usage, validate_experiment_token_usage
from controller_agent.evaluation.official_evaluator import run_official_evaluation
from controller_agent.logging.logger import summarize_trigger_events
from controller_agent.paths import appworld_python, artifact_root, git_commit, official_task_ids


def parse_args() -> argparse.Namespace:
    """Parse one two-branch checkpoint smoke configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment-id", default="checkpoint_dual_smoke_v1")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split", choices=("train", "dev"))
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--agent-model", default="qwen3-coder-flash-2025-07-28")
    parser.add_argument("--controller-model", default="deepseek-v4-pro")
    parser.add_argument("--controller-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--controller-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--trigger-profile", choices=("before_finish", "all"), default="all")
    parser.add_argument("--controller-system-prompt", type=Path, required=True)
    parser.add_argument(
        "--source-task-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--appworld-root", type=Path, default=REPO_ROOT.parent / "appworld")
    parser.add_argument("--max-agent-turns", type=int, default=10_000)
    parser.add_argument("--max-environment-executions", type=int, default=10_000)
    parser.add_argument("--max-controller-calls", type=int, default=10_000)
    parser.add_argument("--max-controller-blocks", type=int, default=10_000)
    parser.add_argument("--max-cost-cny", type=float, default=float("inf"))
    return parser.parse_args()


def main() -> int:
    """Materialize and execute one real two-branch restored checkpoint experiment."""
    args = parse_args()
    appworld_root = args.appworld_root.resolve()
    source_task_dir = args.source_task_dir.resolve()
    development_ids = official_task_ids(appworld_root, "train") | official_task_ids(appworld_root, "dev")
    if args.task_id not in development_ids:
        raise ValueError(f"{args.task_id} is not in the official train/dev task lists")
    if args.split and args.task_id not in official_task_ids(appworld_root, args.split):
        raise ValueError(f"{args.task_id} is not in the official {args.split} split")
    if not args.controller_system_prompt.resolve().is_relative_to(REPO_ROOT / "prompts" / "train_dev"):
        raise ValueError("Development prompts must be stored under prompts/train_dev")
    root = artifact_root() / args.experiment_id
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {root}")
    if not args.controller_system_prompt.is_file():
        raise FileNotFoundError(args.controller_system_prompt)
    checkpoint = _read_json(source_task_dir / "checkpoints" / "post_observation" / f"{args.checkpoint_id}.json")
    if checkpoint.get("task_id") != args.task_id or checkpoint.get("checkpoint_id") != args.checkpoint_id:
        raise ValueError("Checkpoint identity does not match the requested task and checkpoint")
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "task_id": args.task_id, "checkpoint_id": args.checkpoint_id,
                          "source_task_dir": str(source_task_dir), "output": str(root)}, indent=2))
        return 0
    checkpoint_step = int(checkpoint["agent_turn_index"])
    checkpoint_dir = root / "tasks" / args.task_id / "checkpoints" / args.checkpoint_id
    _materialize_checkpoint(checkpoint_dir, source_task_dir, checkpoint, args.task_id, args.checkpoint_id, checkpoint_step)
    _write_json(
        root / "manifest.json",
        {
            "experiment_id": args.experiment_id,
            "task_id": args.task_id,
            "checkpoint_id": args.checkpoint_id,
            "branches": ["baseline", "controller"],
            "source_task_dir": str(source_task_dir),
            "split": args.split,
            "seed": args.seed,
            "agent_model": args.agent_model,
            "controller_model": args.controller_model,
            "controller_base_url": args.controller_base_url,
            "controller_api_key_env": args.controller_api_key_env,
            "trigger_profile": args.trigger_profile,
            "max_controller_calls": args.max_controller_calls,
            "max_controller_blocks": args.max_controller_blocks,
            "controller_prompt_sha256": hashlib.sha256(args.controller_system_prompt.read_bytes()).hexdigest(),
            "controller_code_commit": git_commit(REPO_ROOT),
            "appworld_code_commit": git_commit(appworld_root),
        },
    )
    _write_json(
        root / "tasks" / args.task_id / "task_manifest.json",
        {"task_id": args.task_id, "original_trajectory_path": str(source_task_dir), "checkpoints": [args.checkpoint_id]},
    )
    branch_results: list[dict[str, Any]] = []
    for branch in ("baseline", "controller"):
        branch_dir = root / "tasks" / args.task_id / f"{args.task_id}__{args.checkpoint_id}__{branch}"
        appworld_experiment = f"{args.experiment_id}__{branch}"
        completed = _run_branch(args, branch, branch_dir, appworld_experiment, appworld_root, source_task_dir)
        official = run_official_evaluation(appworld_root, appworld_experiment, args.task_id)
        evaluator = _normalize_evaluator(args.task_id, official)
        _write_json(branch_dir / "evaluator_result.json", evaluator)
        result = _build_result(args, branch, branch_dir, completed, evaluator)
        _write_json(branch_dir / "result.json", result)
        _write_json(
            branch_dir / "manifest.json",
            {
                "task_id": args.task_id,
                "checkpoint_id": args.checkpoint_id,
                "branch": branch,
                "split": args.split,
                "seed": args.seed,
                "agent_model": args.agent_model,
                "controller_model": args.controller_model,
                "source_checkpoint": str(checkpoint_dir / "state"),
                "appworld_experiment": appworld_experiment,
                "appworld_runtime_state": str(appworld_root / "experiments" / "outputs" / appworld_experiment / "tasks" / args.task_id / "checkpoints"),
                "files": {"continuation.jsonl": True, "controller_calls.jsonl": True, "token_usage.jsonl": False},
            },
        )
        branch_results.append(result)

    build_experiment_token_usage(root)
    validation = validate_experiment_token_usage(root)
    _write_json(root / "token_usage_validation.json", {"checked_branches": validation.checked_branches, "errors": validation.errors})
    return 0 if not validation.errors else 1


def _materialize_checkpoint(
    destination: Path,
    source_task_dir: Path,
    checkpoint: dict[str, Any],
    task_id: str,
    checkpoint_id: str,
    checkpoint_step: int,
) -> None:
    """Create the shared immutable prefix/state directory for both branches."""
    destination.mkdir(parents=True, exist_ok=True)
    logger_rows = _read_jsonl(source_task_dir / "logs" / "logger.jsonl")
    prefix_rows = [row for row in logger_rows if isinstance(row.get("step_number"), int) and row["step_number"] <= checkpoint_step]
    _write_jsonl(destination / "prefix_trajectory.jsonl", prefix_rows)
    raw_prefix = destination / "raw_logs"
    raw_prefix.mkdir(exist_ok=True)
    _write_jsonl(raw_prefix / "logger.jsonl", prefix_rows)
    state_id = str(checkpoint["appworld_state_id"])
    state_source = source_task_dir / "checkpoints" / state_id
    shutil.copytree(state_source, destination / "state", dirs_exist_ok=False)
    _write_json(
        destination / "checkpoint_manifest.json",
        {
            "task_id": task_id,
            "checkpoint_id": checkpoint_id,
            "last_step_number": checkpoint_step,
            "continuation_start_step_number": checkpoint_step + 1,
            "state_path": "state",
            "source_trajectory": str(source_task_dir),
            "prefix_logs_path": str(source_task_dir / "logs"),
        },
    )


def _run_branch(
    args: argparse.Namespace,
    branch: str,
    branch_dir: Path,
    appworld_experiment: str,
    appworld_root: Path,
    source_task_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one branch in its own process and AppWorld output namespace."""
    command = [
        str(appworld_python(appworld_root)),
        str(REPO_ROOT / "scripts" / "run_live_controller_checkpoint_smoke.py"),
        "--execute",
        "--task-id", args.task_id,
        "--checkpoint-id", args.checkpoint_id,
        "--source-task-dir", str(source_task_dir),
        "--appworld-root", str(appworld_root),
        "--output-dir", str(branch_dir),
        "--experiment-name", appworld_experiment,
        "--branch", branch,
        "--max-agent-turns", str(args.max_agent_turns),
        "--max-environment-executions", str(args.max_environment_executions),
        "--max-controller-calls", str(args.max_controller_calls),
        "--max-controller-blocks", str(args.max_controller_blocks),
        "--max-cost-cny", str(args.max_cost_cny),
        "--agent-model", getattr(args, "agent_model", "qwen3-coder-flash-2025-07-28"),
        "--controller-model", getattr(args, "controller_model", "deepseek-v4-pro"),
        "--controller-base-url", getattr(args, "controller_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "--controller-api-key-env", getattr(args, "controller_api_key_env", "DASHSCOPE_API_KEY"),
        "--seed", str(getattr(args, "seed", 20260722)),
        "--trigger-profile", getattr(args, "trigger_profile", "all"),
        "--log-lm-calls",
        "--skip-inprocess-evaluate",
    ]
    if branch == "baseline":
        command.append("--disable-controller")
    else:
        command += ["--controller-system-prompt", str(args.controller_system_prompt.resolve())]
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def _build_result(
    args: argparse.Namespace,
    branch: str,
    branch_dir: Path,
    completed: subprocess.CompletedProcess[str],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    """Create one branch result before token totals are added from its ledger."""
    continuation = _read_jsonl(branch_dir / "continuation.jsonl")
    controller_calls = _read_jsonl(branch_dir / "controller_calls.jsonl")
    trigger_events = _read_jsonl(branch_dir / "trigger_events.jsonl")
    restore = _read_json(branch_dir / "restore_result.json")
    final = continuation[-1] if continuation else {}
    return {
        "task_id": args.task_id,
        "checkpoint_id": args.checkpoint_id,
        "branch": branch,
        "split": getattr(args, "split", None),
        "seed": getattr(args, "seed", None),
        "agent_model": getattr(args, "agent_model", None),
        "controller_model": getattr(args, "controller_model", None),
        "restore_success": bool(restore.get("success")),
        "termination_status": _termination_status(final.get("termination_reason"), completed.returncode),
        "termination_reason": final.get("termination_reason"),
        "steps_after_restore": sum(bool(row.get("execution_allowed")) for row in continuation),
        "agent_call_count": len(continuation),
        "controller_call_count": len(controller_calls),
        "controller_block_count": sum(bool(row.get("blocked")) for row in controller_calls),
        **summarize_trigger_events(trigger_events),
        "evaluator": evaluator,
        "evaluator_success": evaluator.get("success"),
        "evaluator_score": evaluator.get("score"),
        "subprocess_returncode": completed.returncode,
        "subprocess_stdout": completed.stdout[-4000:],
        "subprocess_stderr": completed.stderr[-4000:],
    }


def _normalize_evaluator(task_id: str, official: dict[str, Any]) -> dict[str, Any]:
    """Retain official output while exposing stable top-level evaluation fields."""
    task_result = official.get("task_result") if isinstance(official.get("task_result"), dict) else {}
    score = task_result.get("score") if isinstance(task_result.get("score"), (int, float)) else None
    success = task_result.get("success") if isinstance(task_result.get("success"), bool) else None
    passed = task_result.get("passed") if isinstance(task_result.get("passed"), int) else None
    total = task_result.get("total") if isinstance(task_result.get("total"), int) else None
    return {"task_id": task_id, "success": success, "score": score, "passed": passed, "total": total, "official": official}


def _termination_status(reason: Any, returncode: int) -> str:
    """Classify one completed branch without changing Agent execution semantics."""
    if returncode != 0:
        return "error"
    if reason == "task_completed":
        return "success"
    if reason is None:
        return "error"
    return "limit" if "limit" in str(reason) or "exhausted" in str(reason) else "failure"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one optional JSON object."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON object rows from one JSONL file."""
    if not path.is_file():
        return []
    return [value for line in path.read_text(encoding="utf-8").splitlines() if line.strip() if isinstance((value := json.loads(line)), dict)]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
