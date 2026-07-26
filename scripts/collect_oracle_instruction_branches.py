"""Collect baseline, Controller, and oracle-instruction checkpoint branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.token_usage import build_experiment_token_usage, validate_experiment_token_usage
from controller_agent.evaluation.official_evaluator import run_official_evaluation
from scripts.run_checkpoint_dual_branch_smoke import (
    _build_result,
    _materialize_checkpoint,
    _normalize_evaluator,
    _read_json,
    _write_json,
)


DEFAULT_BRANCHES = ("baseline", "controller", "oracle_instruction")
SUPPORTED_PHASE = "post_observation"


def parse_args() -> argparse.Namespace:
    """Parse batch branch collection arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--branches", default=",".join(DEFAULT_BRANCHES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT.parent
        / "appworld"
        / "experiments"
        / "outputs"
        / "controlled_react_checkpoint_collection"
        / "tasks",
    )
    parser.add_argument("--appworld-root", type=Path, default=REPO_ROOT.parent / "appworld")
    parser.add_argument("--agent-model", default="qwen3-coder-flash-2025-07-28")
    parser.add_argument("--max-agent-turns", type=int, default=10_000)
    parser.add_argument("--max-environment-executions", type=int, default=10_000)
    parser.add_argument("--max-controller-calls", type=int, default=10_000)
    parser.add_argument("--max-controller-blocks", type=int, default=10_000)
    parser.add_argument("--max-extra-agent-turns-after-block", type=int, default=10_000)
    parser.add_argument("--max-cost-cny", type=float, default=float("inf"))
    return parser.parse_args()


def read_oracle_annotations(path: Path) -> list[dict[str, Any]]:
    """Read and validate JSONL annotations for post-observation checkpoints."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input JSONL: {path}")
    records: list[dict[str, Any]] = []
    required = {
        "task_id",
        "checkpoint_id",
        "checkpoint_phase",
        "pre_action_record_id",
        "trigger_type",
        "oracle_action",
        "oracle_instruction",
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Input record at {path}:{line_number} must be an object.")
        missing = sorted(field for field in required if field not in record)
        if missing:
            raise ValueError(f"Input record at {path}:{line_number} is missing: {', '.join(missing)}")
        if record["checkpoint_phase"] != SUPPORTED_PHASE:
            raise ValueError(
                f"Input record at {path}:{line_number} has unsupported checkpoint_phase "
                f"{record['checkpoint_phase']!r}; only {SUPPORTED_PHASE!r} is supported."
            )
        if not all(isinstance(record[field], str) and record[field].strip() for field in required):
            raise ValueError(f"Input record at {path}:{line_number} contains an empty required string.")
        records.append(dict(record))
    if not records:
        raise ValueError(f"Input JSONL contains no records: {path}")
    identities = [(record["task_id"], record["checkpoint_id"]) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Input JSONL contains duplicate task_id/checkpoint_id records.")
    return records


def parse_branches(value: str) -> tuple[str, ...]:
    """Parse an ordered, deduplicated branch selection."""
    branches = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = sorted(set(branches) - set(DEFAULT_BRANCHES))
    if invalid or not branches:
        raise ValueError(f"Unsupported branches: {', '.join(invalid) or value}")
    return branches


def branch_dir(root: Path, task_id: str, checkpoint_id: str, branch: str) -> Path:
    """Return the standard branch directory under one experiment."""
    return root / "tasks" / task_id / f"{task_id}__{checkpoint_id}__{branch}"


def is_successfully_completed(path: Path) -> bool:
    """Return whether a branch finished with an official evaluator outcome."""
    result = _read_json(path / "result.json")
    return (
        bool(result.get("restore_success"))
        and result.get("termination_status") not in {"error", "limit"}
        and isinstance(result.get("evaluator_success"), bool)
    )


def build_branch_command(
    args: argparse.Namespace,
    annotation: dict[str, Any],
    branch: str,
    destination: Path,
    appworld_experiment: str,
) -> list[str]:
    """Build one isolated invocation of the existing checkpoint branch runner."""
    source_task_dir = Path(args.source_root).resolve() / annotation["task_id"]
    command = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "run_live_controller_checkpoint_smoke.py"),
        "--execute",
        "--task-id",
        annotation["task_id"],
        "--agent-model",
        args.agent_model,
        "--checkpoint-id",
        annotation["checkpoint_id"],
        "--source-task-dir",
        str(source_task_dir),
        "--appworld-root",
        str(Path(args.appworld_root).resolve()),
        "--output-dir",
        str(destination),
        "--experiment-name",
        appworld_experiment,
        "--branch",
        "controller" if branch == "oracle_instruction" else branch,
        "--max-agent-turns",
        str(args.max_agent_turns),
        "--max-environment-executions",
        str(args.max_environment_executions),
        "--max-controller-calls",
        str(args.max_controller_calls),
        "--max-controller-blocks",
        str(args.max_controller_blocks),
        "--max-extra-agent-turns-after-block",
        str(args.max_extra_agent_turns_after_block),
        "--max-cost-cny",
        str(args.max_cost_cny),
        "--log-lm-calls",
        "--skip-inprocess-evaluate",
    ]
    if branch in {"baseline", "oracle_instruction"}:
        command.append("--disable-controller")
    if branch == "oracle_instruction":
        command.extend(
            [
                "--external-action",
                annotation["oracle_action"],
                "--external-guidance",
                annotation["oracle_instruction"],
            ]
        )
    return command


def execute_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize checkpoints and run selected branches independently."""
    annotations = read_oracle_annotations(Path(args.input).resolve())
    branches = parse_branches(args.branches)
    root = REPO_ROOT / "results" / "branching_experiment" / args.experiment_id
    if root.exists() and not args.resume:
        raise FileExistsError(f"Experiment already exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "manifest.json",
        {
            "experiment_id": args.experiment_id,
            "input": str(Path(args.input).resolve()),
            "branches": list(branches),
            "checkpoint_phase": SUPPORTED_PHASE,
            "source_root": str(Path(args.source_root).resolve()),
            "agent_model": args.agent_model,
        },
    )

    summary: dict[str, Any] = {"success": 0, "failed": 0, "skipped": 0, "failures": []}
    for annotation in annotations:
        task_id = annotation["task_id"]
        checkpoint_id = annotation["checkpoint_id"]
        source_task_dir = Path(args.source_root).resolve() / task_id
        checkpoint = _read_json(source_task_dir / "checkpoints" / SUPPORTED_PHASE / f"{checkpoint_id}.json")
        try:
            checkpoint_step = int(checkpoint["agent_turn_index"])
            checkpoint_root = root / "tasks" / task_id / "checkpoints" / checkpoint_id
            if not checkpoint_root.exists():
                _materialize_checkpoint(
                    checkpoint_root,
                    source_task_dir,
                    checkpoint,
                    task_id,
                    checkpoint_id,
                    checkpoint_step,
                )
            _upsert_task_manifest(root / "tasks" / task_id / "task_manifest.json", source_task_dir, annotation)
        except Exception as exc:
            for branch in branches:
                _record_failure(root, annotation, branch, f"checkpoint_setup_failed: {type(exc).__name__}: {exc}")
                summary["failed"] += 1
                summary["failures"].append(_failure_row(annotation, branch, str(exc)))
            continue

        for branch in branches:
            destination = branch_dir(root, task_id, checkpoint_id, branch)
            if args.resume and is_successfully_completed(destination):
                summary["skipped"] += 1
                continue
            if not args.execute:
                summary["skipped"] += 1
                continue
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            appworld_experiment = f"{args.experiment_id}__{task_id}__{checkpoint_id}__{branch}"
            try:
                completed = subprocess.run(
                    build_branch_command(args, annotation, branch, destination, appworld_experiment),
                    cwd=Path(args.appworld_root).resolve(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                completed = subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr=f"{type(exc).__name__}: {exc}",
                )
            try:
                official = run_official_evaluation(Path(args.appworld_root).resolve(), appworld_experiment, task_id)
                evaluator = _normalize_evaluator(task_id, official)
            except Exception as exc:
                evaluator = {"task_id": task_id, "success": None, "score": None, "error": f"{type(exc).__name__}: {exc}"}
            _write_json(destination / "evaluator_result.json", evaluator)
            result = _build_result(
                SimpleNamespace(task_id=task_id, checkpoint_id=checkpoint_id),
                branch,
                destination,
                completed,
                evaluator,
            )
            if branch == "oracle_instruction":
                result["oracle_action"] = annotation["oracle_action"]
                result["oracle_instruction_injected"] = True
            if completed.returncode != 0:
                result["error"] = _failure_reason(result)
            _write_json(destination / "result.json", result)
            _write_json(
                destination / "manifest.json",
                {
                    "task_id": task_id,
                    "checkpoint_id": checkpoint_id,
                    "branch": branch,
                    "source_checkpoint": str(root / "tasks" / task_id / "checkpoints" / checkpoint_id / "state"),
                    "appworld_experiment": appworld_experiment,
                    "external_intervention": (
                        {"action": annotation["oracle_action"], "instruction": annotation["oracle_instruction"]}
                        if branch == "oracle_instruction"
                        else None
                    ),
                    "files": {"continuation.jsonl": True, "controller_calls.jsonl": True, "token_usage.jsonl": False},
                },
            )
            if is_successfully_completed(destination):
                summary["success"] += 1
            else:
                summary["failed"] += 1
                summary["failures"].append(_failure_row(annotation, branch, _failure_reason(result)))

    if args.execute:
        try:
            build_experiment_token_usage(root)
            validation = validate_experiment_token_usage(root)
            _write_json(root / "token_usage_validation.json", {"checked_branches": validation.checked_branches, "errors": validation.errors})
        except Exception as exc:
            summary["failures"].append({"branch": "token_usage", "reason": f"{type(exc).__name__}: {exc}"})
    _write_json(root / "batch_summary.json", summary)
    return summary


def _record_failure(root: Path, annotation: dict[str, Any], branch: str, error: str) -> None:
    """Write a branch-local failure result without stopping the batch."""
    destination = branch_dir(root, annotation["task_id"], annotation["checkpoint_id"], branch)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination / "result.json",
        {
            "task_id": annotation["task_id"],
            "checkpoint_id": annotation["checkpoint_id"],
            "branch": branch,
            "restore_success": False,
            "termination_status": "error",
            "termination_reason": "checkpoint_setup_failed",
            "evaluator_success": None,
            "evaluator_score": None,
            "error": error,
        },
    )


def _upsert_task_manifest(path: Path, source_task_dir: Path, annotation: dict[str, Any]) -> None:
    """Add one checkpoint annotation without discarding prior task checkpoints."""
    existing = _read_json(path)
    checkpoints = existing.get("checkpoints") if isinstance(existing.get("checkpoints"), list) else []
    checkpoint_id = annotation["checkpoint_id"]
    if checkpoint_id not in checkpoints:
        checkpoints.append(checkpoint_id)
    annotations = existing.get("annotations") if isinstance(existing.get("annotations"), list) else []
    if annotation not in annotations:
        annotations.append(annotation)
    _write_json(
        path,
        {
            "task_id": annotation["task_id"],
            "original_trajectory_path": str(source_task_dir),
            "checkpoints": checkpoints,
            "annotations": annotations,
        },
    )


def _failure_row(annotation: dict[str, Any], branch: str, reason: str) -> dict[str, str]:
    """Return one printable failed-branch summary row."""
    return {"task_id": annotation["task_id"], "checkpoint_id": annotation["checkpoint_id"], "branch": branch, "reason": reason}


def _failure_reason(result: dict[str, Any]) -> str:
    """Extract a concise persisted branch failure reason."""
    if result.get("subprocess_returncode") not in (None, 0):
        return f"branch_runner_returncode={result['subprocess_returncode']}"
    evaluator = result.get("evaluator")
    if isinstance(evaluator, dict) and evaluator.get("error"):
        return str(evaluator["error"])
    return str(result.get("termination_reason") or "official_evaluator_unavailable")


def main() -> int:
    """Run the requested batch, or print a dry-run plan without API calls."""
    args = parse_args()
    summary = execute_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
