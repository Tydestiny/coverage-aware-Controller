"""Copy legacy branch artifacts into the fixed branching experiment layout.

The migration is intentionally copy-only.  It never removes or rewrites the
source experiment directories, and it does not derive new token statistics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_ROOT = REPO_ROOT.parent / "appworld"
LEGACY_BRANCH_ROOT = APPWORLD_ROOT / "experiments" / "outputs" / "lightweight_branch_runs"
BASELINE_TASK_ROOT = (
    APPWORLD_ROOT / "experiments" / "outputs" / "controlled_react_checkpoint_collection" / "tasks"
)

RAW_LOG_NAMES = (
    "logger.log",
    "logger.jsonl",
    "lm_calls.jsonl",
    "api_calls.jsonl",
    "environment_io.md",
    "usage.json",
)


@dataclass(frozen=True)
class BranchSource:
    """One legacy branch directory and its identity."""

    task_id: str
    checkpoint_id: str
    branch: str
    storage_branch: str
    path: Path
    result: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse migration arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment-id", default="legacy_branching_v1")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_BRANCH_ROOT)
    parser.add_argument("--baseline-task-root", type=Path, default=BASELINE_TASK_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "branching_experiment",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load valid JSON object rows from a JSONL file."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def branch_sources(root: Path) -> list[BranchSource]:
    """Find legacy branch result directories without trusting old indices."""
    sources: list[BranchSource] = []
    for result_path in sorted(root.glob("*/*/*/result.json")):
        result = read_json(result_path)
        task_id = str(result.get("task_id") or result_path.parents[2].name)
        checkpoint_id = str(result.get("checkpoint_id") or result_path.parents[1].name)
        branch = str(result.get("branch") or result_path.parent.name)
        sources.append(BranchSource(task_id, checkpoint_id, branch, result_path.parent.name, result_path.parent, result))
    return sources


def copy_file(source: Path, destination: Path) -> bool:
    """Copy a file if present and return whether a copy occurred."""
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def extract_task_text(checkpoint: dict[str, Any]) -> str | None:
    """Extract the final literal Task line from the checkpoint message history."""
    messages = checkpoint.get("agent_messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        task_lines = [line.removeprefix("Task:").strip() for line in content.splitlines() if line.startswith("Task:")]
        if task_lines:
            return task_lines[-1]
    return None


def checkpoint_path(task_root: Path, checkpoint_id: str) -> Path | None:
    """Locate a post-observation checkpoint JSON by its stable ID."""
    candidate = task_root / "checkpoints" / "post_observation" / f"{checkpoint_id}.json"
    return candidate if candidate.exists() else None


def prefix_logger_rows(task_root: Path, checkpoint_turn: int) -> list[dict[str, Any]]:
    """Return source logger rows that cannot be after the checkpoint turn."""
    rows = load_jsonl(task_root / "logs" / "logger.jsonl")
    prefix: list[dict[str, Any]] = []
    for row in rows:
        step = row.get("step_number")
        if isinstance(step, int) and step <= checkpoint_turn:
            prefix.append(row)
    return prefix


def prefix_usage_rows(task_root: Path, checkpoint_turn: int) -> list[dict[str, Any]]:
    """Return existing collection ledger rows up to the checkpoint turn."""
    rows = load_jsonl(task_root / "collection" / "turn_events.jsonl")
    return [row for row in rows if isinstance(row.get("agent_turn_index"), int) and row["agent_turn_index"] <= checkpoint_turn]


def migrate_checkpoint(
    destination: Path,
    task_root: Path,
    source: BranchSource,
) -> tuple[dict[str, Any], str | None]:
    """Materialize the shared checkpoint prefix once for one task/checkpoint pair."""
    destination.mkdir(parents=True, exist_ok=True)
    source_checkpoint_path = checkpoint_path(task_root, source.checkpoint_id)
    if source_checkpoint_path is None:
        manifest = {
            "checkpoint_id": source.checkpoint_id,
            "availability": "missing_source_checkpoint",
            "source_task_path": str(task_root),
            "state_path": None,
            "prefix_logs_path": None,
        }
        write_json(destination / "checkpoint_manifest.json", manifest)
        return manifest, None

    checkpoint = read_json(source_checkpoint_path)
    checkpoint_turn = checkpoint.get("agent_turn_index")
    checkpoint_execution = checkpoint.get("environment_execution_index")
    if not isinstance(checkpoint_turn, int) or not isinstance(checkpoint_execution, int):
        raise ValueError(f"Invalid checkpoint counters: {source_checkpoint_path}")

    prefix_rows = prefix_logger_rows(task_root, checkpoint_turn)
    usage_rows = prefix_usage_rows(task_root, checkpoint_turn)
    write_jsonl(destination / "prefix_trajectory.jsonl", prefix_rows)
    write_jsonl(destination / "prefix_token_usage.jsonl", usage_rows)

    state_id = checkpoint.get("appworld_state_id")
    state_source = task_root / "checkpoints" / str(state_id) if isinstance(state_id, str) else None
    state_destination = destination / "state"
    state_available = bool(state_source and state_source.is_dir())
    if state_available:
        shutil.copytree(state_source, state_destination, dirs_exist_ok=True)
    else:
        state_destination.mkdir(parents=True, exist_ok=True)

    raw_logs = destination / "raw_logs"
    raw_logs.mkdir(parents=True, exist_ok=True)
    # logger.jsonl is filtered above; copying the other baseline raw logs would
    # leak post-checkpoint records, so their original paths remain in metadata.
    write_jsonl(raw_logs / "logger.jsonl", prefix_rows)
    raw_log_availability = {
        "logger.jsonl": "materialized_filtered_prefix",
        "logger.log": "not_materialized_to_avoid_post_checkpoint_content",
        "lm_calls.jsonl": "not_materialized_to_avoid_post_checkpoint_content",
        "api_calls.jsonl": "not_materialized_to_avoid_post_checkpoint_content",
        "environment_io.md": "not_materialized_to_avoid_post_checkpoint_content",
        "usage.json": "not_materialized_to_avoid_post_checkpoint_content",
    }
    manifest = {
        "checkpoint_id": source.checkpoint_id,
        "task_id": source.task_id,
        "agent_turn_index": checkpoint_turn,
        "environment_execution_index": checkpoint_execution,
        "state_path": "state" if state_available else None,
        "source_checkpoint_path": str(source_checkpoint_path),
        "prefix_logs_path": str(task_root / "logs"),
        "prefix_last_event": prefix_rows[-1] if prefix_rows else None,
        "prefix_token_usage_available": bool(usage_rows),
        "raw_log_availability": raw_log_availability,
    }
    write_json(destination / "checkpoint_manifest.json", manifest)
    return manifest, extract_task_text(checkpoint)


def migrate_branch(destination: Path, source: BranchSource, experiment_id: str) -> dict[str, Any]:
    """Copy one continuation and its branch-local raw logs."""
    destination.mkdir(parents=True, exist_ok=True)
    raw_logs = destination / "raw_logs"
    raw_logs.mkdir(parents=True, exist_ok=True)
    availability: dict[str, bool] = {}
    for name in RAW_LOG_NAMES:
        availability[name] = copy_file(source.path / name, raw_logs / name)
    continuation_available = copy_file(source.path / "continuation.jsonl", destination / "continuation.jsonl")
    if not continuation_available:
        # A stable empty location records that this legacy run never captured a
        # structured continuation; its manifest distinguishes that from zero
        # generated events.
        (destination / "continuation.jsonl").write_text("", encoding="utf-8")

    # Legacy guidance branches did not emit these two normalized ledgers.  Empty
    # JSONL files preserve a stable location without inventing records.
    (destination / "controller_calls.jsonl").write_text("", encoding="utf-8")
    (destination / "token_usage.jsonl").write_text("", encoding="utf-8")

    result = dict(source.result)
    result["experiment_id"] = experiment_id
    result["token_usage_available"] = False
    result["continuation_tokens"] = None
    result["effective_total_tokens"] = None
    write_json(destination / "result.json", result)

    evaluator = result.get("evaluator_result")
    evaluator_data = evaluator if isinstance(evaluator, dict) else {
        "available": False,
        "reason": "not present in legacy result.json",
    }
    write_json(destination / "evaluator_result.json", evaluator_data)

    manifest = {
        "task_id": source.task_id,
        "checkpoint_id": source.checkpoint_id,
        "branch": source.branch,
        "storage_branch": source.storage_branch,
        "source_checkpoint_path": None,
        "source_branch_path": str(source.path),
        "agent_model": None,
        "controller_model": None,
        "prompt_hash": None,
        "files": {
            "continuation.jsonl": continuation_available,
            "controller_calls.jsonl": False,
            "token_usage.jsonl": False,
            "result.json": True,
            "evaluator_result.json": isinstance(evaluator, dict),
            "raw_logs": availability,
        },
    }
    write_json(destination / "manifest.json", manifest)
    return result


def migrate_legacy(args: argparse.Namespace) -> Path:
    """Create one canonical, copy-only migration experiment."""
    sources = branch_sources(args.legacy_root)
    if not sources:
        raise FileNotFoundError(f"No legacy branch results found under {args.legacy_root}")
    destination = args.output_root / args.experiment_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")

    branch_rows: list[dict[str, Any]] = []
    task_data: dict[str, dict[str, Any]] = {}
    migrated_checkpoints: set[tuple[str, str]] = set()
    for source in sources:
        task_root = args.baseline_task_root / source.task_id
        checkpoint_destination = destination / "tasks" / source.task_id / "checkpoints" / source.checkpoint_id
        checkpoint_key = (source.task_id, source.checkpoint_id)
        if checkpoint_key not in migrated_checkpoints:
            checkpoint_manifest, task_text = migrate_checkpoint(checkpoint_destination, task_root, source)
            task_data.setdefault(source.task_id, {"task_text": task_text, "checkpoints": []})
            task_data[source.task_id]["checkpoints"].append(checkpoint_manifest)
            migrated_checkpoints.add(checkpoint_key)
        branch_destination = (
            destination
            / "tasks"
            / source.task_id
            / f"{source.task_id}__{source.checkpoint_id}__{source.storage_branch}"
        )
        result = migrate_branch(branch_destination, source, args.experiment_id)
        branch_rows.append(
            {
                "experiment_id": args.experiment_id,
                "task_id": source.task_id,
                "checkpoint_id": source.checkpoint_id,
                "branch": source.branch,
                "storage_branch": source.storage_branch,
                "run_id": result.get("run_id"),
                "result_path": str(branch_destination / "result.json"),
                "termination_status": result.get("termination_status"),
                "restore_success": result.get("restore_success"),
                "token_usage_available": False,
            }
        )

    for task_id, data in task_data.items():
        task_manifest = {
            "task_id": task_id,
            "task_text": data["task_text"],
            "dataset_split": None,
            "original_trajectory_path": str(args.baseline_task_root / task_id),
            "checkpoints": data["checkpoints"],
        }
        write_json(destination / "tasks" / task_id / "task_manifest.json", task_manifest)

    write_jsonl(destination / "branch_results.jsonl", branch_rows)
    source_comparisons = load_jsonl(args.legacy_root / "comparison_summary.jsonl")
    write_jsonl(destination / "comparison_summary.jsonl", source_comparisons)
    root_manifest = {
        "experiment_id": args.experiment_id,
        "migration_kind": "copy_only_legacy_layout_migration",
        "source_branch_root": str(args.legacy_root),
        "source_checkpoint_root": str(args.baseline_task_root),
        "branch_count": len(branch_rows),
        "task_count": len(task_data),
        "comparison_rows_preserved": len(source_comparisons),
        "agent_model": None,
        "controller_model": None,
        "prompt_version": None,
        "prompt_hash": None,
        "trigger_config": None,
        "git_commit": None,
        "appworld_code_version": None,
        "appworld_data_version": None,
    }
    write_json(destination / "manifest.json", root_manifest)
    return destination


def main() -> int:
    """Run the copy-only migration when explicitly requested."""
    args = parse_args()
    destination = args.output_root / args.experiment_id
    if not args.execute:
        print(destination)
        return 0
    migrated = migrate_legacy(args)
    print(migrated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
