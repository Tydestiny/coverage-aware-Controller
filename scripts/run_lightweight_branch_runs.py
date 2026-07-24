"""Run first-round lightweight continuation branches from selected checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.namespace_restore import (  # noqa: E402
    reset_world_interaction_logs,
    restore_python_namespace_from_prefix,
)
from controller_agent.integration.appworld_adapter import GUIDANCE_MARKER  # noqa: E402
from scripts.run_controller_api_smoke import (  # noqa: E402
    AGENT_NAME,
    API_ENDPOINT,
    MODEL_NAME,
    appworld_task_output_dir,
    build_agent_config,
    evaluate_world,
    load_dotenv,
    sanitize_messages,
    write_json,
    write_jsonl,
)
from scripts.run_checkpoint_restore_continue import copy_state_directory  # noqa: E402


GENERIC_REFLECTION = (
    "[CONTROLLER_GUIDANCE]\n\n"
    "Review the task requirements, completed actions, and latest environment feedback.\n"
    "Verify whether the current plan is still valid before continuing."
)
DEFAULT_BRANCHES = ("generic_reflection", "targeted_guidance")
OPTIONAL_BRANCHES = DEFAULT_BRANCHES + ("oracle_instruction",)


def main() -> int:
    """Run all annotated lightweight branches."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    annotations_path = Path(args.annotations).resolve()
    branch_root = resolve_branch_root(args, appworld_root)
    source_root = Path(args.source_root).resolve()
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)
    load_dotenv(appworld_root / "experiments" / ".env")

    annotations = load_annotations(annotations_path)
    annotations = filter_annotations(annotations, args)
    branch_root.mkdir(parents=True, exist_ok=True)
    if args.cleanup_leftovers:
        cleanup_leftover_temporary_runs(branch_root, appworld_root)
        return 0
    write_manifest(branch_root, args, annotations_path, annotations, appworld_root)
    print_run_config(args, annotations, appworld_root, branch_root)
    if args.summarize_only:
        summarize_existing(annotations, branch_root, source_root)
        return 0
    if not args.execute:
        write_json(branch_root / "dry_run_config.json", build_public_config(args, annotations, appworld_root))
        return 0

    if not args.worker:
        return run_workers(args, annotations, appworld_root, branch_root, source_root)

    prepare_appworld_imports(appworld_root)
    for annotation in annotations:
        task_id = str(annotation["task_id"])
        checkpoint_id = str(annotation["checkpoint_id"])
        source_task_dir = source_root / task_id
        branches = [args.only_branch] if args.only_branch else list(DEFAULT_BRANCHES)
        for branch in branches:
            result = run_branch(
                args=args,
                appworld_root=appworld_root,
                branch_root=branch_root,
                source_task_dir=source_task_dir,
                annotation=annotation,
                branch=branch,
            )
            write_json(permanent_run_dir(branch_root, task_id, checkpoint_id, branch) / "result.json", result)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--annotations",
        default=str(REPO_ROOT / "results" / "checkpoint_selection" / "selected_checkpoints.json"),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-id", default="lightweight_v1")
    parser.add_argument(
        "--source-root",
        default=str(
            REPO_ROOT.parent
            / "appworld"
            / "experiments"
            / "outputs"
            / "controlled_react_checkpoint_collection"
            / "tasks"
        ),
    )
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--max-total-agent-turns", type=int, default=20)
    parser.add_argument("--max-total-environment-executions", type=int, default=15)
    parser.add_argument("--max-cost-cny", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--only-task-id", default=None)
    parser.add_argument("--only-checkpoint-id", default=None)
    parser.add_argument("--only-branch", choices=OPTIONAL_BRANCHES, default=None)
    parser.add_argument(
        "--output-branch-name",
        default=None,
        help="Optional safe directory label for one branch rerun; does not change guidance semantics.",
    )
    parser.add_argument("--cleanup-leftovers", action="store_true")
    return parser.parse_args()


def filter_annotations(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Apply optional task/checkpoint filters."""
    filtered = rows
    if args.only_task_id:
        filtered = [row for row in filtered if row.get("task_id") == args.only_task_id]
    if args.only_checkpoint_id:
        filtered = [row for row in filtered if row.get("checkpoint_id") == args.only_checkpoint_id]
    if not filtered:
        raise ValueError("No selected checkpoints matched the requested filters.")
    return filtered


def run_workers(
    args: argparse.Namespace,
    annotations: list[dict[str, Any]],
    appworld_root: Path,
    branch_root: Path,
    source_root: Path,
) -> int:
    """Run each branch in an isolated Python process and then summarize results."""
    failures: list[str] = []
    for annotation in annotations:
        for branch in DEFAULT_BRANCHES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--execute",
                "--worker",
                "--annotations",
                str(Path(args.annotations).resolve()),
                "--output-root",
                str(branch_root),
                "--experiment-id",
                args.experiment_id,
                "--source-root",
                str(source_root),
                "--appworld-root",
                str(appworld_root),
                "--max-total-agent-turns",
                str(args.max_total_agent_turns),
                "--max-total-environment-executions",
                str(args.max_total_environment_executions),
                "--max-cost-cny",
                str(args.max_cost_cny),
                "--seed",
                str(args.seed),
                "--only-task-id",
                str(annotation["task_id"]),
                "--only-checkpoint-id",
                str(annotation["checkpoint_id"]),
                "--only-branch",
                branch,
            ]
            completed = subprocess.run(command, cwd=str(appworld_root), check=False)
            if completed.returncode != 0:
                failures.append(f"{annotation['task_id']} {annotation['checkpoint_id']} {branch}: {completed.returncode}")
    summarize_existing(annotations, branch_root, source_root)
    return 1 if failures else 0


def summarize_existing(
    annotations: list[dict[str, Any]],
    branch_root: Path,
    source_root: Path,
) -> None:
    """Build branch_results.jsonl and comparison_summary.jsonl from result files."""
    branch_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        task_id = str(annotation["task_id"])
        checkpoint_id = str(annotation["checkpoint_id"])
        generic = load_branch_result(branch_root, task_id, checkpoint_id, "generic_reflection")
        targeted = load_branch_result(branch_root, task_id, checkpoint_id, "targeted_guidance")
        branch_results.extend([generic, targeted])
        baseline_result = baseline_status(source_root / task_id)
        comparison_rows.append(
            {
                "task_id": task_id,
                "checkpoint_id": checkpoint_id,
                "baseline_result": baseline_result,
                "generic_reflection_result": generic["termination_status"],
                "targeted_guidance_result": targeted["termination_status"],
                "targeted_action": annotation.get("targeted_action"),
                "targeted_improved_over_baseline": compare_result(targeted["termination_status"], baseline_result),
                "targeted_improved_over_generic": compare_result(
                    targeted["termination_status"],
                    generic["termination_status"],
                ),
            }
        )
    write_jsonl(branch_root / "branch_results.jsonl", branch_results)
    write_jsonl(branch_root / "comparison_summary.jsonl", comparison_rows)


def load_branch_result(branch_root: Path, task_id: str, checkpoint_id: str, branch: str) -> dict[str, Any]:
    """Load a branch result or create a missing-result error row."""
    path = permanent_run_dir(branch_root, task_id, checkpoint_id, branch) / "result.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "branch": branch,
        "controller_action": None,
        "restore_success": False,
        "termination_status": "error",
        "evaluator_result": None,
        "steps_after_restore": 0,
        "controller_calls": 1,
        "experiment_id": None,
        "run_id": None,
        "permanent_run_path": str(path.parent),
        "temporary_run_path": None,
        "archive_status": "missing",
        "cleanup_status": "missing",
        "error": f"Missing result file: {path}",
    }


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load and validate selected checkpoint annotations."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Annotation file must contain a list: {path}")
    required = {
        "task_id",
        "checkpoint_id",
        "checkpoint_phase",
        "case_type",
        "targeted_action",
        "targeted_guidance",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Annotation {index} missing fields: {missing}")
        if row["checkpoint_phase"] != "post_observation":
            raise ValueError(f"Only post_observation checkpoints are supported: {row}")
    return rows


def run_branch(
    *,
    args: argparse.Namespace,
    appworld_root: Path,
    branch_root: Path,
    source_task_dir: Path,
    annotation: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    """Restore one checkpoint, inject one guidance message, and run a branch."""
    task_id = str(annotation["task_id"])
    checkpoint_id = str(annotation["checkpoint_id"])
    output_branch_name = validate_output_branch_name(args.output_branch_name) if args.output_branch_name else branch
    branch_dir = permanent_run_dir(branch_root, task_id, checkpoint_id, output_branch_name)
    prepare_branch_dir(branch_dir)
    result = result_template(task_id, checkpoint_id, branch, annotation)
    unique_run_id = build_unique_run_id(args.experiment_id, task_id, checkpoint_id, branch)
    temporary_run_path = temporary_run_dir(appworld_root, unique_run_id)
    result["experiment_id"] = args.experiment_id
    result["run_id"] = unique_run_id
    result["permanent_run_path"] = str(branch_dir)
    result["storage_branch_name"] = output_branch_name
    result["temporary_run_path"] = str(temporary_run_path)
    try:
        checkpoint = load_checkpoint(source_task_dir, checkpoint_id)
        validate_checkpoint_identity(checkpoint, task_id, checkpoint_id)
        experiment_name = f"_branch_tmp/{unique_run_id}"
        appworld_task_dir = appworld_task_output_dir(appworld_root, experiment_name, task_id)
        run_result = continue_from_checkpoint(
            args=args,
            appworld_root=appworld_root,
            experiment_name=experiment_name,
            appworld_task_dir=appworld_task_dir,
            source_task_dir=source_task_dir,
            checkpoint=checkpoint,
            annotation=annotation,
            branch=branch,
        )
        result.update(run_result)
        archive_status = archive_branch_outputs(appworld_task_dir, branch_dir, result)
        result["archive_status"] = archive_status
        if archive_status == "success":
            result["cleanup_status"] = cleanup_temporary_run(appworld_root, temporary_run_path)
        else:
            result["cleanup_status"] = "failed"
    except Exception as exc:
        result["restore_success"] = False if result["steps_after_restore"] == 0 else result["restore_success"]
        result["termination_status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["archive_status"] = "failed"
        result["cleanup_status"] = "failed"
        write_minimal_error_outputs(branch_dir, result["error"])
    write_json(branch_dir / "result.json", result)
    print(f"{task_id} {checkpoint_id} {branch}: {result['termination_status']}")
    return result


def continue_from_checkpoint(
    *,
    args: argparse.Namespace,
    appworld_root: Path,
    experiment_name: str,
    appworld_task_dir: Path,
    source_task_dir: Path,
    checkpoint: dict[str, Any],
    annotation: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    """Run the AppWorld continuation after checkpoint restore."""
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent, ExecutionIO

    task_id = str(checkpoint["task_id"])
    checkpoint_turn = int(checkpoint.get("agent_turn_index") or 0)
    checkpoint_executions = int(checkpoint.get("environment_execution_index") or 0)
    remaining_turns = max(1, args.max_total_agent_turns - checkpoint_turn)
    remaining_executions = max(0, args.max_total_environment_executions - checkpoint_executions)
    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=remaining_turns,
        max_extra_agent_turns_after_block=0,
    )
    agent = Agent.from_dict(build_agent_config(agent_args, appworld_root))
    guidance = branch_guidance(branch, annotation)
    events: list[dict[str, Any]] = []
    evaluator_result: dict[str, Any] = {}
    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=experiment_name,
            random_seed=args.seed,
            raise_on_extra_parameters=True,
        ):
            world = AppWorld(task_id=task_id)
            try:
                agent.initialize(world)
                copy_state_directory(source_task_dir, appworld_task_dir, checkpoint)
                namespace_restore = restore_python_namespace_from_prefix(world, checkpoint)
                if not namespace_restore.success:
                    raise RuntimeError(f"Python namespace replay failed: {namespace_restore.error}")
                world.load_state(str(checkpoint["appworld_state_id"]))
                reset_world_interaction_logs(world)
                agent.messages = list(checkpoint["agent_messages"])
                inject_guidance_once(agent, guidance)
                agent.step_number = checkpoint_turn
                execution_outputs: Sequence[Any] = [
                    ExecutionIO(content=str(output), metadata={})
                    for output in checkpoint.get("last_execution_outputs", [])
                ]
                restore_success = True
                stop_reason = "unknown"
                for _ in range(remaining_turns):
                    turn_start = time.perf_counter()
                    agent.step_number += 1
                    call_result = agent.next_execution_inputs_usage_and_status(execution_outputs)
                    execution_inputs, usage, status = call_result
                    record_usage(agent, task_id, usage)
                    generated_code = "\n".join(str(item.content) for item in execution_inputs)
                    if getattr(status, "failed", False):
                        stop_reason = "error"
                        events.append(build_event(agent, world, generated_code, False, False, usage, turn_start, stop_reason))
                        break
                    if not generated_code.strip():
                        stop_reason = "error"
                        events.append(build_event(agent, world, generated_code, False, False, usage, turn_start, "model_format_failure"))
                        break
                    if count_environment_executions(events) >= remaining_executions:
                        stop_reason = "limit"
                        events.append(build_event(agent, world, generated_code, True, False, usage, turn_start, "environment_limit"))
                        break
                    outputs = world.batch_execute([item.content for item in execution_inputs])
                    execution_outputs = [
                        type(item)(content=output, metadata=item.metadata)
                        for item, output in zip(execution_inputs, outputs, strict=True)
                    ]
                    events.append(build_event(agent, world, generated_code, True, True, usage, turn_start, None))
                    events[-1]["execution_outputs"] = [output.content for output in execution_outputs]
                    if world.task_completed():
                        stop_reason = "task_completed"
                        events[-1]["termination_reason"] = stop_reason
                        break
                    if usage_exceeded(agent, task_id):
                        stop_reason = "limit"
                        events[-1]["termination_reason"] = "cost_limit"
                        break
                if stop_reason == "unknown":
                    stop_reason = "limit"
                    if events:
                        events[-1]["termination_reason"] = "agent_turn_limit"
                evaluator_result = evaluate_world(world)
            finally:
                try:
                    world.close()
                except Exception as exc:
                    print(f"Non-fatal AppWorld close error for {task_id}: {type(exc).__name__}: {exc}")
    finally:
        os.chdir(previous_cwd)
    success = task_success(evaluator_result)
    return {
        "restore_success": restore_success,
        "termination_status": termination_status(stop_reason, success),
        "evaluator_result": sanitize_messages(evaluator_result),
        "evaluator_score": evaluator_score(evaluator_result),
        "steps_after_restore": count_environment_executions(events),
        "continuation_events": sanitize_messages(events),
        "namespace_restore": namespace_restore.to_dict(),
        "namespace_restore_mode": namespace_restore.mode,
        "namespace_replay_success": namespace_restore.success,
        "namespace_replay_steps": namespace_restore.steps,
        "namespace_replay_error": namespace_restore.error,
        "error": None,
    }


def prepare_appworld_imports(appworld_root: Path) -> None:
    """Expose AppWorld and experiment agent modules on sys.path."""
    if str(appworld_root) not in sys.path:
        sys.path.insert(0, str(appworld_root))
    appworld_src_root = appworld_root / "src"
    if str(appworld_src_root) not in sys.path:
        sys.path.insert(0, str(appworld_src_root))
    appworld_code_root = appworld_root / "experiments" / "code"
    if str(appworld_code_root) not in sys.path:
        sys.path.insert(0, str(appworld_code_root))
    ensure_appworld_agents_package(appworld_root)
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401
    import appworld  # noqa: F401


def ensure_appworld_agents_package(appworld_root: Path) -> None:
    """Expose AppWorld's experiments package under the appworld_agents name."""
    if "appworld_agents" in sys.modules:
        return
    package_root = appworld_root / "experiments"
    init_path = package_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "appworld_agents",
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["appworld_agents"] = module
    spec.loader.exec_module(module)


def load_checkpoint(source_task_dir: Path, checkpoint_id: str) -> dict[str, Any]:
    """Load a checkpoint record."""
    checkpoint_path = source_task_dir / "checkpoints" / "post_observation" / f"{checkpoint_id}.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint record: {checkpoint_path}")
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def validate_checkpoint_identity(checkpoint: dict[str, Any], task_id: str, checkpoint_id: str) -> None:
    """Check checkpoint identity before restore."""
    if checkpoint.get("task_id") != task_id:
        raise ValueError(f"Checkpoint task_id mismatch: {checkpoint.get('task_id')} != {task_id}")
    if checkpoint.get("checkpoint_id") != checkpoint_id:
        raise ValueError(f"Checkpoint id mismatch: {checkpoint.get('checkpoint_id')} != {checkpoint_id}")
    if not checkpoint.get("appworld_state_id"):
        raise ValueError("Checkpoint is missing appworld_state_id")
    if not isinstance(checkpoint.get("agent_messages"), list):
        raise ValueError("Checkpoint is missing agent_messages")


def branch_guidance(branch: str, annotation: dict[str, Any]) -> str:
    """Return the one guidance message for a branch."""
    if branch == "generic_reflection":
        return GENERIC_REFLECTION
    if branch == "oracle_instruction":
        return annotation["oracle_instruction"]
    return f"{GUIDANCE_MARKER}\n\n{annotation['targeted_guidance']}"


def inject_guidance_once(agent: Any, guidance: str) -> None:
    """Append one marked guidance message to visible model history."""
    messages = getattr(agent, "messages", None)
    if not isinstance(messages, list):
        raise TypeError("Agent messages must be a list.")
    messages.append({"role": "user", "content": guidance.strip() + "\n"})


def record_usage(agent: Any, task_id: str, usage: Any) -> None:
    """Update AppWorld usage tracker after a model call."""
    tracker = getattr(agent, "usage_tracker", None)
    if tracker is not None and hasattr(tracker, "add"):
        tracker.add(task_id, usage)
    if hasattr(agent, "log_usage"):
        agent.log_usage()


def usage_exceeded(agent: Any, task_id: str) -> bool:
    """Return whether the agent usage tracker exceeded its budget."""
    tracker = getattr(agent, "usage_tracker", None)
    if tracker is None or not hasattr(tracker, "exceeded"):
        return False
    return bool(tracker.exceeded(task_id))


def build_event(
    agent: Any,
    world: Any,
    generated_code: str,
    execution_allowed: bool,
    world_batch_execute_called: bool,
    usage: Any,
    turn_start: float,
    termination_reason: str | None,
) -> dict[str, Any]:
    """Build one branch event."""
    return {
        "agent_turn_index": int(getattr(agent, "step_number", 0)),
        "generated_code": generated_code,
        "execution_allowed": execution_allowed,
        "world_batch_execute_called": world_batch_execute_called,
        "task_completed": bool(world.task_completed()),
        "agent_input_tokens": usage_value(usage, "input_tokens"),
        "agent_output_tokens": usage_value(usage, "output_tokens"),
        "latency_seconds": time.perf_counter() - turn_start,
        "termination_reason": termination_reason,
    }


def usage_value(usage: Any, name: str) -> int | None:
    """Best-effort extraction of token usage."""
    tokens = getattr(usage, "tokens", None)
    if tokens is not None and name == "input_tokens":
        return getattr(tokens, "input_cache_hit_or_miss", None)
    if tokens is not None and name == "output_tokens":
        return getattr(tokens, "output", None)
    value = getattr(usage, name, None)
    return int(value) if value is not None else None


def count_environment_executions(events: list[dict[str, Any]]) -> int:
    """Count branch environment executions."""
    return sum(1 for event in events if event.get("world_batch_execute_called"))


def task_success(evaluator_result: dict[str, Any]) -> bool:
    """Best-effort extraction of official evaluator success."""
    result = evaluator_result.get("result")
    if isinstance(result, dict):
        if "success" in result:
            return bool(result["success"])
        if "score" in result:
            try:
                return float(result["score"]) >= 1.0
            except (TypeError, ValueError):
                return False
    return False


def evaluator_score(evaluator_result: dict[str, Any]) -> float | None:
    """Best-effort extraction of evaluator score."""
    result = evaluator_result.get("result")
    if isinstance(result, dict):
        if "score" in result:
            try:
                return float(result["score"])
            except (TypeError, ValueError):
                return None
        passed = result.get("num_passed")
        total = result.get("num_tests")
        if isinstance(passed, int) and isinstance(total, int) and total:
            return passed / total
    return None


def termination_status(stop_reason: str, success: bool) -> str:
    """Map stop reason to requested result label."""
    if success:
        return "success"
    if stop_reason in {"limit", "agent_turn_limit", "environment_limit", "cost_limit"}:
        return "limit"
    if stop_reason in {"task_completed", "error"}:
        return "failure"
    return "limit"


def baseline_status(source_task_dir: Path) -> str:
    """Infer baseline success/failure from the official evaluation report."""
    report = source_task_dir / "evaluation" / "report.md"
    if not report.exists():
        return "unknown"
    text = report.read_text(encoding="utf-8", errors="replace")
    failed = re.search(r"Num Failed Tests\s*:\s*(\d+)", text)
    passed = re.search(r"Num Passed Tests\s*:\s*(\d+)", text)
    if failed and int(failed.group(1)) == 0:
        return "success"
    if failed and passed:
        return "failure"
    return "unknown"


def compare_result(candidate: str, reference: str) -> bool | None:
    """Compare result categories only when there is a clear success/failure ordering."""
    ordered = {"success": 1, "failure": 0}
    cand = ordered.get(candidate)
    ref = ordered.get(reference)
    if cand is None or ref is None:
        return None
    return cand > ref


def permanent_run_dir(branch_root: Path, task_id: str, checkpoint_id: str, branch: str) -> Path:
    """Return the permanent task/checkpoint/branch run directory."""
    return branch_root / task_id / checkpoint_id / branch


def validate_output_branch_name(value: str) -> str:
    """Validate a single safe directory label for a non-destructive rerun."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("output branch name must contain only letters, digits, '.', '_' or '-'.")
    return value


def resolve_branch_root(args: argparse.Namespace, appworld_root: Path) -> Path:
    """Return the long-term branch result root."""
    if args.output_root:
        return Path(args.output_root).resolve()
    return appworld_root / "experiments" / "outputs" / "lightweight_branch_runs"


def temporary_root(appworld_root: Path) -> Path:
    """Return the AppWorld temporary branch root."""
    return appworld_root / "experiments" / "outputs" / "_branch_tmp"


def temporary_run_dir(appworld_root: Path, run_id: str) -> Path:
    """Return one exact temporary run directory."""
    return temporary_root(appworld_root) / run_id


def build_unique_run_id(experiment_id: str, task_id: str, checkpoint_id: str, branch: str) -> str:
    """Build a collision-resistant run id with useful provenance."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = f"{experiment_id}__{task_id}__{checkpoint_id}__{branch}__{timestamp}__{uuid.uuid4().hex[:8]}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def prepare_branch_dir(branch_dir: Path) -> None:
    """Create branch directory and remove stale branch-local artifacts only."""
    branch_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "continuation.jsonl",
        "environment_io.md",
        "lm_calls.jsonl",
        "api_calls.jsonl",
        "logger.log",
        "logger.jsonl",
        "result.json",
    ]:
        path = branch_dir / name
        if path.exists():
            path.unlink()


def archive_branch_outputs(appworld_task_dir: Path, branch_dir: Path, result: dict[str, Any]) -> str:
    """Persist branch artifacts in the simplified branch directory."""
    try:
        branch_dir.mkdir(parents=True, exist_ok=True)
        events = result.pop("continuation_events", [])
        write_jsonl(branch_dir / "continuation.jsonl", events)
        write_complete_logger_jsonl(branch_dir / "logger.jsonl", events)
        for source_name, target_name in [
            ("environment_io.md", "environment_io.md"),
            ("lm_calls.jsonl", "lm_calls.jsonl"),
            ("api_calls.jsonl", "api_calls.jsonl"),
            ("logger.log", "logger.log"),
        ]:
            source = appworld_task_dir / "logs" / source_name
            target = branch_dir / target_name
            if source.exists():
                shutil.copy2(source, target)
            else:
                target.write_text("", encoding="utf-8")
        return "success"
    except Exception as exc:
        result["error"] = f"archive_failed: {type(exc).__name__}: {exc}"
        return "failed"


def write_complete_logger_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    """Write a complete continuation logger JSONL from branch events."""
    rows: list[dict[str, Any]] = []
    for event in events:
        step_number = event.get("agent_turn_index")
        generated_code = event.get("generated_code")
        if generated_code is not None:
            rows.append(
                {
                    "role": "agent",
                    "content": str(generated_code),
                    "raw_message": {},
                    "step_number": step_number,
                }
            )
        for output in event.get("execution_outputs") or []:
            rows.append(
                {
                    "role": "environment",
                    "content": str(output),
                    "raw_message": {},
                    "step_number": step_number,
                }
            )
    write_jsonl(path, rows)


def cleanup_temporary_run(appworld_root: Path, run_dir: Path) -> str:
    """Delete one exact script-created temporary run directory."""
    tmp_root = temporary_root(appworld_root).resolve()
    resolved = run_dir.resolve()
    try:
        if resolved.parent != tmp_root:
            return "failed"
        if not resolved.exists():
            return "success"
        for _ in range(3):
            try:
                shutil.rmtree(resolved, onerror=make_writable_and_retry)
                return "success"
            except Exception:
                time.sleep(0.25)
        return "success" if not resolved.exists() else "failed"
    except Exception:
        return "failed"


def make_writable_and_retry(function: Any, path: str, _exc_info: Any) -> None:
    """Make a Windows path writable and retry the failed remove operation."""
    os.chmod(path, 0o700)
    function(path)


def cleanup_leftover_temporary_runs(branch_root: Path, appworld_root: Path) -> None:
    """Clean exact leftover temp dirs only when their permanent archive exists."""
    results_path = branch_root / "branch_results.jsonl"
    if not results_path.exists():
        return
    rows: list[dict[str, Any]] = []
    changed = False
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        if row.get("archive_status") != "success":
            continue
        run_id = row.get("run_id")
        permanent_path = row.get("permanent_run_path")
        if not run_id or not permanent_path:
            continue
        result_path = Path(permanent_path) / "result.json"
        if not result_path.exists():
            continue
        status = cleanup_temporary_run(appworld_root, temporary_run_dir(appworld_root, str(run_id)))
        if status == "success" and row.get("cleanup_status") != "success":
            row["cleanup_status"] = "success"
            changed = True
    if changed:
        write_jsonl(results_path, rows)


def write_minimal_error_outputs(branch_dir: Path, error: str) -> None:
    """Ensure required permanent files exist on early errors."""
    (branch_dir / "continuation.jsonl").write_text("", encoding="utf-8")
    (branch_dir / "environment_io.md").write_text("", encoding="utf-8")
    (branch_dir / "lm_calls.jsonl").write_text("", encoding="utf-8")
    (branch_dir / "api_calls.jsonl").write_text("", encoding="utf-8")
    (branch_dir / "logger.log").write_text(error + "\n", encoding="utf-8")


def write_manifest(
    branch_root: Path,
    args: argparse.Namespace,
    annotations_path: Path,
    annotations: list[dict[str, Any]],
    appworld_root: Path,
) -> None:
    """Write the experiment-level manifest."""
    manifest = {
        "experiment_id": args.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_file": str(annotations_path),
        "run_count": len(annotations) * 2,
        "branches": ["generic_reflection", "targeted_guidance"],
        "temporary_root": str(temporary_root(appworld_root)),
        "storage_policy": {
            "save_continuation_only": True,
            "compress_raw_logs": True,
            "cleanup_temporary_runs": True,
        },
    }
    write_json(branch_root / "manifest.json", manifest)


def result_template(task_id: str, checkpoint_id: str, branch: str, annotation: dict[str, Any]) -> dict[str, Any]:
    """Return the requested result schema with defaults."""
    controller_action = None
    if branch == "oracle_instruction":
        controller_action = "ORACLE_INSTRUCTION"
    elif branch != "generic_reflection":
        controller_action = annotation.get("targeted_action")
    return {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "branch": branch,
        "controller_action": controller_action,
        "restore_success": True,
        "termination_status": "error",
        "evaluator_result": None,
        "steps_after_restore": 0,
        "controller_calls": 1,
        "error": None,
        "experiment_id": None,
        "run_id": None,
        "permanent_run_path": None,
        "temporary_run_path": None,
        "archive_status": "pending",
        "cleanup_status": "pending",
    }


def print_run_config(
    args: argparse.Namespace,
    annotations: list[dict[str, Any]],
    appworld_root: Path,
    output_root: Path,
) -> None:
    """Print safe experiment configuration."""
    print("Lightweight branch run config:")
    print(f"experiment_id: {args.experiment_id}")
    print(f"annotations: {args.annotations}")
    print(f"selected checkpoints: {len(annotations)}")
    print(f"branches per checkpoint: 2")
    print(f"model: {MODEL_NAME}")
    print(f"agent: {AGENT_NAME}")
    print(f"temperature: 0.0")
    print(f"seed: {args.seed}")
    print(f"max_total_agent_turns: {args.max_total_agent_turns}")
    print(f"max_total_environment_executions: {args.max_total_environment_executions}")
    print(f"max_cost_cny_per_branch: {args.max_cost_cny}")
    print(f"endpoint: {API_ENDPOINT}")
    print(f"appworld_root: {appworld_root}")
    print(f"permanent_root: {output_root}")
    print(f"temporary_root: {temporary_root(appworld_root)}")


def build_public_config(
    args: argparse.Namespace,
    annotations: list[dict[str, Any]],
    appworld_root: Path,
) -> dict[str, Any]:
    """Return JSON-safe dry-run config without secrets."""
    return {
        "annotations": args.annotations,
        "experiment_id": args.experiment_id,
        "num_checkpoints": len(annotations),
        "branches": ["generic_reflection", "targeted_guidance"],
        "model": MODEL_NAME,
        "agent": AGENT_NAME,
        "temperature": 0.0,
        "seed": args.seed,
        "max_total_agent_turns": args.max_total_agent_turns,
        "max_total_environment_executions": args.max_total_environment_executions,
        "max_cost_cny_per_branch": args.max_cost_cny,
        "endpoint": API_ENDPOINT,
        "appworld_root": str(appworld_root),
        "temporary_root": str(temporary_root(appworld_root)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
