"""Restore one AppWorld runtime checkpoint and continue with the real agent API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def main() -> int:
    """Run a checkpoint restore continuation test."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)
    load_dotenv(appworld_root / "experiments" / ".env")

    source_task_dir = Path(args.source_task_dir).resolve()
    checkpoint = load_checkpoint(source_task_dir, args.checkpoint_id)
    task_id = str(checkpoint["task_id"])
    checkpoint_id = str(checkpoint["checkpoint_id"])
    output_task_dir = appworld_task_output_dir(appworld_root, args.experiment_name, task_id)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else output_task_dir / "restore_continue" / checkpoint_id
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)
    copy_state_directory(source_task_dir, output_task_dir, checkpoint)

    config = {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "source_task_dir": str(source_task_dir),
        "source_state_id": checkpoint.get("appworld_state_id"),
        "source_agent_turn_index": checkpoint.get("agent_turn_index"),
        "source_environment_execution_index": checkpoint.get("environment_execution_index"),
        "namespace_restore_mode": checkpoint.get("namespace_restore_mode"),
        "agent_model": MODEL_NAME,
        "agent_name": AGENT_NAME,
        "api_endpoint": API_ENDPOINT,
        "experiment_name": args.experiment_name,
        "max_agent_turns_after_restore": args.max_agent_turns_after_restore,
        "max_environment_executions_after_restore": args.max_environment_executions_after_restore,
        "max_cost_cny": args.max_cost_cny,
        "temperature": 0.0,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "config.json", config)
    print_run_config(config)
    if not args.execute:
        write_report(output_dir, config, [], {"dry_run": True}, "dry_run", False, False)
        return 0

    if str(appworld_root) not in sys.path:
        sys.path.insert(0, str(appworld_root))
    appworld_code_root = appworld_root / "experiments" / "code"
    if str(appworld_code_root) not in sys.path:
        sys.path.insert(0, str(appworld_code_root))
    ensure_appworld_agents_package(appworld_root)
    try:
        import appworld_agents.code.simplified.react_code_agent  # noqa: F401
        from appworld import AppWorld
        from appworld_agents.code.simplified.agent import Agent, ExecutionIO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Run this script with the AppWorld uv environment.") from exc

    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    evaluator_result: dict[str, Any] = {}
    restore_success = False
    continuation_started = False
    stop_reason = "unknown"
    close_error: str | None = None
    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    try:
        agent_args = argparse.Namespace(
            max_cost_cny=args.max_cost_cny,
            max_agent_turns=args.max_agent_turns_after_restore,
            max_extra_agent_turns_after_block=0,
        )
        agent = Agent.from_dict(build_agent_config(agent_args, appworld_root))
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=args.experiment_name,
            random_seed=100,
            raise_on_extra_parameters=True,
        ):
            world = AppWorld(task_id=task_id)
            try:
                agent.initialize(world)
                copy_state_directory(source_task_dir, output_task_dir, checkpoint)
                world.load_state(str(checkpoint["appworld_state_id"]))
                restore_success = True
                agent.messages = checkpoint["agent_messages"]
                agent.step_number = int(checkpoint["agent_turn_index"])
                execution_outputs: Sequence[Any] = [
                    ExecutionIO(content=str(output), metadata={})
                    for output in checkpoint.get("last_execution_outputs", [])
                ]
                for _ in range(args.max_agent_turns_after_restore):
                    turn_start = time.perf_counter()
                    agent.step_number += 1
                    result = agent.next_execution_inputs_usage_and_status(execution_outputs)
                    execution_inputs, usage, status = result
                    record_usage(agent, task_id, usage)
                    generated_code = "\n".join(str(item.content) for item in execution_inputs)
                    if getattr(status, "failed", False):
                        stop_reason = "agent_api_failure"
                        events.append(build_event(agent, world, result, generated_code, False, False, turn_start, stop_reason))
                        break
                    if not generated_code.strip():
                        stop_reason = "model_format_failure"
                        events.append(build_event(agent, world, result, generated_code, False, False, turn_start, stop_reason))
                        break
                    if count_executions(events) >= args.max_environment_executions_after_restore:
                        stop_reason = "max_environment_executions_reached"
                        events.append(build_event(agent, world, result, generated_code, True, False, turn_start, stop_reason))
                        break
                    outputs = world.batch_execute([item.content for item in execution_inputs])
                    continuation_started = True
                    execution_outputs = [
                        type(item)(content=output, metadata=item.metadata)
                        for item, output in zip(execution_inputs, outputs, strict=True)
                    ]
                    task_completed = bool(world.task_completed())
                    events.append(build_event(agent, world, result, generated_code, True, True, turn_start, None))
                    events[-1]["execution_outputs"] = [output.content for output in execution_outputs]
                    if task_completed:
                        stop_reason = "task_completed"
                        events[-1]["termination_reason"] = stop_reason
                        break
                    if usage_exceeded(agent, task_id):
                        stop_reason = "cost_limit_reached"
                        events[-1]["termination_reason"] = stop_reason
                        break
                if stop_reason == "unknown":
                    stop_reason = "max_agent_turns_reached"
                    if events:
                        events[-1]["termination_reason"] = stop_reason
                evaluator_result = evaluate_world(world)
                messages = sanitize_messages(getattr(agent, "messages", []))
            finally:
                try:
                    world.close()
                except Exception as exc:  # AppWorld load_state can leave freezegun in a double-close state.
                    close_error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous_cwd)
    wall_time = time.perf_counter() - started
    for event in events:
        event["total_wall_time_seconds"] = wall_time
    category = classify_restore_result(restore_success, continuation_started, evaluator_result, stop_reason)
    write_jsonl(output_dir / "turn_events.jsonl", events)
    write_jsonl(output_dir / "sanitized_messages.jsonl", messages if "messages" in locals() else [])
    write_json(output_dir / "evaluator_result.json", evaluator_result)
    report_config = config | {"close_error": close_error}
    write_json(output_dir / "config.json", report_config)
    write_report(output_dir, report_config, events, evaluator_result, category, restore_success, continuation_started)
    return 0 if category in {"restore_continue_success", "restore_continue_completed_with_task_failure"} else 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--source-task-dir",
        default=str(
            REPO_ROOT.parent
            / "appworld"
            / "experiments"
            / "outputs"
            / "controlled_react_checkpoint_collection"
            / "tasks"
            / "23cf851_1"
        ),
    )
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--experiment-name", default="checkpoint_restore_continue")
    parser.add_argument("--max-agent-turns-after-restore", type=int, default=8)
    parser.add_argument("--max-environment-executions-after-restore", type=int, default=8)
    parser.add_argument("--max-cost-cny", type=float, default=1.0)
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_checkpoint(source_task_dir: Path, checkpoint_id: str | None) -> dict[str, Any]:
    """Load a named or middle post-observation checkpoint."""
    manifest_path = source_task_dir / "checkpoints" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    posts = manifest.get("post_observation") or []
    if not posts:
        raise ValueError(f"No post-observation checkpoints in {manifest_path}")
    selected = next((item for item in posts if item.get("checkpoint_id") == checkpoint_id), None)
    if selected is None:
        selected = posts[max(0, len(posts) // 2 - 1)]
    selected_id = str(selected["checkpoint_id"])
    checkpoint_path = source_task_dir / "checkpoints" / "post_observation" / f"{selected_id}.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint record: {checkpoint_path}")
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def ensure_appworld_agents_package(appworld_root: Path) -> None:
    """Expose AppWorld's experiments package under the appworld_agents name."""
    if "appworld_agents" in sys.modules:
        return
    package_root = appworld_root / "experiments"
    init_path = package_root / "__init__.py"
    if not init_path.exists():
        return
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


def copy_state_directory(source_task_dir: Path, output_task_dir: Path, checkpoint: dict[str, Any]) -> None:
    """Copy the saved AppWorld state into the restore experiment checkpoint directory."""
    state_id = str(checkpoint["appworld_state_id"])
    source = source_task_dir / "checkpoints" / state_id
    target = output_task_dir / "checkpoints" / state_id
    if not source.exists():
        raise FileNotFoundError(f"Missing source state directory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


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


def count_executions(events: list[dict[str, Any]]) -> int:
    """Count actual environment executions in prior continuation events."""
    return sum(1 for event in events if event.get("world_batch_execute_called"))


def build_event(
    agent: Any,
    world: Any,
    result: Any,
    generated_code: str,
    execution_allowed: bool,
    world_batch_execute_called: bool,
    turn_start: float,
    termination_reason: str | None,
) -> dict[str, Any]:
    """Build one continuation turn event."""
    _execution_inputs, usage, _status = result
    return {
        "agent_turn_index": int(getattr(agent, "step_number", 0)),
        "environment_execution_index": None,
        "generated_code": generated_code,
        "execution_allowed": execution_allowed,
        "world_batch_execute_called": world_batch_execute_called,
        "task_completed": bool(world.task_completed()),
        "agent_input_tokens": usage_value(usage, "input_tokens"),
        "agent_output_tokens": usage_value(usage, "output_tokens"),
        "cumulative_cost": total_cost(agent),
        "latency_seconds": time.perf_counter() - turn_start,
        "termination_reason": termination_reason,
    }


def usage_value(usage: Any, name: str) -> Any:
    """Best-effort extraction of a usage field."""
    value = getattr(usage, name, None)
    if value is not None:
        return value
    tokens = getattr(usage, "tokens", None)
    if tokens is not None and name == "input_tokens":
        return (getattr(tokens, "input_cache_miss", 0) or 0) + (getattr(tokens, "input_cache_hit", 0) or 0)
    if tokens is not None and name == "output_tokens":
        return getattr(tokens, "output", None)
    value = getattr(usage, name.replace("_tokens", ""), None)
    return value


def total_cost(agent: Any) -> Any:
    """Best-effort extraction of accumulated cost."""
    tracker = getattr(agent, "usage_tracker", None)
    usage = getattr(tracker, "overall_usage", None)
    cost = getattr(usage, "cost", None)
    return getattr(cost, "total", None)


def classify_restore_result(
    restore_success: bool,
    continuation_started: bool,
    evaluator_result: dict[str, Any],
    stop_reason: str,
) -> str:
    """Classify the restore-continuation outcome."""
    if not restore_success:
        return "restore_failed"
    if not continuation_started:
        return "continuation_failed"
    if task_success(evaluator_result):
        return "restore_continue_success"
    if stop_reason in {"task_completed", "max_agent_turns_reached", "max_environment_executions_reached"}:
        return "restore_continue_completed_with_task_failure"
    return "continuation_failed"


def task_success(evaluator_result: dict[str, Any]) -> bool:
    """Extract a boolean success flag from AppWorld evaluator output."""
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


def print_run_config(config: dict[str, Any]) -> None:
    """Print the safe run configuration."""
    print("Checkpoint restore continuation config:")
    for key in [
        "task_id",
        "checkpoint_id",
        "source_agent_turn_index",
        "source_environment_execution_index",
        "namespace_restore_mode",
        "agent_model",
        "agent_name",
        "max_agent_turns_after_restore",
        "max_environment_executions_after_restore",
        "max_cost_cny",
        "api_endpoint",
    ]:
        print(f"{key}: {config[key]}")


def clean_output_dir(output_dir: Path) -> None:
    """Remove stale restore-continuation artifacts."""
    for name in [
        "config.json",
        "turn_events.jsonl",
        "sanitized_messages.jsonl",
        "evaluator_result.json",
        "restore_continue_report.md",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_report(
    output_dir: Path,
    config: dict[str, Any],
    events: list[dict[str, Any]],
    evaluator_result: dict[str, Any],
    category: str,
    restore_success: bool,
    continuation_started: bool,
) -> None:
    """Write a concise restore-continuation report."""
    task_completed = bool(events and events[-1].get("task_completed"))
    complete_calls = [event for event in events if "complete_task" in event.get("generated_code", "")]
    total_input = sum(event.get("agent_input_tokens") or 0 for event in events)
    total_output = sum(event.get("agent_output_tokens") or 0 for event in events)
    final_cost = next((event.get("cumulative_cost") for event in reversed(events) if event.get("cumulative_cost") is not None), None)
    lines = [
        "# Checkpoint Restore Continuation Smoke Test",
        "",
        f"- Category: `{category}`",
        f"- Task: `{config['task_id']}`",
        f"- Checkpoint: `{config['checkpoint_id']}`",
        f"- Restored from agent turn: `{config['source_agent_turn_index']}`",
        f"- Restored from environment execution: `{config['source_environment_execution_index']}`",
        f"- Namespace restore mode: `{config['namespace_restore_mode']}`",
        f"- Restore success: `{restore_success}`",
        f"- Continuation executed environment: `{continuation_started}`",
        f"- Continuation turns: `{len(events)}`",
        f"- complete_task calls after restore: `{len(complete_calls)}`",
        f"- Final task_completed: `{task_completed}`",
        f"- Official evaluator success: `{task_success(evaluator_result)}`",
        f"- Agent input tokens after restore: `{total_input}`",
        f"- Agent output tokens after restore: `{total_output}`",
        f"- Estimated continuation cost CNY: `{final_cost}`",
        "",
        "## Notes",
        "",
        "- The run restores AppWorld database state with `world.load_state(...)`.",
        "- Agent visible messages and last execution output are restored from the checkpoint record.",
        "- The checkpoint reports `namespace_restore_mode = replay_required`, so Python local variables are not independently serialized; the model must rely on visible history unless the environment state already contains the required state.",
        "- Evaluator failure is treated as model/task failure, not as restore failure, when the restored environment continues correctly.",
    ]
    output_dir.joinpath("restore_continue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
