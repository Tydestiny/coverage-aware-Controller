"""Collect one full-context AppWorld trajectory with dense checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.checkpoint_capture_hook import RuntimeCheckpointCaptureHook
from controller_agent.integration.appworld_adapter import ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import ControllerConfig
from scripts.run_controller_api_smoke import (
    AGENT_NAME,
    API_ENDPOINT,
    MODEL_NAME,
    appworld_task_output_dir,
    build_agent_config,
    evaluate_world,
    load_dotenv,
    read_text_if_exists,
    sanitize_messages,
    write_json,
    write_jsonl,
)


def main() -> int:
    """Collect a single baseline task with runtime checkpoints."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)
    load_dotenv(appworld_root / "experiments" / ".env")

    task_output_dir = appworld_task_output_dir(appworld_root, args.experiment_name, args.task_id)
    output_dir = task_output_dir / "collection"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)

    config = build_config(args, appworld_root, task_output_dir)
    write_json(output_dir / "config.json", config)
    print_run_config(config)
    if not args.execute:
        write_report(output_dir, config, [], {"dry_run": True}, "dry_run", None, [])
        return 0

    if str(appworld_root) not in sys.path:
        sys.path.insert(0, str(appworld_root))
    ensure_appworld_agents_package(appworld_root)
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent

    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=args.max_agent_turns,
        max_extra_agent_turns_after_block=0,
    )
    base_agent = Agent.from_dict(build_agent_config(agent_args, appworld_root))
    controlled = ControlledReActAgent(
        base_agent,
        config=ControllerConfig(enabled=False),
        logger=ExperimentLogger(),
    )
    checkpoint_hook = RuntimeCheckpointCaptureHook(
        root_dir=Path(args.checkpoint_root).resolve(),
        run_id=args.checkpoint_run_id,
        task_id=args.task_id,
        appworld_code_version=read_text_if_exists(appworld_root / "version" / "code.txt"),
        appworld_data_version=read_text_if_exists(appworld_root / "version" / "data.txt"),
        base_dir=task_output_dir / "checkpoints",
    )
    runner = ControlledAppWorldRunner(
        controlled,
        budget=RunnerBudget(
            max_agent_turns=args.max_agent_turns,
            max_environment_executions=args.max_environment_executions,
            max_controller_calls=0,
            max_controller_blocks=0,
            max_extra_agent_turns_after_block=0,
            max_cost_cny=args.max_cost_cny,
        ),
        checkpoint_capture_hook=checkpoint_hook,
    )

    started = time.perf_counter()
    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    evaluator_result: dict[str, Any]
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=args.experiment_name,
            random_seed=args.seed,
            raise_on_extra_parameters=True,
        ):
            with AppWorld(task_id=args.task_id) as world:
                runner.run_initialized_world(world)
                evaluator_result = evaluate_world(world)
    finally:
        os.chdir(previous_cwd)

    wall_time = time.perf_counter() - started
    events = [event | {"total_wall_time_seconds": wall_time} for event in runner.turn_events]
    messages = sanitize_messages(getattr(base_agent, "messages", []))
    category = classify_result(runner.stop_reason, evaluator_result, runner.checkpoint_hook_errors)
    write_jsonl(output_dir / "turn_events.jsonl", events)
    write_jsonl(output_dir / "sanitized_messages.jsonl", messages)
    write_json(output_dir / "evaluator_result.json", evaluator_result)
    write_report(output_dir, config, events, evaluator_result, category, runner.stop_reason, runner.checkpoint_hook_errors)
    return 0 if category in {"collected_success", "collected_task_failure"} else 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-agent-turns", type=int, default=20)
    parser.add_argument("--max-environment-executions", type=int, default=15)
    parser.add_argument("--max-cost-cny", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--experiment-name", default="controlled_react_checkpoint_collection")
    parser.add_argument("--checkpoint-root", default=str(REPO_ROOT / "results" / "checkpoints"))
    parser.add_argument("--checkpoint-run-id", default="baseline_checkpoint_collection")
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    return parser.parse_args()


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
        raise RuntimeError(f"Cannot load appworld_agents from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["appworld_agents"] = module
    spec.loader.exec_module(module)


def build_config(args: argparse.Namespace, appworld_root: Path, task_output_dir: Path) -> dict[str, Any]:
    """Build a public config without secrets."""
    return {
        "task_id": args.task_id,
        "agent_model": MODEL_NAME,
        "agent_name": AGENT_NAME,
        "controller_enabled": False,
        "context_policy": "full_context_no_compaction",
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_cost_cny": args.max_cost_cny,
        "seed": args.seed,
        "api_endpoint": API_ENDPOINT,
        "experiment_name": args.experiment_name,
        "task_output_dir": str(task_output_dir),
        "checkpoint_dir": str(task_output_dir / "checkpoints"),
    }


def print_run_config(config: dict[str, Any]) -> None:
    """Print safe run config."""
    print("Baseline checkpoint collection config:")
    for key in [
        "task_id",
        "agent_model",
        "agent_name",
        "controller_enabled",
        "max_agent_turns",
        "max_environment_executions",
        "max_cost_cny",
        "api_endpoint",
        "task_output_dir",
    ]:
        print(f"{key}: {config[key]}")


def classify_result(
    stop_reason: str | None,
    evaluator_result: dict[str, Any],
    checkpoint_errors: list[str],
) -> str:
    """Classify collection outcome."""
    if checkpoint_errors:
        return "checkpoint_capture_failed"
    if stop_reason in {"runner_or_api_error", "model_format_failure"}:
        return "collection_failed"
    return "collected_success" if task_success(evaluator_result) else "collected_task_failure"


def task_success(evaluator_result: dict[str, Any]) -> bool:
    """Extract official evaluator success."""
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


def clean_output_dir(output_dir: Path) -> None:
    """Remove stale collection report files only."""
    for name in [
        "config.json",
        "turn_events.jsonl",
        "sanitized_messages.jsonl",
        "evaluator_result.json",
        "baseline_collection_report.md",
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
    stop_reason: str | None,
    checkpoint_errors: list[str],
) -> None:
    """Write baseline collection report."""
    task_completed = bool(events and events[-1].get("task_completed"))
    manifest = Path(config["checkpoint_dir"]) / "manifest.json"
    complete_calls = [event for event in events if "complete_task" in event.get("generated_code", "")]
    total_input = sum((event.get("cumulative_tokens") or {}).get("input", 0) for event in events[-1:]) if events else 0
    total_output = sum((event.get("cumulative_tokens") or {}).get("output", 0) for event in events[-1:]) if events else 0
    total_cost = events[-1].get("cumulative_cost") if events else None
    lines = [
        "# Baseline Checkpoint Collection Report",
        "",
        f"- Category: `{category}`",
        f"- Task: `{config['task_id']}`",
        f"- Model: `{config['agent_model']}`",
        f"- Agent: `{config['agent_name']}`",
        f"- Controller enabled: `{config['controller_enabled']}`",
        f"- Stop reason: `{stop_reason}`",
        f"- Task completed: `{task_completed}`",
        f"- Official evaluator success: `{task_success(evaluator_result)}`",
        f"- Agent turns: `{len(events)}`",
        f"- complete_task calls: `{len(complete_calls)}`",
        f"- Input tokens: `{total_input}`",
        f"- Output tokens: `{total_output}`",
        f"- Cost CNY: `{total_cost}`",
        f"- Checkpoint manifest: `{manifest}`",
        f"- Checkpoint errors: `{checkpoint_errors}`",
        "",
        "## Storage Layout",
        "",
        f"- Full AppWorld task output: `{config['task_output_dir']}`",
        f"- Dense checkpoints: `{config['checkpoint_dir']}`",
        f"- Collection report files: `{output_dir}`",
    ]
    (output_dir / "baseline_collection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
