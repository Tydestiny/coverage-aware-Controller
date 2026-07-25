"""Run one real AppWorld Controller integration smoke test."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.controller.smoke import DeterministicSmokeController
from controller_agent.branching.checkpoint_capture_hook import RuntimeCheckpointCaptureHook
from controller_agent.integration.appworld_adapter import ControlledAppWorldRunner, GUIDANCE_MARKER, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import ControllerConfig


MODEL_NAME = "qwen3-coder-flash-2025-07-28"
AGENT_NAME = "simplified_react_code_agent"
API_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def main() -> int:
    """Run the smoke test CLI."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    task_output_dir = appworld_task_output_dir(appworld_root, args.experiment_name, args.task_id)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else task_output_dir / "controller"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)
    load_dotenv(appworld_root / "experiments" / ".env")
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)

    config = build_config(args, appworld_root)
    write_json(output_dir / "config.json", config)
    print_run_config(config)
    if not args.execute:
        write_report(output_dir, config, [], [], [], {"dry_run": True}, "budget_exhausted", "dry_run")
        return 0

    try:
        return run_once(args, appworld_root, output_dir, config)
    except Exception as exc:
        if is_infrastructure_or_api_error(exc) and args.retry_once:
            return run_once(args, appworld_root, output_dir, config, retry_index=1)
        write_failure(output_dir, config, exc)
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task-id", default="23cf851_1")
    parser.add_argument("--max-agent-turns", type=int, default=10_000)
    parser.add_argument("--max-environment-executions", type=int, default=10_000)
    parser.add_argument("--max-controller-calls", type=int, default=10_000)
    parser.add_argument("--max-controller-blocks", type=int, default=10_000)
    parser.add_argument("--max-extra-agent-turns-after-block", type=int, default=10_000)
    parser.add_argument("--max-cost-cny", type=float, default=float("inf"))
    parser.add_argument("--retry-once", action="store_true", default=True)
    parser.add_argument("--capture-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-root", default=str(REPO_ROOT / "results" / "checkpoints"))
    parser.add_argument("--checkpoint-run-id", default="controller_api_smoke")
    parser.add_argument("--checkpoint-layout", choices=["task_local", "central"], default="task_local")
    parser.add_argument("--experiment-name", default="controlled_react_checkpoint_collection")
    parser.add_argument(
        "--appworld-root",
        default=str(REPO_ROOT.parent / "appworld"),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def run_once(
    args: argparse.Namespace,
    appworld_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    retry_index: int = 0,
) -> int:
    """Run one real AppWorld smoke test."""
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    if str(appworld_root) not in sys.path:
        sys.path.insert(0, str(appworld_root))
    ensure_appworld_imports()
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401

    agent_config = build_agent_config(args, appworld_root)
    base_agent = Agent.from_dict(agent_config)
    logger = ExperimentLogger()
    controlled = ControlledReActAgent(
        base_agent,
        controller=DeterministicSmokeController(),
        config=ControllerConfig(
            enabled=True,
            max_controller_calls_per_task=args.max_controller_calls,
            max_controller_blocks=args.max_controller_blocks,
        ),
        logger=logger,
    )
    budget = RunnerBudget(
        max_agent_turns=args.max_agent_turns,
        max_environment_executions=args.max_environment_executions,
        max_controller_calls=args.max_controller_calls,
        max_controller_blocks=args.max_controller_blocks,
        max_extra_agent_turns_after_block=args.max_extra_agent_turns_after_block,
        max_cost_cny=args.max_cost_cny,
    )
    checkpoint_hook = (
        RuntimeCheckpointCaptureHook(
            root_dir=Path(args.checkpoint_root).resolve(),
            run_id=args.checkpoint_run_id,
            task_id=args.task_id,
            appworld_code_version=read_text_if_exists(appworld_root / "version" / "code.txt"),
            appworld_data_version=read_text_if_exists(appworld_root / "version" / "data.txt"),
            base_dir=checkpoint_base_dir(args, appworld_root),
        )
        if args.capture_checkpoints
        else None
    )
    runner = ControlledAppWorldRunner(
        controlled,
        budget=budget,
        checkpoint_capture_hook=checkpoint_hook,
    )
    evaluator_result: dict[str, Any] = {}
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=args.experiment_name,
            random_seed=100,
            raise_on_extra_parameters=True,
        ):
            with AppWorld(task_id=args.task_id) as world:
                runner.run_initialized_world(world)
                evaluator_result = evaluate_world(world)
    finally:
        os.chdir(previous_cwd)
    wall_time = time.perf_counter() - started
    events = [with_wall_total(event, wall_time) for event in runner.turn_events]
    messages = sanitize_messages(getattr(base_agent, "messages", []))
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "turn_events.jsonl", events)
    write_jsonl(output_dir / "controller_events.jsonl", sanitize_messages(logger.events))
    write_json(output_dir / "evaluator_result.json", evaluator_result)
    checkpoint_manifest = checkpoint_manifest_path(args, appworld_root) if args.capture_checkpoints else None
    category = classify_result(events, messages, runner.stop_reason, evaluator_result)
    write_report(
        output_dir,
        config | {
            "retry_index": retry_index,
            "checkpoint_manifest": str(checkpoint_manifest) if checkpoint_manifest else None,
            "checkpoint_hook_errors": runner.checkpoint_hook_errors,
        },
        events,
        logger.events,
        messages,
        evaluator_result,
        category,
        runner.stop_reason,
    )
    return 0 if category in {"integration_success", "model_task_failure"} else 1


def ensure_appworld_imports() -> None:
    """Fail early when the script is not run in the AppWorld environment."""
    try:
        import appworld  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("Run this script with the AppWorld .venv Python or uv environment.") from exc


def build_agent_config(args: argparse.Namespace, appworld_root: Path) -> dict[str, Any]:
    """Build the AppWorld simplified ReAct agent config."""
    return {
        "type": AGENT_NAME,
        "model_config": {
            "client_name": "openai",
            "api_type": "chat_completions",
            "base_url": API_ENDPOINT,
            "api_key_env_name": "DASHSCOPE_API_KEY",
            "name": MODEL_NAME,
            "temperature": 0.0,
            "seed": 100,
            "max_completion_tokens": 65536,
            "drop_reasoning_content": False,
            "cost_per_token": {
                "input_cache_hit": 1.44e-07,
                "input_cache_miss": 1.44e-07,
                "input_cache_write": 0.0,
                "output": 5.74e-07,
            },
            "retry_after_n_seconds": 15,
            "use_cache": False,
            "max_retries": 1,
        },
        "appworld_config": {
            "random_seed": 100,
            "raise_on_extra_parameters": True,
        },
        "logger_config": {
            "color": False,
            "verbose": True,
        },
        "usage_tracker_config": {
            "max_cost_overall": args.max_cost_cny,
            "max_cost_per_task": args.max_cost_cny,
            "max_output_tokens_per_task": float("inf"),
        },
        "prompt_file_path": str(appworld_root / "experiments" / "prompts" / "react_code_agent" / "instructions.txt"),
        "ignore_multiple_calls": True,
        "max_prompt_length": None,
        "max_output_length": None,
        "max_steps": None,
        "log_lm_calls": True,
        "skip_if_finished": False,
    }


def build_config(args: argparse.Namespace, appworld_root: Path) -> dict[str, Any]:
    """Build public smoke-test config without secrets."""
    return {
        "task_id": args.task_id,
        "agent_model": MODEL_NAME,
        "agent_name": AGENT_NAME,
        "controller_type": "DeterministicSmokeController",
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_controller_calls": args.max_controller_calls,
        "max_controller_blocks": args.max_controller_blocks,
        "max_extra_agent_turns_after_block": args.max_extra_agent_turns_after_block,
        "maximum_cost_cny": args.max_cost_cny,
        "api_endpoint": API_ENDPOINT,
        "appworld_root": str(appworld_root),
        "temperature": 0.0,
        "capture_checkpoints": args.capture_checkpoints,
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "checkpoint_run_id": args.checkpoint_run_id,
        "checkpoint_layout": args.checkpoint_layout,
        "experiment_name": args.experiment_name,
        "task_output_dir": str(appworld_task_output_dir(appworld_root, args.experiment_name, args.task_id)),
    }


def print_run_config(config: dict[str, Any]) -> None:
    """Print the safe run configuration."""
    print("Controller integration smoke config:")
    for key in [
        "task_id",
        "agent_model",
        "agent_name",
        "controller_type",
        "max_agent_turns",
        "max_environment_executions",
        "max_controller_calls",
        "max_controller_blocks",
        "max_extra_agent_turns_after_block",
        "maximum_cost_cny",
        "api_endpoint",
    ]:
        print(f"{key}: {config[key]}")


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines without printing secrets."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_text_if_exists(path: Path) -> str | None:
    """Return stripped file text when present."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def appworld_task_output_dir(appworld_root: Path, experiment_name: str, task_id: str) -> Path:
    """Return the AppWorld task output directory used for trajectory files."""
    return appworld_root / "experiments" / "outputs" / experiment_name / "tasks" / task_id


def checkpoint_base_dir(args: argparse.Namespace, appworld_root: Path) -> Path | None:
    """Return an optional task-local checkpoint directory."""
    if args.checkpoint_layout == "central":
        return None
    return appworld_task_output_dir(appworld_root, args.experiment_name, args.task_id) / "checkpoints"


def checkpoint_manifest_path(args: argparse.Namespace, appworld_root: Path) -> Path:
    """Return the expected runtime checkpoint manifest path."""
    base_dir = checkpoint_base_dir(args, appworld_root)
    if base_dir is not None:
        return base_dir / "manifest.json"
    return Path(args.checkpoint_root).resolve() / args.checkpoint_run_id / args.task_id / "manifest.json"


def evaluate_world(world: Any) -> dict[str, Any]:
    """Best-effort evaluator call."""
    try:
        result = world.evaluate(suppress_errors=True)
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__, "error": str(exc)}
    if hasattr(result, "to_dict"):
        return {"available": True, "result": result.to_dict()}
    if hasattr(result, "dict"):
        return {"available": True, "result": result.dict()}
    return {"available": True, "result": str(result)}


def integration_success(events: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    """Evaluate integration success criteria."""
    complete_events = [event for event in events if "complete_task" in event.get("generated_code", "")]
    blocked = [event for event in complete_events if not event.get("execution_allowed")]
    after_block_events = events[events.index(blocked[0]) + 1 :] if blocked else []
    allowed_execution_after_block = [
        event for event in after_block_events if event.get("execution_allowed") and event.get("world_batch_execute_called")
    ]
    return bool(
        complete_events
        and blocked
        and blocked[0]["world_batch_execute_called"] is False
        and any(str(message.get("content", "")).startswith(GUIDANCE_MARKER) for message in messages)
        and any(event.get("generated_code") for event in after_block_events)
        and allowed_execution_after_block
        and all(not (event.get("generated_code") == "" and event.get("world_batch_execute_called")) for event in events)
    )


def classify_result(
    events: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    stop_reason: str | None,
    evaluator_result: dict[str, Any],
) -> str:
    """Classify the final smoke-test result."""
    if integration_success(events, messages):
        return "integration_success" if task_success(evaluator_result) else "model_task_failure"
    if not events:
        return "environment_or_api_failure"
    complete_events = [event for event in events if "complete_task" in event.get("generated_code", "")]
    if not complete_events:
        return "model_format_failure"
    if complete_events and complete_events[0].get("execution_allowed"):
        return "environment_or_api_failure"
    if stop_reason in {
        "cost_limit_reached",
        "max_agent_turns_reached",
        "max_environment_executions_reached",
        "blocked_extra_turns_exhausted",
    }:
        return "budget_exhausted"
    if stop_reason == "model_format_failure":
        return "model_format_failure"
    return "environment_or_api_failure"


def task_success(evaluator_result: dict[str, Any]) -> bool:
    """Best-effort extraction of official evaluator success."""
    result = evaluator_result.get("result")
    if isinstance(result, dict):
        for key in ("success", "task_completed"):
            if key in result:
                return bool(result[key])
        if "score" in result:
            try:
                return float(result["score"]) >= 1.0
            except (TypeError, ValueError):
                return False
    return False


def is_infrastructure_or_api_error(exc: Exception) -> bool:
    """Return whether an exception appears retryable."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(term in text for term in ["timeout", "connection", "api", "rate", "http", "dashscope"])


def sanitize_messages(value: Any) -> Any:
    """Remove secrets from nested logs and messages."""
    if isinstance(value, list):
        return [sanitize_messages(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(r"api[_-]?key|authorization|secret", str(key), re.IGNORECASE):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_messages(item)
        return sanitized
    if isinstance(value, str):
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}", "sk-[REDACTED]", value)
        return re.sub(
            r"(?i)(\b(?:password|access_?token|refresh_?token|authorization|api_?key|secret|credential)\b(?:\\?[\"']?\s*)(?:=|:|\bis\b)\s*(?:\\?[\"']?))([^'\"\s,}\]\\]+)",
            r"\1[REDACTED]",
            value,
        )
    return value


def with_wall_total(event: dict[str, Any], total_wall_time: float) -> dict[str, Any]:
    """Attach total wall time to a turn event."""
    return event | {"total_wall_time": total_wall_time}


def write_json(path: Path, data: Any) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_messages(data), indent=2, ensure_ascii=False), encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    """Remove stale smoke-test artifacts before a fresh run."""
    for name in [
        "config.json",
        "turn_events.jsonl",
        "controller_events.jsonl",
        "evaluator_result.json",
        "final_smoke_report.md",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_jsonl(path: Path, rows: list[Any]) -> None:
    """Write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(sanitize_messages(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_failure(output_dir: Path, config: dict[str, Any], exc: Exception) -> None:
    """Write failure artifacts."""
    failure = {"error_type": type(exc).__name__, "error": str(exc)}
    write_json(output_dir / "evaluator_result.json", {"available": False, "failure": failure})
    write_report(output_dir, config, [], [], [], {"failure": failure}, "environment_or_api_failure", "runner_or_api_error")


def write_report(
    output_dir: Path,
    config: dict[str, Any],
    events: list[dict[str, Any]],
    controller_events: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    evaluator_result: dict[str, Any],
    category: str,
    stop_reason: str | None,
) -> None:
    """Write the Markdown smoke-test report."""
    complete_events = [event for event in events if "complete_task" in event.get("generated_code", "")]
    blocked = [event for event in complete_events if not event.get("execution_allowed")]
    guidance_present = any(str(message.get("content", "")).startswith(GUIDANCE_MARKER) for message in messages)
    second_allowed = len(complete_events) > 1 and bool(complete_events[1].get("execution_allowed"))
    completed = bool(events and events[-1].get("task_completed"))
    blocked_index = events.index(blocked[0]) if blocked else None
    after_block_events = events[blocked_index + 1 :] if blocked_index is not None else []
    non_empty_after_block = any(event.get("generated_code") for event in after_block_events)
    any_execution_after_block = any(
        event.get("execution_allowed") and event.get("world_batch_execute_called") for event in after_block_events
    )
    agent_input = sum(event.get("agent_input_tokens") or 0 for event in events)
    agent_output = sum(event.get("agent_output_tokens") or 0 for event in events)
    controller_input = sum(event.get("controller_input_tokens") or 0 for event in events)
    controller_output = sum(event.get("controller_output_tokens") or 0 for event in events)
    cost = events[-1].get("total_cost") if events else None
    report = f"""# Controller Integration Smoke Test

Result category: {category}

Termination reason: {stop_reason}

## Configuration

- task_id: {config.get("task_id")}
- agent model: {config.get("agent_model")}
- agent: {config.get("agent_name")}
- controller: {config.get("controller_type")}
- max_agent_turns: {config.get("max_agent_turns")}
- max_environment_executions: {config.get("max_environment_executions")}
- max_controller_calls: {config.get("max_controller_calls")}
- max_controller_blocks: {config.get("max_controller_blocks")}
- max_extra_agent_turns_after_block: {config.get("max_extra_agent_turns_after_block")}
- max cost: {config.get("maximum_cost_cny")}
- endpoint: {config.get("api_endpoint")}
- checkpoint capture: {config.get("capture_checkpoints")}
- checkpoint layout: {config.get("checkpoint_layout")}
- checkpoint manifest: {config.get("checkpoint_manifest")}
- AppWorld task output: {config.get("task_output_dir")}
- checkpoint hook errors: {json.dumps(config.get("checkpoint_hook_errors", []), ensure_ascii=False)}

## Questions

1. complete_task intercepted: {bool(blocked)}
2. blocked turn skipped world.batch_execute: {bool(blocked and blocked[0].get("world_batch_execute_called") is False)}
3. guidance entered next model context: {guidance_present}
4. model generated a new action after guidance: {non_empty_after_block}
5. repeated complete_task occurred: {len(complete_events) > 1}
6. second complete_task allowed: {second_allowed}
7. task completed: {completed}; evaluator: {json.dumps(evaluator_result, ensure_ascii=False)[:2000]}
8. token and cost: agent_input={agent_input}, agent_output={agent_output}, controller_input={controller_input}, controller_output={controller_output}, total_cost={cost}
9. ordinary code or second complete_task executed after block: {any_execution_after_block}
10. suitable for real LLM Controller development: {category in {"integration_success", "model_task_failure"}}
"""
    (output_dir / "final_smoke_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
