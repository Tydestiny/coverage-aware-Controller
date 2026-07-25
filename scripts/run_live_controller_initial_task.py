"""Run one initial-state AppWorld task through the live Controller loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend
from controller_agent.integration.appworld_adapter import ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import ControllerConfig
from scripts.run_controller_api_smoke import (
    AGENT_NAME,
    API_ENDPOINT,
    MODEL_NAME,
    build_agent_config,
    load_dotenv,
    sanitize_messages,
    write_json,
    write_jsonl,
)
from scripts.run_checkpoint_restore_continue import ensure_appworld_agents_package


def parse_args() -> argparse.Namespace:
    """Parse the fixed single-task live run configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="50e1ac9_1")
    parser.add_argument("--dataset-name", default="dev")
    parser.add_argument("--experiment-name", default="controller_live_initial_50e1ac9_1")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "controller_initial_50e1ac9_1"))
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--max-agent-turns", type=int, default=10000)
    parser.add_argument("--max-environment-executions", type=int, default=10000)
    parser.add_argument("--max-controller-calls", type=int, default=2)
    parser.add_argument("--max-controller-blocks", type=int, default=2)
    parser.add_argument("--max-cost-cny", type=float, default=float("inf"))
    return parser.parse_args()


def main() -> int:
    """Run one task from its initial state and persist controller audit files."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "dataset_name": args.dataset_name,
        "task_id": args.task_id,
        "selection_seed": None,
        "selection_reason": "First dev task without an existing current Controller smoke result; fixed before execution.",
        "excluded_task_ids": ["287e338_1"],
        "execution_mode": "initial_state",
    }
    write_json(output_dir / "task_selection.json", selection)
    load_dotenv(appworld_root / "experiments" / ".env")
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)

    config = {
        "task_id": args.task_id,
        "dataset_name": args.dataset_name,
        "experiment_name": args.experiment_name,
        "agent_name": AGENT_NAME,
        "agent_model": MODEL_NAME,
        "controller_model": MODEL_NAME,
        "controller_backend": "LiveLLMBackend",
        "context_policy": "FULL_CONTEXT",
        "enabled_triggers": ["before_finish"],
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_controller_calls": args.max_controller_calls,
        "max_controller_blocks": args.max_controller_blocks,
        "max_cost_cny": args.max_cost_cny,
        "api_endpoint": API_ENDPOINT,
    }
    write_json(output_dir / "run_summary.json", {"config": config, "status": "started"})
    print(json.dumps({key: value for key, value in config.items() if key != "api_key"}, ensure_ascii=False, indent=2))

    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=args.max_agent_turns,
        max_extra_agent_turns_after_block=10000,
    )
    agent_config = build_agent_config(agent_args, appworld_root)
    agent_config["max_steps"] = 10000
    agent_config["usage_tracker_config"]["max_output_tokens_per_task"] = 10_000_000
    base_agent = None
    runner = None
    logger = ExperimentLogger()
    summary: dict[str, Any] = {"config": config, "status": "error"}
    close_error: str | None = None
    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    started = time.perf_counter()
    try:
        from appworld import AppWorld
        ensure_appworld_agents_package(appworld_root)
        from appworld_agents.code.simplified.agent import Agent
        import appworld_agents.code.simplified.react_code_agent  # noqa: F401

        base_agent = Agent.from_dict(agent_config)
        controlled = ControlledReActAgent(
            base_agent,
            controller=Controller(backend=LiveLLMBackend()),
            config=ControllerConfig(
                enabled=True,
                max_controller_calls_per_task=args.max_controller_calls,
                max_controller_blocks=args.max_controller_blocks,
                max_consecutive_blocks=args.max_controller_blocks,
            ),
            logger=logger,
        )
        runner = ControlledAppWorldRunner(
            controlled,
            budget=RunnerBudget(
                max_agent_turns=args.max_agent_turns,
                max_environment_executions=args.max_environment_executions,
                max_controller_calls=args.max_controller_calls,
                max_controller_blocks=args.max_controller_blocks,
                max_extra_agent_turns_after_block=10000,
                max_cost_cny=args.max_cost_cny,
            ),
        )
        world = None
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=args.experiment_name,
            random_seed=100,
            raise_on_extra_parameters=True,
        ):
            world = AppWorld(task_id=args.task_id)
            try:
                runner.run_initialized_world(world)
            finally:
                try:
                    world.close()
                except Exception as exc:
                    close_error = f"{type(exc).__name__}: {exc}"
        summary = {
            "config": config,
            "status": "completed",
            "termination_reason": runner.stop_reason,
            "controller_calls": controlled.runtime_state.total_calls,
            "controller_blocks": controlled.runtime_state.total_blocks,
            "close_error": close_error,
            "wall_time_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        summary = {
            "config": config,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "close_error": close_error,
            "wall_time_seconds": time.perf_counter() - started,
        }
    finally:
        os.chdir(previous_cwd)

    events = [] if runner is None else sanitize_messages(runner.turn_events)
    controller_events = sanitize_messages(logger.events)
    write_jsonl(output_dir / "turn_events.jsonl", events)
    write_jsonl(output_dir / "controller_events.jsonl", controller_events)
    summary["event_count"] = len(events)
    summary["controller_event_count"] = len(controller_events)
    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "evaluator_result.json", {"pending": True, "evaluation_mode": "official_subprocess"})
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
