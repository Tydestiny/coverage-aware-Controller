"""Run one restored AppWorld checkpoint through the live Controller loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger
from controller_agent.integration.dashscope_no_thinking import install_deepseek_no_thinking_adapter
from controller_agent.branching.namespace_restore import reset_world_interaction_logs, restore_python_namespace_from_prefix
from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend
from controller_agent.controller.policy_g0 import load_g0_system_prompt
from controller_agent.integration.appworld_adapter import AppWorldInterceptionApplied, ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.logger import ExperimentLogger
from controller_agent.state.schemas import ControllerConfig
from scripts.run_checkpoint_restore_continue import (
    copy_state_directory,
    ensure_appworld_agents_package,
    load_checkpoint,
)
from scripts.run_controller_api_smoke import (
    API_ENDPOINT,
    OPENAI_GPT4O_MINI,
    DEEPSEEK_V4_FLASH,
    appworld_task_output_dir,
    build_agent_config,
    evaluate_world,
    load_dotenv,
    sanitize_messages,
    write_json,
    write_jsonl,
)


DEFAULT_CONTROLLER_MODEL = "qwen3.7-max-2026-05-20"


def parse_args() -> argparse.Namespace:
    """Parse the one-run live Controller smoke configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task-id", default="287e338_1")
    parser.add_argument("--agent-model", default="qwen3-coder-flash-2025-07-28")
    parser.add_argument("--checkpoint-id", default="287e338_1_post_0008")
    parser.add_argument(
        "--source-task-dir",
        default=str(
            REPO_ROOT.parent
            / "appworld"
            / "experiments"
            / "outputs"
            / "controlled_react_checkpoint_collection"
            / "tasks"
            / "287e338_1"
        ),
    )
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "controller_live_checkpoint_smoke"))
    parser.add_argument("--max-agent-turns", type=int, default=10_000)
    parser.add_argument("--max-environment-executions", type=int, default=10_000)
    parser.add_argument("--max-controller-calls", type=int, default=10_000)
    parser.add_argument("--max-controller-blocks", type=int, default=10_000)
    parser.add_argument("--max-extra-agent-turns-after-block", type=int, default=10_000)
    parser.add_argument("--max-cost-cny", type=float, default=float("inf"))
    parser.add_argument("--branch", choices=("baseline", "controller"), default="controller")
    parser.add_argument("--disable-controller", action="store_true")
    parser.add_argument("--external-action", default=None)
    parser.add_argument("--external-guidance", default=None)
    parser.add_argument("--skip-inprocess-evaluate", action="store_true")
    parser.add_argument(
        "--retain-smoke-artifacts",
        action="store_true",
        help="Write controller_response.json and smoke_test_report.md for a standalone smoke test.",
    )
    parser.add_argument(
        "--log-lm-calls",
        action="store_true",
        help="Preserve AppWorld lm_calls.jsonl for an archived branch result.",
    )
    parser.add_argument("--candidate-audit", default=None)
    parser.add_argument("--experiment-name", default="controller_live_checkpoint_smoke")
    return parser.parse_args()


def main() -> int:
    """Restore one checkpoint and execute the existing controlled runner once."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    source_task_dir = Path(args.source_task_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = load_checkpoint(source_task_dir, args.checkpoint_id)
    config = {
        "task_id": args.task_id,
        "checkpoint_id": args.checkpoint_id,
        "agent_model": (
            OPENAI_GPT4O_MINI
            if args.agent_model == "openai/gpt-4o-mini"
            else DEEPSEEK_V4_FLASH
            if args.agent_model == DEEPSEEK_V4_FLASH
            else args.agent_model
        ),
        "controller_model": DEFAULT_CONTROLLER_MODEL,
        "controller_backend": "LiveLLMBackend",
        "branch": args.branch,
        "controller_enabled": not args.disable_controller,
        "external_intervention": (
            {"action": args.external_action, "guidance": args.external_guidance}
            if args.external_guidance
            else None
        ),
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_controller_calls": args.max_controller_calls,
        "max_controller_blocks": args.max_controller_blocks,
        "max_consecutive_blocks": args.max_controller_blocks,
        "max_cost_cny": args.max_cost_cny,
        "api_endpoint": API_ENDPOINT,
        "source_checkpoint": str(source_task_dir / "checkpoints" / "post_observation" / f"{args.checkpoint_id}.json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    if not args.execute:
        write_json(output_dir / "evaluator_result.json", {"dry_run": True})
        return 0

    load_dotenv(appworld_root / "experiments" / ".env")
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)
    if str(appworld_root) not in sys.path:
        sys.path.insert(0, str(appworld_root))
    ensure_appworld_agents_package(appworld_root)
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent, ExecutionIO
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401
    if args.agent_model == DEEPSEEK_V4_FLASH:
        install_deepseek_no_thinking_adapter()

    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=args.max_agent_turns,
        max_extra_agent_turns_after_block=10_000,
    )
    agent_config = build_agent_config(agent_args, appworld_root, args.agent_model)
    agent_config["max_steps"] = None
    agent_config["log_lm_calls"] = args.log_lm_calls
    base_agent = Agent.from_dict(agent_config)
    logger = ExperimentLogger()
    controller = Controller(backend=LiveLLMBackend(model_name=DEFAULT_CONTROLLER_MODEL))
    controlled = ControlledReActAgent(
        base_agent,
        controller=controller,
        config=ControllerConfig(
            enabled=not args.disable_controller,
            max_controller_calls_per_task=args.max_controller_calls,
            max_controller_blocks=args.max_controller_blocks,
            max_consecutive_blocks=args.max_controller_blocks,
        ),
        logger=logger,
    )
    branch_logger = BranchRuntimeLogger(
        output_dir,
        branch=args.branch,
        checkpoint_last_step=int(checkpoint["agent_turn_index"]),
    )
    branch_logger.write_checkpoint_manifest(
        output_dir / "checkpoint_manifest.json",
        checkpoint_id=args.checkpoint_id,
        state_path=str(source_task_dir / "checkpoints" / str(checkpoint["appworld_state_id"])),
        source_trajectory=str(source_task_dir),
        source_logs=str(source_task_dir / "logs"),
    )
    runner = ControlledAppWorldRunner(
        controlled,
        budget=RunnerBudget(
            max_agent_turns=args.max_agent_turns,
            max_environment_executions=args.max_environment_executions,
            max_controller_calls=args.max_controller_calls,
            max_controller_blocks=args.max_controller_blocks,
            max_extra_agent_turns_after_block=args.max_extra_agent_turns_after_block,
            max_cost_cny=args.max_cost_cny,
        ),
        branch_runtime_logger=branch_logger,
        controller_prompt_hash=hashlib.sha256(load_g0_system_prompt().encode("utf-8")).hexdigest(),
    )

    restore_result: dict[str, Any]
    evaluator_result: dict[str, Any] = {}
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    raw_logs_source: Path | None = None
    os.chdir(appworld_root)
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=args.experiment_name,
            random_seed=100,
            raise_on_extra_parameters=True,
        ):
            with AppWorld(task_id=args.task_id) as world:
                base_agent.initialize(world)
                output_task_dir = appworld_task_output_dir(
                    appworld_root, args.experiment_name, args.task_id
                )
                copy_state_directory(source_task_dir, output_task_dir, checkpoint)
                namespace_result = restore_python_namespace_from_prefix(world, checkpoint)
                restore_result = namespace_result.to_dict()
                if not namespace_result.success:
                    raise RuntimeError(namespace_result.error or "namespace restore failed")
                world.load_state(str(checkpoint["appworld_state_id"]))
                reset_world_interaction_logs(world)
                base_agent.messages = checkpoint["agent_messages"]
                base_agent.step_number = int(checkpoint["agent_turn_index"])
                _inject_external_guidance(base_agent, args.external_action, args.external_guidance)
                raw_logs_source = output_task_dir / "logs"
                if args.candidate_audit:
                    _run_cached_pre_action(
                        Path(args.candidate_audit).resolve(),
                        controlled,
                        runner,
                        world,
                        ExecutionIO,
                    )
                else:
                    runner.run_restored_world(world)
                evaluator_result = (
                    {"available": False, "reason": "in_process_evaluator_skipped"}
                    if args.skip_inprocess_evaluate
                    else evaluate_world(world)
                )
                _flush_agent_runtime_logs(base_agent)
    except Exception as exc:
        restore_result = locals().get("restore_result", {"success": False, "error": f"{type(exc).__name__}: {exc}"})
        evaluator_result = {"available": False, "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        os.chdir(previous_cwd)

    if raw_logs_source is not None:
        branch_logger.archive_raw_logs(raw_logs_source)

    events = sanitize_messages(runner.turn_events)
    controller_events = sanitize_messages(logger.events)
    messages = sanitize_messages(getattr(base_agent, "messages", []))
    response = controller.last_response
    response_data = None if response is None else {
        "decision": {
            "action": response.decision.action.value,
            "reason": response.decision.reason,
            "instruction": response.decision.instruction,
        },
        "model_name": response.model_name,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_seconds": response.latency_seconds,
        "cost_cny": response.cost_cny,
        "raw_response": response.raw_response,
    }
    write_jsonl(output_dir / "turn_events.jsonl", events)
    write_jsonl(output_dir / "controller_events.jsonl", controller_events)
    write_jsonl(output_dir / "sanitized_messages.jsonl", messages)
    if args.retain_smoke_artifacts:
        write_json(output_dir / "controller_response.json", response_data)
    write_json(output_dir / "evaluator_result.json", evaluator_result)
    write_json(output_dir / "restore_result.json", restore_result)
    if args.retain_smoke_artifacts:
        _write_report(
            output_dir,
            config,
            restore_result,
            events,
            controller_events,
            response_data,
            evaluator_result,
            time.perf_counter() - started,
        )
    return 0 if restore_result.get("success") else 1


def _write_report(
    output_dir: Path,
    config: dict[str, Any],
    restore_result: dict[str, Any],
    events: list[dict[str, Any]],
    controller_events: list[dict[str, Any]],
    response: dict[str, Any] | None,
    evaluator_result: dict[str, Any],
    wall_time: float,
) -> None:
    """Write a concise auditable report without raw credentials."""
    blocked = [event for event in events if not event.get("execution_allowed")]
    executed = [event for event in events if event.get("world_batch_execute_called")]
    guidance = any("[CONTROLLER_GUIDANCE]" in str(message.get("content", "")) for message in controller_events + [])
    report = "\n".join([
        "# Live Controller Checkpoint Smoke",
        "",
        f"- checkpoint restored: {restore_result.get('success')}",
        f"- controller decision: {None if response is None else response['decision']}",
        f"- blocked proposed action: {bool(blocked)}",
        f"- environment executions after restore: {len(executed)}",
        f"- guidance injection recorded: {any(event.get('guidance_injected') for event in events)}",
        f"- runner termination: {runner_termination(events)}",
        f"- evaluator: {json.dumps(evaluator_result, ensure_ascii=False)[:1000]}",
        f"- wall_time_seconds: {wall_time:.3f}",
    ])
    (output_dir / "smoke_test_report.md").write_text(report + "\n", encoding="utf-8")


def runner_termination(events: list[dict[str, Any]]) -> str | None:
    """Return the final recorded termination reason."""
    return events[-1].get("termination_reason") if events else None


def _flush_agent_runtime_logs(agent: Any) -> None:
    """Flush AppWorld Agent log streams before copying branch raw logs."""
    logger = getattr(agent, "logger", None)
    if logger is None:
        return
    jsonl_stream = getattr(logger, "file_jsonl", None)
    if jsonl_stream is not None and not jsonl_stream.closed:
        jsonl_stream.flush()
    console = getattr(logger, "file_console", None)
    console_stream = getattr(console, "file", None)
    if console_stream is not None and not console_stream.closed:
        console_stream.flush()


def _inject_external_guidance(agent: Any, action: str | None, guidance: str | None) -> None:
    """Inject one non-Controller intervention after restore and before generation."""
    if not guidance:
        return
    messages = getattr(agent, "messages", None)
    if not isinstance(messages, list):
        raise TypeError("AppWorld agent must expose a list-like messages attribute.")
    heading = "[ORACLE_INSTRUCTION]"
    action_line = f"Action: {action}\n" if action else ""
    content = f"{heading}\n{action_line}\n{guidance.strip()}\n"
    if any(message.get("content") == content for message in messages if isinstance(message, dict)):
        return
    messages.append({"role": "user", "content": content})


def _run_cached_pre_action(
    audit_path: Path,
    controlled: ControlledReActAgent,
    runner: ControlledAppWorldRunner,
    world: Any,
    execution_io_type: type[Any],
) -> None:
    """Run a previously generated real Agent pre-action through the live Controller.

    The audit stores a genuine agent output captured before environment
    execution.  It is reused solely to make the controller checkpoint test
    deterministic; the continuation after a block still calls the real Agent.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output = str(audit["controller_input"]["current_agent_output"])
    code = _first_python_block(output)
    if not code:
        raise ValueError("candidate audit has no executable Python code block")
    base_agent = controlled.agent
    base_agent.messages.append({"role": "assistant", "content": output})
    base_agent.step_number = int(getattr(base_agent, "step_number", 0)) + 1
    controlled.reset_for_task(str(getattr(world, "task_id", "")))
    controlled.runtime_state.agent_turn_count = 1
    execution_inputs = [execution_io_type(content=code, metadata={})]
    context = controlled._build_trigger_context(execution_inputs, [])
    interception = controlled.inspect_proposed_execution(context, original_result=None, original_execution_inputs=execution_inputs)
    controlled.last_interception_result = interception
    applied = controlled.integration_adapter.apply(None, interception, base_agent)
    if isinstance(applied, AppWorldInterceptionApplied):
        runner.first_block_turn = 1
        runner.turn_events.append(runner._blocked_event(applied, world, time.perf_counter()))
        runner.run_restored_world(world, reset_controller=False)
        return
    if not interception.allow_execution:
        raise RuntimeError("blocked action was not handled by AppWorldInterceptionAdapter")
    outputs = world.batch_execute([code])
    controlled.runtime_state.environment_execution_count = 1
    runner.turn_events.append(
        runner._executed_event(
            (execution_inputs, None, None), execution_inputs, world, bool(world.task_completed()), time.perf_counter()
        )
    )
    if not world.task_completed() and outputs:
        runner.run_restored_world(world, reset_controller=False)


def _first_python_block(output: str) -> str:
    """Extract one executable block from a captured assistant output."""
    import re

    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    raise SystemExit(main())
