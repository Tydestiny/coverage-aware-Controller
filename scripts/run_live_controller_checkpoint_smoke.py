"""Run one restored AppWorld checkpoint through the live Controller loop."""

from __future__ import annotations

import argparse
import hashlib
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

from controller_agent.branching.namespace_restore import restore_python_namespace_from_prefix
from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import (
    DEFAULT_MODEL as CONTROLLER_MODEL_NAME,
    LiveLLMBackend,
)
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
    appworld_task_output_dir,
    build_agent_config,
    evaluate_world,
    load_dotenv,
    sanitize_messages,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse the one-run live Controller smoke configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task-id", default="287e338_1")
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
    parser.add_argument("--max-agent-turns", type=int, default=3)
    parser.add_argument("--max-environment-executions", type=int, default=3)
    parser.add_argument("--max-controller-calls", type=int, default=1)
    parser.add_argument("--max-controller-blocks", type=int, default=1)
    parser.add_argument("--max-extra-agent-turns-after-block", type=int, default=3)
    parser.add_argument("--max-cost-cny", type=float, default=1.0)
    parser.add_argument("--candidate-audit", default=None)
    parser.add_argument("--experiment-name", default="controller_live_checkpoint_smoke")
    parser.add_argument("--controller-prompt", choices=("g0", "g1"), default="g0")
    return parser.parse_args()


def main() -> int:
    """Restore one checkpoint and execute the existing controlled runner once."""
    args = parse_args()
    appworld_root = Path(args.appworld_root).resolve()
    source_task_dir = Path(args.source_task_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    controller_prompt_path = (
        REPO_ROOT / "controller_agent" / "controller" / "prompts" / f"{args.controller_prompt}_system.txt"
    )
    checkpoint = load_checkpoint(source_task_dir, args.checkpoint_id)
    config = {
        "task_id": args.task_id,
        "checkpoint_id": args.checkpoint_id,
        "agent_model": "qwen3-coder-flash-2025-07-28",
        "controller_model": CONTROLLER_MODEL_NAME,
        "controller_backend": "LiveLLMBackend",
        "controller_prompt": args.controller_prompt,
        "controller_prompt_path": str(controller_prompt_path),
        "controller_prompt_sha256": hashlib.sha256(controller_prompt_path.read_bytes()).hexdigest(),
        "external_context_redaction": "credentials_email_phone",
        "private_context_strategy": "local_repl_variables",
        "candidate_audit": None if args.candidate_audit is None else str(Path(args.candidate_audit).resolve()),
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_controller_calls": args.max_controller_calls,
        "max_controller_blocks": args.max_controller_blocks,
        "max_consecutive_blocks": args.max_controller_blocks,
        "max_extra_agent_turns_after_block": args.max_extra_agent_turns_after_block,
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

    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=args.max_agent_turns,
        max_extra_agent_turns_after_block=args.max_extra_agent_turns_after_block,
    )
    agent_config = build_agent_config(agent_args, appworld_root)
    agent_config["log_lm_calls"] = False
    base_agent = Agent.from_dict(agent_config)
    logger = ExperimentLogger()
    controller = Controller(
        backend=LiveLLMBackend(
            model_name=CONTROLLER_MODEL_NAME,
            system_prompt_path=controller_prompt_path,
        )
    )
    controlled = ControlledReActAgent(
        base_agent,
        controller=controller,
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
            max_extra_agent_turns_after_block=args.max_extra_agent_turns_after_block,
            max_cost_cny=args.max_cost_cny,
        ),
    )

    restore_result: dict[str, Any]
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
                # AppWorld.load_state() closes the world's original time
                # freezer but does not install a replacement.  Recreate the
                # normal initialized-world state so evaluation and context
                # cleanup do not stop the same freezer twice.
                world._set_datetime()
                private_context = _install_private_context_variables(world, checkpoint)
                base_agent.messages = redact_external_context(
                    checkpoint["agent_messages"],
                    private_context=private_context,
                )
                base_agent.step_number = int(checkpoint["agent_turn_index"])
                if args.candidate_audit:
                    _run_cached_pre_action(
                        Path(args.candidate_audit).resolve(),
                        controlled,
                        runner,
                        world,
                        ExecutionIO,
                        private_context,
                    )
                else:
                    runner.run_restored_world(world)
                evaluator_result = evaluate_world(world)
    except Exception as exc:
        restore_result = locals().get("restore_result", {"success": False, "error": f"{type(exc).__name__}: {exc}"})
        evaluator_result = {"available": False, "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        os.chdir(previous_cwd)

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
    write_json(output_dir / "controller_response.json", response_data)
    write_json(output_dir / "evaluator_result.json", evaluator_result)
    write_json(output_dir / "restore_result.json", restore_result)
    _write_report(output_dir, config, restore_result, events, controller_events, response_data, evaluator_result, time.perf_counter() - started)
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


def _run_cached_pre_action(
    audit_path: Path,
    controlled: ControlledReActAgent,
    runner: ControlledAppWorldRunner,
    world: Any,
    execution_io_type: type[Any],
    private_context: dict[str, str],
) -> None:
    """Run a previously generated real Agent pre-action through the live Controller.

    The audit stores a genuine agent output captured before environment
    execution.  It is reused solely to make the controller checkpoint test
    deterministic; the continuation after a block still calls the real Agent.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output = _candidate_agent_output(audit)
    code = _first_python_block(output)
    if not code:
        raise ValueError("candidate audit has no executable Python code block")
    base_agent = controlled.agent
    audit_messages = audit.get("messages")
    if isinstance(audit_messages, list) and audit_messages:
        base_agent.messages = redact_external_context(
            audit_messages,
            private_context=private_context,
        )
    else:
        base_agent.messages.append({"role": "assistant", "content": output})
    base_agent.step_number = int(
        audit.get("agent_turn_index", int(getattr(base_agent, "step_number", 0)) + 1)
    )
    controlled.reset_for_task(str(getattr(world, "task_id", "")))
    controlled.runtime_state.agent_turn_count = 1
    execution_inputs = [execution_io_type(content=code, metadata={})]
    context = controlled._build_trigger_context(
        execution_inputs,
        list(audit.get("last_execution_outputs", [])),
    )
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


def _candidate_agent_output(audit: dict[str, Any]) -> str:
    """Read either an older Controller audit or a runtime pre-action record."""
    controller_input = audit.get("controller_input")
    if isinstance(controller_input, dict):
        output = controller_input.get("current_agent_output")
        if isinstance(output, str) and output.strip():
            return output
    proposed_execution = audit.get("proposed_execution")
    if isinstance(proposed_execution, str) and proposed_execution.strip():
        return f"```python\n{proposed_execution.strip()}\n```\n"
    raise ValueError("candidate audit has no Agent output or proposed execution")


def _first_python_block(output: str) -> str:
    """Extract one executable block from a captured assistant output."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def redact_external_context(
    value: Any,
    *,
    private_context: dict[str, str] | None = None,
) -> Any:
    """Remove credentials and direct profile identifiers before model calls."""
    aliased = _replace_private_context_with_variables(value, private_context or {})
    return _redact_profile_identifiers(sanitize_messages(aliased))


def _install_private_context_variables(world: Any, checkpoint: dict[str, Any]) -> dict[str, str]:
    """Keep private task fields local while exposing safe REPL variable names."""
    messages = checkpoint.get("agent_messages", [])
    task_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user" and "Task:" in str(message.get("content", ""))
    )
    private_context: dict[str, str] = {}
    email_match = re.search(
        r"personal email is:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        task_text,
        flags=re.IGNORECASE,
    )
    phone_match = re.search(
        r"phone number is\s*([0-9]{9,15})",
        task_text,
        flags=re.IGNORECASE,
    )
    user_namespace = getattr(getattr(world, "shell", None), "user_ns", None)
    if not isinstance(user_namespace, dict):
        return private_context
    if email_match:
        private_context["supervisor_email"] = email_match.group(1)
        user_namespace["supervisor_email"] = email_match.group(1)
    if phone_match:
        private_context["supervisor_phone"] = phone_match.group(1)
        user_namespace["supervisor_phone"] = phone_match.group(1)
    return private_context


def _replace_private_context_with_variables(value: Any, private_context: dict[str, str]) -> Any:
    """Replace local private values with executable variable references."""
    if isinstance(value, list):
        return [_replace_private_context_with_variables(item, private_context) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_private_context_with_variables(item, private_context)
            for key, item in value.items()
        }
    if isinstance(value, str):
        for variable_name, private_value in private_context.items():
            escaped = re.escape(private_value)
            value = re.sub(rf"(['\"]){escaped}\1", variable_name, value)
            value = re.sub(escaped, variable_name, value)
        return value
    return value


def _redact_profile_identifiers(value: Any) -> Any:
    """Recursively remove email addresses and phone-number-like values."""
    if isinstance(value, list):
        return [_redact_profile_identifiers(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_profile_identifiers(item) for key, item in value.items()}
    if isinstance(value, str):
        value = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "<redacted-email>",
            value,
        )
        return re.sub(r"(?<!\d)\d{9,15}(?!\d)", "<redacted-phone>", value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
