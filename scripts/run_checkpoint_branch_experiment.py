"""Run one real post-observation checkpoint as baseline and Controller branches."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.namespace_restore import restore_python_namespace_from_prefix
from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import DEFAULT_MODEL as CONTROLLER_MODEL, LiveLLMBackend
from controller_agent.integration.appworld_adapter import ControlledAppWorldRunner, RunnerBudget
from controller_agent.integration.controlled_react_agent import ControlledReActAgent
from controller_agent.logging.branch_artifacts import (
    AgentCallAuditor,
    PersistentExperimentLogger,
    RunIdentity,
    TokenUsageLedger,
    branch_result_row,
    build_checkpoint_context,
    build_checkpoint_prefix_rows,
    build_continuation_rows,
    build_controller_summary,
    build_full_trajectory_rows,
    build_run_manifest,
    combined_usage,
    comparison_row,
    controller_result_row,
    copy_raw_logs,
    extract_final_answer,
    normalize_evaluator_result,
    read_jsonl,
    sha256_file,
    usage_summary,
    write_json,
    write_jsonl,
)
from controller_agent.state.schemas import ControllerConfig
from scripts.run_checkpoint_restore_continue import copy_state_directory, ensure_appworld_agents_package, load_checkpoint
from scripts.run_controller_api_smoke import (
    API_ENDPOINT,
    MODEL_NAME as AGENT_MODEL,
    appworld_task_output_dir,
    build_agent_config,
    evaluate_world,
    load_dotenv,
)
from scripts.run_live_controller_checkpoint_smoke import _install_private_context_variables, redact_external_context


BRANCHES = ("baseline", "controller")
ALL_BRANCHES = ("baseline", "controller", "oracle")


def parse_args() -> argparse.Namespace:
    """Parse the single-checkpoint experiment configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--branch", choices=BRANCHES, default=None)
    parser.add_argument("--experiment-id", default="checkpoint_branch_v1")
    parser.add_argument("--task-id", default="07b42fd_1")
    parser.add_argument("--checkpoint-id", default="07b42fd_1_post_0008")
    parser.add_argument(
        "--source-task-dir",
        default=str(
            REPO_ROOT.parent
            / "appworld"
            / "experiments"
            / "outputs"
            / "qwen3_coder_flash_train_part_4_checkpoint_collection"
            / "tasks"
            / "07b42fd_1"
        ),
    )
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--controller-prompt", choices=("g0", "g1"), default="g1")
    parser.add_argument("--max-agent-turns", type=int, default=3)
    parser.add_argument("--max-environment-executions", type=int, default=3)
    parser.add_argument("--max-controller-calls", type=int, default=1)
    parser.add_argument("--max-controller-blocks", type=int, default=1)
    parser.add_argument("--max-extra-agent-turns-after-block", type=int, default=3)
    parser.add_argument("--max-cost-cny", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    """Run exactly one checkpoint pair or one isolated branch worker."""
    args = parse_args()
    validate_label(args.experiment_id, "experiment_id")
    source_task_dir = Path(args.source_task_dir).resolve()
    checkpoint = load_checkpoint(source_task_dir, args.checkpoint_id)
    validate_checkpoint(checkpoint, args.task_id, args.checkpoint_id)
    experiment_root = Path(args.results_root).resolve() / args.experiment_id
    prompt_path = prompt_file(args.controller_prompt)
    manifest = experiment_manifest(args, experiment_root, source_task_dir, prompt_path)
    experiment_root.mkdir(parents=True, exist_ok=True)
    if not args.worker or not (experiment_root / "manifest.json").exists():
        write_json(experiment_root / "manifest.json", manifest)
    if not args.execute:
        return 0
    if args.worker:
        if args.branch is None:
            raise ValueError("--worker requires --branch")
        return run_branch(args, checkpoint, experiment_root, source_task_dir, prompt_path)
    failures = run_workers(args)
    summarize_experiment(experiment_root, args.task_id, args.checkpoint_id)
    return 1 if failures else 0


def run_workers(args: argparse.Namespace) -> list[str]:
    """Isolate baseline and Controller AppWorld lifecycles in separate processes."""
    failures: list[str] = []
    for branch in BRANCHES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--execute",
            "--worker",
            "--branch",
            branch,
            "--experiment-id",
            args.experiment_id,
            "--task-id",
            args.task_id,
            "--checkpoint-id",
            args.checkpoint_id,
            "--source-task-dir",
            str(Path(args.source_task_dir).resolve()),
            "--results-root",
            str(Path(args.results_root).resolve()),
            "--appworld-root",
            str(Path(args.appworld_root).resolve()),
            "--controller-prompt",
            args.controller_prompt,
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
            "--seed",
            str(args.seed),
        ]
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            failures.append(f"{branch}: exit {completed.returncode}")
    return failures


def run_branch(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    experiment_root: Path,
    source_task_dir: Path,
    prompt_path: Path,
) -> int:
    """Restore and continue one branch from the selected real checkpoint."""
    branch = str(args.branch)
    run_id = f"{args.task_id}__{args.checkpoint_id}__{branch}"
    identity = RunIdentity(args.experiment_id, run_id, args.task_id, args.checkpoint_id, branch)
    run_dir = experiment_root / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = source_task_dir / "checkpoints" / "post_observation" / f"{args.checkpoint_id}.json"
    prompt_hash = sha256_file(prompt_path)
    token_ledger = TokenUsageLedger(run_dir / "token_usage.jsonl", identity)
    logger = PersistentExperimentLogger(
        path=run_dir / "controller_calls.jsonl",
        runtime_events_path=run_dir / "controller_runtime_events.jsonl",
        identity=identity,
        token_ledger=token_ledger,
        prompt_hash=prompt_hash,
    )
    appworld_root = Path(args.appworld_root).resolve()
    appworld_experiment_name = safe_appworld_experiment_name(args.experiment_id, branch)
    appworld_task_dir = appworld_task_output_dir(appworld_root, appworld_experiment_name, args.task_id)
    budget = budget_config(args)
    lifecycle_error: str | None = None
    evaluator_raw: dict[str, Any] = {"available": False, "error": "run_not_started"}
    restore_result: dict[str, Any] = {"success": False}
    runner: ControlledAppWorldRunner | None = None
    auditor: AgentCallAuditor | None = None
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ.setdefault("MODEL_SERVER_URL", API_ENDPOINT)
    load_dotenv(appworld_root / "experiments" / ".env")
    prepare_appworld_imports(appworld_root)
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401

    agent_args = argparse.Namespace(
        max_cost_cny=args.max_cost_cny,
        max_agent_turns=args.max_agent_turns,
        max_extra_agent_turns_after_block=args.max_extra_agent_turns_after_block,
    )
    agent_config = build_agent_config(agent_args, appworld_root)
    agent_config["log_lm_calls"] = True
    base_agent = Agent.from_dict(agent_config)
    auditor = AgentCallAuditor(agent=base_agent, identity=identity, ledger=token_ledger, model=AGENT_MODEL)
    auditor.install()
    controller = Controller(
        backend=(
            LiveLLMBackend(model_name=CONTROLLER_MODEL, system_prompt_path=prompt_path)
            if branch == "controller"
            else None
        )
    )
    controlled = ControlledReActAgent(
        base_agent,
        controller=controller,
        config=ControllerConfig(
            enabled=branch == "controller",
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

    previous_cwd = Path.cwd()
    os.chdir(appworld_root)
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=appworld_experiment_name,
            random_seed=args.seed,
            raise_on_extra_parameters=True,
        ):
            with AppWorld(task_id=args.task_id) as world:
                base_agent.initialize(world)
                copy_state_directory(source_task_dir, appworld_task_dir, checkpoint)
                namespace_result = restore_python_namespace_from_prefix(world, checkpoint)
                restore_result = namespace_result.to_dict()
                if not namespace_result.success:
                    raise RuntimeError(namespace_result.error or "namespace restore failed")
                world.load_state(str(checkpoint["appworld_state_id"]))
                world._set_datetime()
                private_context = _install_private_context_variables(world, checkpoint)
                base_agent.messages = redact_external_context(
                    checkpoint["agent_messages"],
                    private_context=private_context,
                )
                base_agent.step_number = int(checkpoint["agent_turn_index"])
                runner.run_restored_world(world)
                evaluator_raw = evaluate_world(world)
    except Exception as exc:
        lifecycle_error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous_cwd)

    runner_events = [] if runner is None else runner.turn_events
    checkpoint_executions = int(checkpoint.get("environment_execution_index") or 0)
    for event in runner_events:
        if event.get("environment_execution_index") is not None:
            event["environment_execution_index"] = checkpoint_executions + int(event["environment_execution_index"])
    continuation = build_continuation_rows(identity, runner_events, [] if auditor is None else auditor.calls)
    if continuation and continuation[-1].get("termination_reason") is None:
        continuation[-1]["termination_reason"] = lifecycle_error or (None if runner is None else runner.stop_reason)
    write_jsonl(run_dir / "continuation.jsonl", continuation)
    checkpoint_context = build_checkpoint_context(identity, checkpoint, checkpoint_path)
    write_json(run_dir / "checkpoint_context.json", checkpoint_context)
    prefix = build_checkpoint_prefix_rows(identity, source_task_dir, checkpoint)
    trajectory = build_full_trajectory_rows(prefix, continuation)
    write_jsonl(run_dir / "trajectory.jsonl", trajectory)

    evaluator_version = read_text(appworld_task_dir / "evaluation" / "version.txt")
    report_path = appworld_task_dir / "evaluation" / "report.md"
    evaluator = normalize_evaluator_result(
        identity=identity,
        raw_result=evaluator_raw,
        evaluator_version=evaluator_version,
        report_path=report_path if report_path.exists() else None,
    )
    write_json(run_dir / "evaluator_result.json", evaluator)

    raw_log_hashes: dict[str, str] = {}
    try:
        raw_log_hashes = copy_raw_logs(appworld_task_dir, run_dir / "raw_logs")
    except Exception as exc:
        if lifecycle_error is None:
            lifecycle_error = f"{type(exc).__name__}: {exc}"

    usage_rows = token_ledger.read_all()
    agent_tokens = usage_summary(usage_rows, "agent")
    controller_tokens = usage_summary(usage_rows, "controller")
    combined_tokens = combined_usage(agent_tokens, controller_tokens)
    controller_calls = read_jsonl(run_dir / "controller_calls.jsonl")
    controller_runtime_events = read_jsonl(run_dir / "controller_runtime_events.jsonl")
    controller_summary = build_controller_summary(
        identity=identity,
        enabled=branch == "controller",
        model=CONTROLLER_MODEL,
        prompt_version=args.controller_prompt,
        prompt_hash=prompt_hash,
        controller_calls=controller_calls,
        runtime_events=controller_runtime_events,
        token_rows=usage_rows,
    )
    write_json(run_dir / "controller_summary.json", controller_summary)
    success = evaluator.get("success")
    result = identity.to_dict() | {
        "final_answer": extract_final_answer(continuation),
        "success": success,
        "score": evaluator.get("score"),
        "passed": evaluator.get("passed"),
        "total": evaluator.get("total"),
        "termination_reason": lifecycle_error or (None if runner is None else runner.stop_reason),
        "agent_turns": len([] if auditor is None else auditor.calls),
        "environment_executions": sum(1 for event in continuation if event["world_batch_execute_called"]),
        "controller_calls": len(controller_calls),
        "controller_blocks": sum(1 for call in controller_calls if call.get("blocked")),
        "fail_open_count": controller_summary["fail_open_count"],
        "agent_tokens": agent_tokens,
        "controller_tokens": controller_tokens,
        "combined_tokens": combined_tokens,
        "trajectory_path": "trajectory.jsonl",
        "checkpoint_context_path": "checkpoint_context.json",
        "trajectory_prefix_events": len(prefix),
        "continuation_events": len(continuation),
        "checkpoint_boundary_agent_turn_index": int(checkpoint["agent_turn_index"]),
        "token_accounting_scope": "branch_continuation_only",
        "controller_summary_path": "controller_summary.json",
        "evaluator_result_path": "evaluator_result.json",
        "raw_logs_path": "raw_logs",
        "lifecycle_error": lifecycle_error,
        "checkpoint_restore": restore_result,
    }
    write_json(run_dir / "result.json", result)

    manifest = build_run_manifest(
        identity=identity,
        checkpoint_path=checkpoint_path,
        agent_model=AGENT_MODEL,
        controller_model=CONTROLLER_MODEL,
        controller_enabled=branch == "controller",
        prompt_version=args.controller_prompt,
        prompt_hash=prompt_hash,
        seed=args.seed,
        budget=budget,
        appworld_code_version=read_text(appworld_task_dir / "version" / "code.txt"),
        appworld_data_version=read_text(appworld_task_dir / "version" / "data.txt"),
        raw_log_hashes=raw_log_hashes,
        trajectory_summary={
            "path": "trajectory.jsonl",
            "prefix_event_count": len(prefix),
            "continuation_event_count": len(continuation),
            "checkpoint_boundary_agent_turn_index": int(checkpoint["agent_turn_index"]),
            "token_accounting_scope": "branch_continuation_only",
            "checkpoint_context_path": "checkpoint_context.json",
            "checkpoint_cumulative_agent_tokens": checkpoint.get("cumulative_agent_tokens"),
            "checkpoint_cumulative_controller_tokens": checkpoint.get("cumulative_controller_tokens"),
            "checkpoint_cumulative_cost": checkpoint.get("cumulative_cost"),
        },
    )
    write_json(run_dir / "manifest.json", manifest)
    return 0 if lifecycle_error is None else 1


def summarize_experiment(experiment_root: Path, task_id: str, checkpoint_id: str) -> None:
    """Generate the two root JSONL summaries from completed branch results."""
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        run_id = f"{task_id}__{checkpoint_id}__{branch}"
        result_path = experiment_root / "runs" / run_id / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results[branch] = result
        rows.append(branch_result_row(result, result_path.relative_to(experiment_root)))
        controller_summary_path = result_path.parent / "controller_summary.json"
        if controller_summary_path.exists():
            controller_summary = json.loads(controller_summary_path.read_text(encoding="utf-8"))
            controller_rows.append(
                controller_result_row(
                    controller_summary,
                    controller_summary_path.relative_to(experiment_root),
                )
            )
    write_jsonl(experiment_root / "branch_results.jsonl", rows)
    write_jsonl(experiment_root / "controller_results.jsonl", controller_rows)
    comparisons: list[dict[str, Any]] = []
    if all(branch in results for branch in BRANCHES):
        identity = RunIdentity(args_experiment_id(results), "comparison", task_id, checkpoint_id, "comparison")
        comparisons.append(comparison_row(identity, results["baseline"], results["controller"]))
    write_jsonl(experiment_root / "comparison_summary.jsonl", comparisons)


def experiment_manifest(
    args: argparse.Namespace,
    experiment_root: Path,
    source_task_dir: Path,
    prompt_path: Path,
) -> dict[str, Any]:
    """Build the root manifest for exactly one checkpoint and two branches."""
    runs = [
        {
            "run_id": f"{args.task_id}__{args.checkpoint_id}__{branch}",
            "branch": branch,
            "path": f"runs/{args.task_id}__{args.checkpoint_id}__{branch}",
        }
        for branch in BRANCHES
    ]
    return {
        "experiment_id": args.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(experiment_root),
        "configuration": {
            "seed": args.seed,
            "budget": budget_config(args),
            "source_task_dir": str(source_task_dir),
            "checkpoint_phase": "post_observation",
        },
        "tasks": [{"task_id": args.task_id, "checkpoint_id": args.checkpoint_id}],
        "agent_model": AGENT_MODEL,
        "controller_model": CONTROLLER_MODEL,
        "prompt": {
            "version": args.controller_prompt,
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
        },
        "trigger": {"enabled": ["before_finish"]},
        "branches": list(ALL_BRANCHES),
        "executed_branches": list(BRANCHES),
        "artifacts": {
            "branch_results": "branch_results.jsonl",
            "comparison_summary": "comparison_summary.jsonl",
            "controller_results": "controller_results.jsonl",
        },
        "runs": runs,
    }


def budget_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the shared branch budgets."""
    return {
        "max_agent_turns": args.max_agent_turns,
        "max_environment_executions": args.max_environment_executions,
        "max_controller_calls": args.max_controller_calls,
        "max_controller_blocks": args.max_controller_blocks,
        "max_extra_agent_turns_after_block": args.max_extra_agent_turns_after_block,
        "max_cost_cny": args.max_cost_cny,
    }


def prepare_appworld_imports(appworld_root: Path) -> None:
    """Expose AppWorld and its experiment Agent package."""
    for path in (appworld_root, appworld_root / "src", appworld_root / "experiments" / "code"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    ensure_appworld_agents_package(appworld_root)


def validate_checkpoint(checkpoint: dict[str, Any], task_id: str, checkpoint_id: str) -> None:
    """Reject initial-state or mismatched checkpoint inputs."""
    if checkpoint.get("task_id") != task_id:
        raise ValueError("checkpoint task_id does not match")
    if checkpoint.get("checkpoint_id") != checkpoint_id:
        raise ValueError("checkpoint_id does not match")
    if checkpoint.get("phase") != "post_observation":
        raise ValueError("experiment requires a real post_observation checkpoint")
    if not checkpoint.get("appworld_state_id"):
        raise ValueError("checkpoint is missing appworld_state_id")


def prompt_file(version: str) -> Path:
    """Return the fixed existing Controller prompt path."""
    return REPO_ROOT / "controller_agent" / "controller" / "prompts" / f"{version}_system.txt"


def safe_appworld_experiment_name(experiment_id: str, branch: str) -> str:
    """Build a deterministic AppWorld output namespace for one branch."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"checkpoint_branch_{experiment_id}_{branch}")


def validate_label(value: str, name: str) -> None:
    """Require safe, stable directory labels."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must contain only letters, digits, '.', '_' or '-'")


def read_text(path: Path) -> str | None:
    """Read optional version metadata."""
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def args_experiment_id(results: dict[str, dict[str, Any]]) -> str:
    """Read and cross-check the experiment id from both branches."""
    values = {str(result["experiment_id"]) for result in results.values()}
    if len(values) != 1:
        raise ValueError(f"branch experiment_id mismatch: {sorted(values)}")
    return values.pop()


if __name__ == "__main__":
    raise SystemExit(main())
