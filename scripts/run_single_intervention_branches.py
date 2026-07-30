"""Run one real Controller decision at each scanned failed-Baseline checkpoint.

The script deliberately stages each prefix from the original AppWorld logs in
an isolated AppWorld output.  Sanitized durable logs are never replayed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.artifact_compaction import compact_completed_branch
from controller_agent.branching.single_intervention import branch_directory, summarize_single_interventions
from controller_agent.evaluation.official_evaluator import run_official_evaluation
from scripts.collect_oracle_instruction_branches import _normalize_evaluator
from scripts.run_checkpoint_restore_continue import ensure_appworld_agents_package
from scripts.run_controller_api_smoke import API_ENDPOINT, MODEL_NAME, load_dotenv


DEEPSEEK_CONTROLLER = "deepseek-v4-pro"


def parse_args() -> argparse.Namespace:
    """Parse a bounded single-intervention batch without implicit execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--agent-model", default=MODEL_NAME)
    parser.add_argument("--controller-model", default=DEEPSEEK_CONTROLLER)
    parser.add_argument("--controller-base-url", default=API_ENDPOINT)
    parser.add_argument("--controller-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--appworld-experiment-prefix",
        default="single_intervention",
        help="Unique AppWorld output prefix for this isolated branch batch.",
    )
    parser.add_argument(
        "--controller-system-prompt",
        default=None,
        help="Optional system-prompt file forwarded to the live Controller backend.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--mode",
        choices=("controller", "no_guidance", "continuous_controller"),
        default="controller",
        help="Run one Controller intervention or a clean replay from the same checkpoint.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Materialize independent prefixes and run one Controller call per row."""
    args = parse_args()
    if args.execute and importlib.util.find_spec("munch") is None:
        # AppWorld's own virtual environment owns runtime dependencies such as
        # munch. Re-exec before opening any checkpoint or calling a model.
        appworld_python = Path(args.appworld_root).resolve() / ".venv" / "Scripts" / "python.exe"
        if not appworld_python.is_file():
            raise RuntimeError(f"AppWorld Python runtime not found: {appworld_python}")
        completed = subprocess.run([str(appworld_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=REPO_ROOT, check=False)
        return completed.returncode
    root = Path(args.experiment_root).resolve()
    candidates = _read_jsonl(Path(args.candidates))
    if args.limit is not None:
        candidates = candidates[: max(args.limit, 0)]
    summary: dict[str, Any] = {"planned": len(candidates), "completed": 0, "failed": 0, "skipped": 0, "failures": []}
    if not args.execute:
        plan_name = "single_intervention_plan.json" if args.mode == "controller" else "no_guidance_replay_plan.json"
        _write_json(root / plan_name, summary | {"execute": False})
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    for candidate in candidates:
        try:
            status = _run_candidate(args, root, candidate)
        except Exception as exc:
            status = "failed"
            summary["failures"].append({"candidate": candidate, "error": f"{type(exc).__name__}: {exc}"})
        summary[status] += 1
    aggregate = (
        summarize_single_interventions(root)
        if args.mode == "controller"
        else _summarize_mode(root, args.mode)
    )
    scan_counts = Counter(str(row.get("trigger_type") or "unknown") for row in candidates)
    report = summary | aggregate | {
        "candidate_task_count": len({str(row.get("task_id")) for row in candidates}),
        "scanned_checkpoint_count": len(candidates),
        "scanned_trigger_counts": dict(sorted(scan_counts.items())),
    }
    summary_name = {
        "controller": "single_intervention_summary.json",
        "no_guidance": "no_guidance_replay_summary.json",
        "continuous_controller": "continuous_controller_summary.json",
    }[args.mode]
    _write_json(root / summary_name, report)
    if args.mode == "controller":
        _write_final_outcomes(root)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


def _run_candidate(args: argparse.Namespace, root: Path, candidate: dict[str, Any]) -> str:
    """Execute one isolated checkpoint branch, never modifying its baseline."""
    required = ("task_id", "checkpoint_step", "checkpoint_id", "trigger_type", "proposed_execution", "baseline_run_path", "baseline_event_index")
    missing = [key for key in required if candidate.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Candidate missing required fields: {', '.join(missing)}")
    task_id = str(candidate["task_id"])
    destination = _branch_destination(root / "tasks" / task_id, candidate, args.mode)
    if destination.exists():
        if args.resume and _completed_result(destination):
            return "skipped"
        # Only this script creates names beginning ``single_intervention__``;
        # an incomplete attempt has no valid result and is safe to rebuild.
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    stage: StageResult | None = None
    try:
        stage = _materialize_prefix_checkpoint(args, candidate)
        _write_json(destination / "checkpoint_info.json", _public_checkpoint_info(candidate, stage))
        completed = subprocess.run(
            _branch_command(args, candidate, stage, destination),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        official = run_official_evaluation(Path(args.appworld_root).resolve(), stage.branch_experiment_name, task_id)
        evaluator = _normalize_evaluator(task_id, official)
        _write_json(destination / "evaluator_result.json", evaluator)
        _write_result(
            destination,
            candidate,
            stage,
            completed,
            evaluator,
            controller_calls_expected=1 if args.mode == "controller" else 0 if args.mode == "no_guidance" else 4,
        )
        _write_json(destination / "manifest.json", {
            "task_id": task_id,
            "checkpoint_id": candidate["checkpoint_id"],
            "checkpoint_step": candidate["checkpoint_step"],
            "trigger_type": candidate["trigger_type"],
            "branch": args.mode,
            "agent_model": args.agent_model,
            "controller_model": args.controller_model,
            "controller_system_prompt": args.controller_system_prompt,
            "controller_calls_per_checkpoint": 1 if args.mode == "controller" else 0 if args.mode == "no_guidance" else 4,
            "trigger_profile": "all" if args.mode == "continuous_controller" else "forced_checkpoint_only" if args.mode == "controller" else "disabled",
            "source_baseline": candidate["baseline_run_path"],
            "source_appworld_task": candidate.get("source_appworld_task_path"),
            "appworld_experiment": stage.branch_experiment_name,
        })
        _write_token_usage(destination)
        _strip_money_fields(destination)
        compact_completed_branch(destination)
        return "completed" if completed.returncode == 0 else "failed"
    finally:
        if stage is not None and stage.stage_task_dir.exists():
            shutil.rmtree(stage.stage_task_dir.parent, ignore_errors=False)


class StageResult:
    """Private metadata needed only while invoking the checkpoint runner."""

    def __init__(self, source_task_dir: Path, audit_path: Path, branch_experiment_name: str, state_id: str, replayed_steps: int) -> None:
        self.source_task_dir = source_task_dir
        self.audit_path = audit_path
        self.branch_experiment_name = branch_experiment_name
        self.state_id = state_id
        self.replayed_steps = replayed_steps

    @property
    def stage_task_dir(self) -> Path:
        return self.source_task_dir


def _materialize_prefix_checkpoint(args: argparse.Namespace, candidate: dict[str, Any]) -> StageResult:
    """Replay a raw Baseline prefix and write a temporary compatible checkpoint."""
    appworld_root = Path(args.appworld_root).resolve()
    source_task = Path(str(candidate.get("source_appworld_task_path") or ""))
    if not (source_task / "logs" / "logger.jsonl").is_file():
        raise FileNotFoundError("Candidate has no original AppWorld logger; sanitized durable logs are not replayable.")
    task_id = str(candidate["task_id"])
    target_step = int(candidate["checkpoint_step"])
    records = _read_jsonl(source_task / "logs" / "logger.jsonl")
    pairs = _execution_pairs(records)
    target_index = int(candidate["baseline_event_index"])
    if target_index < 0 or target_index >= len(pairs):
        raise ValueError(f"No raw Agent action at baseline event {target_index} in {source_task}")
    target = pairs[target_index]
    inputs = _read_jsonl(source_task / "logs" / "lm_calls.jsonl")
    if target_index >= len(inputs):
        raise ValueError("Raw lm_calls.jsonl cannot provide the target Agent context.")
    messages = inputs[target_index].get("input", {}).get("messages")
    if not isinstance(messages, list):
        raise ValueError("Target lm call has no list-valued input.messages.")
    stage_name = f"single_intervention_stage__{task_id}__{target_step:04d}__{uuid.uuid4().hex[:8]}"
    branch_name = (
        f"{args.appworld_experiment_prefix}__{task_id}__{target_step:04d}__"
        f"{candidate['trigger_type']}"
    )
    load_dotenv(appworld_root / "experiments" / ".env")
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    sys.path.insert(0, str(appworld_root)) if str(appworld_root) not in sys.path else None
    ensure_appworld_agents_package(appworld_root)
    from appworld import AppWorld

    old_cwd = Path.cwd()
    state_id = f"{candidate['checkpoint_id']}_state"
    try:
        os.chdir(appworld_root)
        with AppWorld.initializer(update_defaults=True, experiment_name=stage_name, random_seed=args.seed, raise_on_extra_parameters=True):
            world = AppWorld(task_id=task_id)
            try:
                prefix = pairs[:target_index]
                for pair in prefix:
                    actual = world.batch_execute([pair["code"]])[0]
                    if _normalize(actual) != _normalize(pair["observation"]):
                        raise RuntimeError(
                            f"replay_diverged at step {pair['step_number']}; "
                            f"expected={_normalize(pair['observation'])[:240]!r}; "
                            f"actual={_normalize(actual)[:240]!r}"
                        )
                world.save_state(state_id)
            finally:
                try:
                    world.close()
                except AttributeError as exc:
                    if "_freeze_time" not in str(exc) or "fake_names" not in str(exc):
                        raise
    finally:
        os.chdir(old_cwd)
    stage_task_dir = appworld_root / "experiments" / "outputs" / stage_name / "tasks" / task_id
    checkpoint_dir = stage_task_dir / "checkpoints"
    record = {
        "checkpoint_id": candidate["checkpoint_id"],
        "task_id": task_id,
        "agent_turn_index": target_step - 1,
        "environment_execution_index": target_index,
        "appworld_state_id": state_id,
        "agent_messages": _restore_messages(messages),
        "last_execution_outputs": [] if target_index == 0 else [pairs[target_index - 1]["observation"]],
        "executed_code_prefix": [pair["code"] for pair in pairs[:target_index]],
        "namespace_restore_mode": "replay_required",
        "phase": "pre_action",
    }
    _write_json(checkpoint_dir / "post_observation" / f"{candidate['checkpoint_id']}.json", record)
    _write_json(checkpoint_dir / "manifest.json", {"task_id": task_id, "post_observation": [{"checkpoint_id": candidate["checkpoint_id"]}]})
    audit_path = stage_task_dir / "candidate_audit.json"
    _write_json(audit_path, {
        "controller_input": {
            "current_agent_output": target["agent_output"],
            # These are the exact messages passed to the Agent when it
            # produced the cached pre-action. They are required for a causal
            # Controller decision before that action is executed.
            "agent_context_messages": _restore_messages(messages),
        }
    })
    return StageResult(stage_task_dir, audit_path, branch_name, state_id, target_index)


def _branch_command(args: argparse.Namespace, candidate: dict[str, Any], stage: StageResult, destination: Path) -> list[str]:
    """Build one invocation of the existing restored Controller runner."""
    branch = "controller" if args.mode != "no_guidance" else "baseline"
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_live_controller_checkpoint_smoke.py"), "--execute",
        "--task-id", str(candidate["task_id"]), "--agent-model", args.agent_model,
        "--controller-model", args.controller_model, "--controller-base-url", args.controller_base_url,
        "--controller-api-key-env", args.controller_api_key_env, "--checkpoint-id", str(candidate["checkpoint_id"]),
        "--source-task-dir", str(stage.source_task_dir), "--candidate-audit", str(stage.audit_path),
        "--appworld-root", str(Path(args.appworld_root).resolve()), "--output-dir", str(destination),
        "--experiment-name", stage.branch_experiment_name, "--branch", branch,
        "--max-agent-turns", "10000", "--max-environment-executions", "10000",
        "--max-controller-calls", "1" if args.mode == "controller" else "0" if args.mode == "no_guidance" else "4",
        "--max-controller-blocks", "1" if args.mode == "controller" else "0" if args.mode == "no_guidance" else "4",
        "--max-extra-agent-turns-after-block", "10000",
        "--skip-inprocess-evaluate", "--log-lm-calls",
        "--prefix-logger-jsonl", str(
            Path(str(candidate["source_appworld_task_path"])) / "logs" / "logger.jsonl"
        ),
        "--prefix-logger-log", str(
            Path(str(candidate["source_appworld_task_path"])) / "logs" / "logger.log"
        ),
    ]
    if args.mode in {"controller", "continuous_controller"}:
        command.extend([
            "--force-controller-at-checkpoint",
            "--forced-trigger-reason", str(candidate["trigger_type"]),
        ])
        if args.mode == "controller":
            command.append("--controller-once")
        else:
            command.extend(["--trigger-profile", "all"])
    else:
        command.append("--disable-controller")
    return command + (
        ["--controller-system-prompt", str(Path(args.controller_system_prompt).resolve())]
        if args.controller_system_prompt
        else []
    )


def _execution_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair raw Agent code with its following raw environment observation."""
    pairs: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for record in records:
        if record.get("role") == "agent":
            content = str(record.get("content") or "")
            code = _first_python_block(content)
            pending = {"step_number": record.get("step_number"), "agent_output": content, "code": code}
        elif record.get("role") == "environment" and pending is not None:
            if not isinstance(pending.get("step_number"), int) or not pending.get("code"):
                pending = None
                continue
            pairs.append(pending | {"observation": str(record.get("content") or "")})
            pending = None
    if pending is not None and isinstance(pending.get("step_number"), int) and pending.get("code"):
        # ``complete_task`` can terminate AppWorld before a logger environment
        # entry is emitted. It is still a valid pre-action intervention point.
        pairs.append(pending | {"observation": None})
    return pairs


def _write_result(
    destination: Path,
    candidate: dict[str, Any],
    stage: StageResult,
    completed: subprocess.CompletedProcess[str],
    evaluator: dict[str, Any],
    *,
    controller_calls_expected: int,
) -> None:
    continuation = _read_jsonl(destination / "continuation.jsonl")
    calls = _read_jsonl(destination / "controller_calls.jsonl")
    final = continuation[-1] if continuation else {}
    agent_total = _agent_runtime_tokens(destination / "raw_logs" / "lm_calls.jsonl")
    controller_total = _controller_runtime_tokens(calls)
    combined_total = agent_total + controller_total if agent_total is not None and controller_total is not None else None
    _write_json(destination / "result.json", {
        "task_id": candidate["task_id"], "checkpoint_id": candidate["checkpoint_id"], "checkpoint_step": candidate["checkpoint_step"],
        "trigger_type": candidate["trigger_type"], "status": "completed" if completed.returncode == 0 else "error",
        "restore_success": bool(_read_json(destination / "restore_result.json").get("success")),
        "success": evaluator.get("success"), "evaluator": evaluator,
        "termination_reason": final.get("termination_reason"), "agent_call_count": len(continuation),
        "controller_call_count": len(calls), "controller_calls_expected": controller_calls_expected,
        "tokens": {
            "agent_total": agent_total,
            "controller_total": controller_total,
            "total": combined_total,
            "scope": "post_checkpoint_runtime_only",
            "cached_proposed_action_tokens": None,
        },
        "replay": {"replayed_prefix_steps": stage.replayed_steps, "validated": True},
        "subprocess_returncode": completed.returncode, "subprocess_stdout": completed.stdout[-4000:], "subprocess_stderr": completed.stderr[-4000:],
    })


def _agent_runtime_tokens(path: Path) -> int | None:
    """Sum only real post-checkpoint Agent API calls from raw provider logs."""
    totals: list[int] = []
    for row in _read_jsonl(path):
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        value = usage.get("total_tokens")
        if not isinstance(value, int) or value < 0:
            return None
        totals.append(value)
    return sum(totals)


def _controller_runtime_tokens(calls: list[dict[str, Any]]) -> int | None:
    """Sum direct Controller usage, preserving null when provider usage is absent."""
    if not calls:
        return 0
    totals: list[int] = []
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        value = usage.get("total_tokens")
        if not isinstance(value, int) or value < 0:
            return None
        totals.append(value)
    return sum(totals)


def _write_final_outcomes(root: Path) -> None:
    """Write the compact final metric index for first-trigger experiments."""
    records: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("tasks/*/single_intervention__*/result.json")):
        result = _read_json(path)
        calls = _read_jsonl(path.parent / "controller_calls.jsonl")
        action = calls[-1].get("action") if calls else None
        tokens = result.get("tokens") if isinstance(result.get("tokens"), dict) else {}
        records.append({
            "task_id": result.get("task_id"),
            "checkpoint_id": result.get("checkpoint_id"),
            "final_outcome": "success" if result.get("success") is True else "failure" if result.get("success") is False else "unknown",
            "controller_action": action,
            "agent_tokens": tokens.get("agent_total"),
            "controller_tokens": tokens.get("controller_total"),
            "total_tokens": tokens.get("total"),
        })
    (Path(root) / "final_outcomes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _branch_destination(task_dir: Path, candidate: dict[str, Any], mode: str) -> Path:
    """Return the distinct durable directory for one comparison branch."""
    if mode == "controller":
        return branch_directory(task_dir, _candidate_object(candidate))
    if mode == "continuous_controller":
        prefix = "continuous_controller"
    else:
        prefix = "no_guidance_replay"
    trigger = str(candidate["trigger_type"]).replace("/", "_").replace("\\", "_")
    return Path(task_dir) / f"{prefix}__step_{int(candidate['checkpoint_step']):04d}__{trigger}"


def _summarize_mode(root: Path, mode: str) -> dict[str, Any]:
    """Summarize independent replay modes without mixing their branch directories."""
    prefix = "no_guidance_replay" if mode == "no_guidance" else "continuous_controller"
    results = [_read_json(path) for path in sorted(Path(root).glob(f"tasks/*/{prefix}__*/result.json"))]
    completed = [result for result in results if result.get("status") == "completed"]
    success_count = sum(result.get("success") is True for result in completed)
    return {
        "branch_count": len(results),
        "completed_branch_count": len(completed),
        f"{mode}_success_count": success_count,
        f"{mode}_success_rate": _ratio(success_count, len(completed)),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a safe ratio for optional replay aggregates."""
    return None if denominator == 0 else numerator / denominator


def _strip_money_fields(destination: Path) -> None:
    """Remove money-related keys from generated structured branch artifacts."""
    for path in destination.glob("*.json"):
        value = _read_json(path)
        path.write_text(json.dumps(_without_money_fields(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in destination.glob("*.jsonl"):
        rows = _read_jsonl(path)
        path.write_text(
            "".join(json.dumps(_without_money_fields(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _without_money_fields(value: Any) -> Any:
    """Return a deep copy without keys that represent monetary values or limits."""
    if isinstance(value, dict):
        return {
            key: _without_money_fields(item)
            for key, item in value.items()
            if "cost" not in key.lower() and "price" not in key.lower()
        }
    if isinstance(value, list):
        return [_without_money_fields(item) for item in value]
    return value


def _write_token_usage(destination: Path) -> None:
    """Write a ledger that distinguishes the cached proposal from real calls."""
    continuation = _read_jsonl(destination / "continuation.jsonl")
    raw_calls = _read_jsonl(destination / "raw_logs" / "lm_calls.jsonl")
    calls = _read_jsonl(destination / "controller_calls.jsonl")
    records: list[dict[str, Any]] = []
    if continuation:
        first = continuation[0]
        records.append({"step": first.get("step_number"), "call_id": first.get("agent_call_id"), "source": "agent", "model": None, "input_tokens": None, "output_tokens": None, "total_tokens": None, "available": False, "record_origin": "cached_baseline_proposal"})
    for event, raw in zip(continuation[1:], raw_calls, strict=False):
        output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        input_tokens, output_tokens, total_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        available = all(isinstance(value, int) and value >= 0 for value in (input_tokens, output_tokens, total_tokens))
        records.append({"step": event.get("step_number"), "call_id": event.get("agent_call_id"), "source": "agent", "model": raw.get("input", {}).get("model") if isinstance(raw.get("input"), dict) else None, "input_tokens": input_tokens if available else None, "output_tokens": output_tokens if available else None, "total_tokens": total_tokens if available else None, "available": available, "record_origin": "runtime_direct"})
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        records.append({"step": call.get("step_number"), "call_id": call.get("controller_call_id"), "source": "controller", "model": call.get("model"), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens"), "available": all(isinstance(usage.get(key), int) and usage[key] >= 0 for key in ("input_tokens", "output_tokens", "total_tokens")), "record_origin": "runtime_direct"})
    path = destination / "token_usage.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _public_checkpoint_info(candidate: dict[str, Any], stage: StageResult) -> dict[str, Any]:
    """Persist only non-secret checkpoint provenance in the branch directory."""
    return {"task_id": candidate["task_id"], "checkpoint_id": candidate["checkpoint_id"], "checkpoint_step": candidate["checkpoint_step"], "trigger_type": candidate["trigger_type"], "trigger_metadata": candidate.get("trigger_metadata", {}), "proposed_execution": candidate["proposed_execution"], "replayed_prefix_steps": stage.replayed_steps}


def _candidate_object(candidate: dict[str, Any]) -> Any:
    """Adapt serialized candidates to the shared branch-directory helper."""
    from controller_agent.branching.single_intervention import SingleInterventionCandidate
    return SingleInterventionCandidate(**candidate)


def _completed_result(destination: Path) -> bool:
    result = _read_json(destination / "result.json")
    return result.get("status") == "completed" and result.get("restore_success") is True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                values.append(value)
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_python_block(text: str) -> str | None:
    import re
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else None


def _restore_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Copy model-visible messages while dropping provider cache directives."""
    restored: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Raw lm call contains a non-mapping message.")
        copied = copy.deepcopy(message)
        copied.pop("cache_control", None)
        restored.append(copied)
    return restored


def _normalize(value: Any) -> str:
    import re
    text = str(value).replace("\r\n", "\n").strip()
    # Authentication tokens are intentionally process-local and can change
    # during a deterministic database replay. Retain the key and compare the
    # remaining response exactly.
    text = re.sub(r'("(?:access_token|refresh_token)"\s*:\s*")[^"]*(")', r"\1<token>\2", text)
    # Some AppWorld APIs print a freshly minted JWT as a bare string or a
    # Python-dict value. It is process-local authentication noise, not a
    # replayed environment-state difference.
    return re.sub(r"eyJ[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){2}", "<token>", text)


if __name__ == "__main__":
    raise SystemExit(main())
