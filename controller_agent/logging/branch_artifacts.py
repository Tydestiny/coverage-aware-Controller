"""Auditable artifacts for one restored-checkpoint branch experiment."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from controller_agent.logging.logger import ExperimentLogger


RAW_LOG_SOURCES = {
    "logger.log": ("logs", "logger.log"),
    "logger.jsonl": ("logs", "logger.jsonl"),
    "lm_calls.jsonl": ("logs", "lm_calls.jsonl"),
    "api_calls.jsonl": ("logs", "api_calls.jsonl"),
    "environment_io.md": ("logs", "environment_io.md"),
    "usage.json": ("misc", "usage.json"),
}

CONTROLLER_ACTIONS = (
    "PROCEED",
    "VERIFY_TARGET_SET",
    "LOOKUP_API_SPEC",
    "REPLAN",
    "BLOCK_RISKY_ACTION",
)

RUN_REQUIRED_FILES = (
    "manifest.json",
    "checkpoint_context.json",
    "trajectory.jsonl",
    "continuation.jsonl",
    "controller_calls.jsonl",
    "controller_runtime_events.jsonl",
    "controller_summary.json",
    "token_usage.jsonl",
    "result.json",
    "evaluator_result.json",
    *(f"raw_logs/{name}" for name in RAW_LOG_SOURCES),
)


@dataclass(frozen=True)
class RunIdentity:
    """Stable identifiers shared by every structured run artifact."""

    experiment_id: str
    run_id: str
    task_id: str
    checkpoint_id: str
    branch: str

    def to_dict(self) -> dict[str, str]:
        """Return the common identity fields."""
        return asdict(self)

    def event_id(self, turn_index: int) -> str:
        """Return the stable event identifier for an Agent turn."""
        return f"{self.run_id}__event_{turn_index:04d}"

    def agent_call_id(self, call_index: int) -> str:
        """Return a stable Agent call identifier."""
        return f"{self.run_id}__agent_{call_index:04d}"

    def controller_call_id(self, call_index: int) -> str:
        """Return a stable Controller call identifier."""
        return f"{self.run_id}__controller_{call_index:04d}"

    def controller_runtime_event_id(self, event_index: int) -> str:
        """Return a stable identifier for a non-call Controller runtime event."""
        return f"{self.run_id}__controller_runtime_{event_index:04d}"


class JsonlLedger:
    """Append one JSON object at a time and flush it immediately."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        """Persist one row without waiting for run finalization."""
        payload = json.dumps(_jsonable(row), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        """Read all persisted rows."""
        return read_jsonl(self.path)


class TokenUsageLedger:
    """Single source of truth for per-call Agent and Controller usage."""

    def __init__(self, path: Path, identity: RunIdentity) -> None:
        self.identity = identity
        self.ledger = JsonlLedger(path)

    def record_agent(
        self,
        *,
        call_id: str,
        event_id: str,
        turn_index: int,
        model: str,
        usage: Any,
    ) -> dict[str, Any]:
        """Persist one Agent model call directly from its returned Usage."""
        input_tokens = usage_tokens(usage, "input")
        output_tokens = usage_tokens(usage, "output")
        cost = usage_cost(usage)
        row = self.identity.to_dict() | {
            "source": "agent",
            "call_id": call_id,
            "event_id": event_id,
            "turn_index": turn_index,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": add_nullable(input_tokens, output_tokens),
            "cost": cost,
            "cost_source": "appworld_usage" if cost is not None else "unavailable",
        }
        self.ledger.append(row)
        return row

    def record_controller(
        self,
        *,
        call_id: str,
        event_id: str,
        turn_index: int,
        response: Any,
    ) -> dict[str, Any]:
        """Persist one Controller model call directly from its response."""
        input_tokens = optional_int(getattr(response, "input_tokens", None))
        output_tokens = optional_int(getattr(response, "output_tokens", None))
        cost = optional_float(getattr(response, "cost_cny", None))
        row = self.identity.to_dict() | {
            "source": "controller",
            "call_id": call_id,
            "controller_call_id": call_id,
            "event_id": event_id,
            "turn_index": turn_index,
            "model": getattr(response, "model_name", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": add_nullable(input_tokens, output_tokens),
            "cost": cost,
            "cost_source": "provider" if cost is not None else "unavailable",
        }
        self.ledger.append(row)
        return row

    def read_all(self) -> list[dict[str, Any]]:
        """Return the persisted usage rows."""
        return self.ledger.read_all()


class PersistentExperimentLogger(ExperimentLogger):
    """Preserve complete runtime Controller calls without reconstructing them."""

    def __init__(
        self,
        *,
        path: Path,
        runtime_events_path: Path | None = None,
        identity: RunIdentity,
        token_ledger: TokenUsageLedger,
        prompt_hash: str,
    ) -> None:
        super().__init__()
        self.identity = identity
        self.call_ledger = JsonlLedger(path)
        self.runtime_event_ledger = JsonlLedger(
            runtime_events_path or path.with_name("controller_runtime_events.jsonl")
        )
        self.token_ledger = token_ledger
        self.prompt_hash = prompt_hash
        self._call_count = 0
        self._runtime_event_count = 0

    def log_controller_event(
        self,
        context: Any,
        trigger: Any,
        controller_input: Any,
        response: Any,
        runtime_state: Any,
        blocked: bool,
    ) -> None:
        """Persist the exact input and response received at decision time."""
        super().log_controller_event(
            context=context,
            trigger=trigger,
            controller_input=controller_input,
            response=response,
            runtime_state=runtime_state,
            blocked=blocked,
        )
        self._call_count += 1
        turn_index = int(context.step_index)
        event_id = self.identity.event_id(turn_index)
        controller_call_id = self.identity.controller_call_id(self._call_count)
        decision = response.decision
        fail_open = decision.reason == "controller_failure_fallback"
        raw_response = parse_raw_response(getattr(response, "raw_response", None))
        response_payload = {
            "decision": {
                "action": decision.action.value,
                "reason": decision.reason,
                "instruction": decision.instruction,
            },
            "model_name": response.model_name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_seconds": response.latency_seconds,
            "cost_cny": response.cost_cny,
            "raw_response": raw_response["value"],
        }
        row = self.identity.to_dict() | {
            "controller_call_id": controller_call_id,
            "event_id": event_id,
            "trigger": {
                "reason": trigger.reason,
                "metadata": copy.deepcopy(trigger.metadata),
            },
            "trigger_reason": trigger.reason,
            "trigger_metadata": copy.deepcopy(trigger.metadata),
            "controller_input": _jsonable(controller_input),
            "controller_input_provenance": {
                "available": True,
                "reconstructed": False,
                "source": "runtime_state_builder",
            },
            "controller_response": response_payload,
            "provider_raw_response": raw_response,
            "action": decision.action.value,
            "reason": decision.reason,
            "guidance": decision.instruction,
            "blocked": blocked,
            "fail_open": fail_open,
            "model": response.model_name,
            "prompt_hash": self.prompt_hash,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": add_nullable(response.input_tokens, response.output_tokens),
        }
        self.call_ledger.append(row)
        self.token_ledger.record_controller(
            call_id=controller_call_id,
            event_id=event_id,
            turn_index=turn_index,
            response=response,
        )

    def log_runtime_event(
        self,
        *,
        task_id: str,
        step_index: int,
        event_type: str,
        reason: str,
        trigger: Any,
        error: str | None,
        runtime_state: Any,
    ) -> None:
        """Persist fail-open and budget events when they occur."""
        super().log_runtime_event(
            task_id=task_id,
            step_index=step_index,
            event_type=event_type,
            reason=reason,
            trigger=trigger,
            error=error,
            runtime_state=runtime_state,
        )
        self._runtime_event_count += 1
        row = self.identity.to_dict() | {
            "controller_runtime_event_id": self.identity.controller_runtime_event_id(self._runtime_event_count),
            "event_id": self.identity.event_id(int(step_index)),
            "event_type": event_type,
            "reason": reason,
            "trigger": (
                None
                if trigger is None
                else {
                    "reason": trigger.reason,
                    "metadata": copy.deepcopy(trigger.metadata),
                }
            ),
            "error": error,
            "runtime_state": _jsonable(runtime_state),
        }
        self.runtime_event_ledger.append(row)


class AgentCallAuditor:
    """Patch one concrete Agent instance to log every model call at return."""

    def __init__(
        self,
        *,
        agent: Any,
        identity: RunIdentity,
        ledger: TokenUsageLedger,
        model: str,
    ) -> None:
        self.agent = agent
        self.identity = identity
        self.ledger = ledger
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self._call_count = 0
        self._original = agent.next_execution_inputs_usage_and_status

    def install(self) -> None:
        """Install the instance-local audited call wrapper."""
        self.agent.next_execution_inputs_usage_and_status = self._audited_call

    def _audited_call(self, last_execution_outputs: Any) -> Any:
        self._call_count += 1
        turn_index = int(getattr(self.agent, "step_number", 0))
        event_id = self.identity.event_id(turn_index)
        call_id = self.identity.agent_call_id(self._call_count)
        try:
            result = self._original(last_execution_outputs)
        except Exception:
            row = self.ledger.record_agent(
                call_id=call_id,
                event_id=event_id,
                turn_index=turn_index,
                model=self.model,
                usage=None,
            )
            self.calls.append(row | {"agent_output": None})
            raise
        usage = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        row = self.ledger.record_agent(
            call_id=call_id,
            event_id=event_id,
            turn_index=turn_index,
            model=self.model,
            usage=usage,
        )
        self.calls.append(row | {"agent_output": latest_assistant_output(self.agent)})
        return result


def build_continuation_rows(
    identity: RunIdentity,
    runner_events: list[dict[str, Any]],
    agent_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join runner outcomes to the directly observed Agent calls."""
    rows: list[dict[str, Any]] = []
    for sequence_index, call in enumerate(agent_calls):
        event = runner_events[sequence_index] if sequence_index < len(runner_events) else {}
        observation = event.get("execution_outputs")
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        trigger_reason = event.get("trigger_reason")
        if trigger_reason is None and event.get("trigger_detected"):
            trigger_reason = "before_finish"
        termination_reason = event.get("termination_reason")
        if termination_reason is None and sequence_index == len(agent_calls) - 1:
            termination_reason = event.get("stop_reason")
        rows.append(
            identity.to_dict()
            | {
                "event_id": call["event_id"],
                "sequence_index": sequence_index,
                "agent_turn_index": call["turn_index"],
                "environment_execution_index": event.get("environment_execution_index"),
                "agent_output": call.get("agent_output"),
                "generated_code": event.get("generated_code", ""),
                "environment_observation": observation,
                "trigger": {
                    "detected": bool(event.get("trigger_detected")),
                    "reason": trigger_reason,
                    "action": event.get("controller_action"),
                },
                "execution_allowed": event.get("execution_allowed"),
                "guidance_injected": bool(event.get("guidance_injected")),
                "world_batch_execute_called": bool(event.get("world_batch_execute_called")),
                "task_completed": bool(event.get("task_completed")),
                "termination_reason": termination_reason,
            }
        )
    return rows


def build_checkpoint_prefix_rows(
    identity: RunIdentity,
    source_task_dir: Path,
    selected_checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the immutable trajectory prefix ending at a checkpoint.

    Runtime turn events are preferred because they also retain blocked turns. The
    post-observation checkpoints supply exact observations and Agent messages.
    """
    selected_turn = int(selected_checkpoint["agent_turn_index"])
    checkpoint_dir = source_task_dir / "checkpoints" / "post_observation"
    checkpoints: dict[int, dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("task_id") != identity.task_id or checkpoint.get("phase") != "post_observation":
            continue
        turn_index = int(checkpoint["agent_turn_index"])
        if turn_index <= selected_turn:
            checkpoints[turn_index] = checkpoint
    if selected_turn not in checkpoints:
        checkpoints[selected_turn] = copy.deepcopy(selected_checkpoint)

    source_events_path = source_task_dir / "collection" / "turn_events.jsonl"
    source_events = [
        event
        for event in read_jsonl(source_events_path)
        if int(event.get("agent_turn_index") or 0) <= selected_turn
    ]
    if not source_events:
        source_events = [
            {
                "agent_turn_index": checkpoint["agent_turn_index"],
                "environment_execution_index": checkpoint.get("environment_execution_index"),
                "generated_code": checkpoint.get("executed_code", ""),
                "execution_allowed": True,
                "world_batch_execute_called": True,
            }
            for checkpoint in sorted(checkpoints.values(), key=lambda value: int(value["agent_turn_index"]))
        ]

    rows: list[dict[str, Any]] = []
    for sequence_index, event in enumerate(source_events):
        turn_index = int(event["agent_turn_index"])
        checkpoint = checkpoints.get(turn_index)
        observation = None if checkpoint is None else checkpoint.get("last_execution_outputs")
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        trigger_reason = event.get("trigger_reason")
        if trigger_reason is None and event.get("trigger_detected"):
            trigger_reason = "before_finish"
        rows.append(
            identity.to_dict()
            | {
                "event_id": identity.event_id(turn_index),
                "sequence_index": sequence_index,
                "segment_sequence_index": sequence_index,
                "trajectory_segment": "checkpoint_prefix",
                "source_checkpoint_id": None if checkpoint is None else checkpoint.get("checkpoint_id"),
                "agent_turn_index": turn_index,
                "environment_execution_index": event.get("environment_execution_index"),
                "agent_output": (
                    None if checkpoint is None else latest_assistant_output_from_messages(checkpoint.get("agent_messages"))
                ),
                "generated_code": event.get(
                    "generated_code",
                    "" if checkpoint is None else checkpoint.get("executed_code", ""),
                ),
                "environment_observation": observation,
                "trigger": {
                    "detected": bool(event.get("trigger_detected")),
                    "reason": trigger_reason,
                    "action": event.get("controller_action"),
                },
                "execution_allowed": event.get("execution_allowed"),
                "guidance_injected": bool(event.get("guidance_injected")),
                "world_batch_execute_called": bool(event.get("world_batch_execute_called")),
                "task_completed": bool(event.get("task_completed")),
                "termination_reason": event.get("termination_reason"),
            }
        )
    if not rows or rows[-1]["agent_turn_index"] != selected_turn:
        raise ValueError(f"trajectory prefix does not end at checkpoint turn {selected_turn}")
    return rows


def build_checkpoint_context(
    identity: RunIdentity,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Preserve the exact shared context needed to interpret the prefix."""
    return identity.to_dict() | {
        "checkpoint_source_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "agent_turn_index": checkpoint.get("agent_turn_index"),
        "environment_execution_index": checkpoint.get("environment_execution_index"),
        "appworld_state_id": checkpoint.get("appworld_state_id"),
        "phase": checkpoint.get("phase"),
        "agent_messages": copy.deepcopy(checkpoint.get("agent_messages", [])),
        "executed_code_prefix": copy.deepcopy(checkpoint.get("executed_code_prefix", [])),
        "last_execution_outputs": copy.deepcopy(checkpoint.get("last_execution_outputs", [])),
        "cumulative_agent_tokens": copy.deepcopy(checkpoint.get("cumulative_agent_tokens")),
        "cumulative_controller_tokens": copy.deepcopy(checkpoint.get("cumulative_controller_tokens")),
        "cumulative_cost": checkpoint.get("cumulative_cost"),
        "controller_runtime_state": copy.deepcopy(checkpoint.get("controller_runtime_state")),
        "accounting_note": (
            "These cumulative values describe the shared checkpoint prefix and are excluded "
            "from branch token_usage.jsonl and branch comparison costs."
        ),
    }


def build_full_trajectory_rows(
    prefix_rows: list[dict[str, Any]],
    continuation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one directly readable prefix-plus-branch trajectory."""
    rows: list[dict[str, Any]] = []
    for row in prefix_rows:
        materialized = copy.deepcopy(row)
        materialized["sequence_index"] = len(rows)
        materialized.setdefault("segment_sequence_index", len(rows))
        materialized["trajectory_segment"] = "checkpoint_prefix"
        rows.append(materialized)
    for segment_index, row in enumerate(continuation_rows):
        materialized = copy.deepcopy(row)
        materialized["sequence_index"] = len(rows)
        materialized["segment_sequence_index"] = segment_index
        materialized["trajectory_segment"] = "branch_continuation"
        materialized["source_checkpoint_id"] = None
        rows.append(materialized)
    return rows


def build_controller_summary(
    *,
    identity: RunIdentity,
    enabled: bool,
    model: str | None,
    prompt_version: str | None,
    prompt_hash: str | None,
    controller_calls: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a Controller-only analysis record derived from runtime ledgers."""
    controller_tokens = usage_summary(token_rows, "controller")
    action_counts: dict[str, int] = {action: 0 for action in CONTROLLER_ACTIONS}
    for call in controller_calls:
        action = call.get("action")
        if action not in action_counts:
            raise ValueError(f"unknown Controller action in call ledger: {action!r}")
        action_counts[action] = action_counts.get(action, 0) + 1
    costs = [row.get("cost") for row in token_rows if row.get("source") == "controller"]
    latencies = [
        call.get("controller_response", {}).get("latency_seconds")
        for call in controller_calls
        if isinstance(call.get("controller_response"), dict)
    ]
    return identity.to_dict() | {
        "enabled": enabled,
        "accounting_scope": "branch_continuation_only",
        "model": model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "calls": len(controller_calls),
        "blocks": sum(1 for call in controller_calls if call.get("blocked")),
        "fail_open_count": (
            sum(1 for call in controller_calls if call.get("fail_open"))
            + sum(1 for event in runtime_events if event.get("event_type") == "controller_fail_open")
        ),
        "runtime_event_counts": count_values(runtime_events, "event_type"),
        "fail_open_by_reason": count_values(
            [call for call in controller_calls if call.get("fail_open")]
            + [event for event in runtime_events if event.get("event_type") == "controller_fail_open"],
            "reason",
        ),
        "blocked_guidance_count": sum(
            1 for call in controller_calls if call.get("blocked") and call.get("guidance")
        ),
        "action_counts": action_counts,
        "tokens": controller_tokens,
        "cost_cny": nullable_metric_summary(costs),
        "latency_seconds": nullable_metric_summary(latencies, include_average=True),
        "controller_calls_path": "controller_calls.jsonl",
        "controller_runtime_events_path": "controller_runtime_events.jsonl",
        "token_usage_path": "token_usage.jsonl",
    }


def normalize_evaluator_result(
    *,
    identity: RunIdentity,
    raw_result: dict[str, Any],
    evaluator_version: str | None,
    report_path: Path | None,
) -> dict[str, Any]:
    """Expose stable metrics while preserving the official evaluator output."""
    result = raw_result.get("result") if isinstance(raw_result, dict) else None
    result = result if isinstance(result, dict) else {}
    failures_value = result.get("failures")
    passes_value = result.get("passes")
    failures = failures_value if isinstance(failures_value, list) else []
    passes = passes_value if isinstance(passes_value, list) else []
    total = optional_int(result.get("num_tests"))
    passed = len(passes) if total is not None and isinstance(passes_value, list) else None
    if total is not None and passed is None and isinstance(failures_value, list):
        passed = max(0, total - len(failures))
    score = passed / total if passed is not None and total else None
    success = bool(result.get("success")) if "success" in result else None
    return identity.to_dict() | {
        "success": success,
        "score": score,
        "passed": passed,
        "total": total,
        "evaluator_version": evaluator_version,
        "report_path": None if report_path is None else str(report_path),
        "raw_result": raw_result,
    }


def usage_summary(rows: list[dict[str, Any]], source: str) -> dict[str, int | None]:
    """Aggregate one source while propagating unavailable token fields."""
    selected = [row for row in rows if row.get("source") == source]
    if not selected:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = sum_complete(row.get("input_tokens") for row in selected)
    output_tokens = sum_complete(row.get("output_tokens") for row in selected)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": add_nullable(input_tokens, output_tokens),
    }


def combined_usage(agent: dict[str, Any], controller: dict[str, Any]) -> dict[str, int | None]:
    """Combine Agent and Controller aggregates without inventing missing values."""
    return {
        "input_tokens": add_nullable(agent["input_tokens"], controller["input_tokens"]),
        "output_tokens": add_nullable(agent["output_tokens"], controller["output_tokens"]),
        "total_tokens": add_nullable(agent["total_tokens"], controller["total_tokens"]),
    }


def copy_raw_logs(appworld_task_dir: Path, raw_logs_dir: Path) -> dict[str, str]:
    """Copy AppWorld branch logs byte-for-byte and return their SHA-256 hashes."""
    raw_logs_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for target_name, source_parts in RAW_LOG_SOURCES.items():
        source = appworld_task_dir.joinpath(*source_parts)
        if not source.exists():
            raise FileNotFoundError(f"Missing AppWorld raw log: {source}")
        target = raw_logs_dir / target_name
        shutil.copy2(source, target)
        hashes[target_name] = sha256_file(target)
    return hashes


def build_run_manifest(
    *,
    identity: RunIdentity,
    checkpoint_path: Path,
    agent_model: str,
    controller_model: str | None,
    controller_enabled: bool,
    prompt_version: str | None,
    prompt_hash: str | None,
    seed: int,
    budget: dict[str, Any],
    appworld_code_version: str | None,
    appworld_data_version: str | None,
    raw_log_hashes: dict[str, str],
    trajectory_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete immutable identity record for one branch."""
    return identity.to_dict() | {
        "checkpoint": {
            "source_path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "agent_model": agent_model,
        "controller_model": controller_model,
        "controller_enabled": controller_enabled,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "seed": seed,
        "budget": budget,
        "appworld_code_version": appworld_code_version,
        "appworld_data_version": appworld_data_version,
        "trajectory": trajectory_summary,
        "required_files": list(RUN_REQUIRED_FILES),
        "raw_log_sha256": raw_log_hashes,
    }


def branch_result_row(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    """Build the root-level batch-readable row for one branch."""
    keys = (
        "experiment_id",
        "run_id",
        "task_id",
        "checkpoint_id",
        "branch",
        "success",
        "score",
        "termination_reason",
        "agent_turns",
        "controller_calls",
        "controller_blocks",
        "fail_open_count",
    )
    return {key: result.get(key) for key in keys} | {
        "agent_tokens": result.get("agent_tokens", {}).get("total_tokens"),
        "controller_tokens": result.get("controller_tokens", {}).get("total_tokens"),
        "combined_tokens": result.get("combined_tokens", {}).get("total_tokens"),
        "result_path": str(result_path),
    }


def controller_result_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    """Flatten one Controller summary for root-level analysis."""
    keys = (
        "experiment_id",
        "run_id",
        "task_id",
        "checkpoint_id",
        "branch",
        "enabled",
        "model",
        "prompt_version",
        "prompt_hash",
        "calls",
        "blocks",
        "fail_open_count",
        "blocked_guidance_count",
        "action_counts",
        "runtime_event_counts",
        "fail_open_by_reason",
        "cost_cny",
        "latency_seconds",
    )
    return {key: summary.get(key) for key in keys} | {
        "input_tokens": summary.get("tokens", {}).get("input_tokens"),
        "output_tokens": summary.get("tokens", {}).get("output_tokens"),
        "total_tokens": summary.get("tokens", {}).get("total_tokens"),
        "summary_path": str(summary_path),
    }


def comparison_row(
    identity: RunIdentity,
    baseline: dict[str, Any],
    controller: dict[str, Any],
) -> dict[str, Any]:
    """Compare the two implemented branches without fabricating an oracle."""
    baseline_success = baseline.get("success")
    controller_success = controller.get("success")
    baseline_score = baseline.get("score")
    controller_score = controller.get("score")
    baseline_tokens = baseline.get("combined_tokens", {}).get("total_tokens")
    controller_tokens = controller.get("combined_tokens", {}).get("total_tokens")
    return {
        "experiment_id": identity.experiment_id,
        "task_id": identity.task_id,
        "checkpoint_id": identity.checkpoint_id,
        "baseline_success": baseline_success,
        "controller_success": controller_success,
        "oracle_success": None,
        "controller_improved": (
            bool(controller_success and not baseline_success)
            if baseline_success is not None and controller_success is not None
            else None
        ),
        "checkpoint_recoverable": None,
        "score_difference": rounded_subtract_nullable(controller_score, baseline_score),
        "token_difference": subtract_nullable(controller_tokens, baseline_tokens),
    }


def extract_final_answer(continuation: list[dict[str, Any]]) -> Any:
    """Extract a literal complete_task answer when the generated code contains one."""
    for event in reversed(continuation):
        code = event.get("generated_code")
        if not isinstance(code, str) or "complete_task" not in code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_complete_task_call(node.func):
                continue
            for keyword in node.keywords:
                if keyword.arg == "answer":
                    try:
                        return ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        return None
            return None
    return None


def write_json(path: Path, value: Any) -> None:
    """Write one stable UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a complete JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_assistant_output(agent: Any) -> str | None:
    """Return the exact latest assistant message after an Agent model call."""
    for message in reversed(list(getattr(agent, "messages", []))):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def latest_assistant_output_from_messages(messages: Any) -> str | None:
    """Read the latest Assistant output from a persisted message list."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def usage_tokens(usage: Any, kind: str) -> int | None:
    """Read AppWorld token fields without replacing missing values with zero."""
    if usage is None:
        return None
    tokens = getattr(usage, "tokens", None)
    if tokens is not None and kind == "input":
        value = getattr(tokens, "input_cache_hit_or_miss", None)
        if value is None:
            hit = getattr(tokens, "input_cache_hit", None)
            miss = getattr(tokens, "input_cache_miss", None)
            if hit is not None and miss is not None:
                value = hit + miss
        return optional_int(value)
    if tokens is not None and kind == "output":
        return optional_int(getattr(tokens, "output", None))
    return optional_int(getattr(usage, f"{kind}_tokens", None))


def usage_cost(usage: Any) -> float | None:
    """Read one AppWorld call's total cost when available."""
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    value = getattr(cost, "total", None) if cost is not None else None
    if value is None and isinstance(cost, (int, float)):
        value = cost
    return optional_float(value)


def parse_raw_response(value: Any) -> dict[str, Any]:
    """Represent provider response availability explicitly."""
    if value is None:
        return {"available": False, "value": None}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return {"available": True, "value": _jsonable(value)}


def optional_int(value: Any) -> int | None:
    """Convert a present numeric field to int."""
    return None if value is None else int(value)


def optional_float(value: Any) -> float | None:
    """Convert a present numeric field to float."""
    return None if value is None else float(value)


def add_nullable(left: Any, right: Any) -> int | float | None:
    """Add two values only when both are available."""
    return None if left is None or right is None else left + right


def subtract_nullable(left: Any, right: Any) -> int | float | None:
    """Subtract two values only when both are available."""
    return None if left is None or right is None else left - right


def rounded_subtract_nullable(left: Any, right: Any) -> int | float | None:
    """Subtract display metrics without binary floating-point noise."""
    difference = subtract_nullable(left, right)
    return round(difference, 12) if isinstance(difference, float) else difference


def sum_complete(values: Any) -> int | None:
    """Sum values while propagating any unavailable call."""
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(int(value) for value in materialized)


def sum_nullable_numbers(values: Any) -> int | float | None:
    """Sum numeric values while preserving provider unavailability."""
    materialized = list(values)
    if not materialized:
        return 0
    if any(value is None for value in materialized):
        return None
    return sum(materialized)


def average_nullable_numbers(values: Any) -> float | None:
    """Average numeric values while preserving provider unavailability."""
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(float(value) for value in materialized) / len(materialized)


def nullable_metric_summary(values: Any, *, include_average: bool = False) -> dict[str, Any]:
    """Describe nullable per-call metrics without conflating zero and missing."""
    materialized = list(values)
    known = [value for value in materialized if value is not None]
    result: dict[str, Any] = {
        "total": sum_nullable_numbers(materialized),
        "known_calls": len(known),
        "unknown_calls": len(materialized) - len(known),
    }
    if include_average:
        result["average"] = average_nullable_numbers(materialized)
    return result


def count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Return a stable frequency table for present string values."""
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _is_complete_task_call(function: ast.expr) -> bool:
    if isinstance(function, ast.Attribute):
        return function.attr == "complete_task"
    if isinstance(function, ast.Name):
        return function.id == "complete_task"
    return False


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
