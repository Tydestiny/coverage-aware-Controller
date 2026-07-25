"""Direct structured logging for one restored checkpoint branch."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

from controller_agent.state.schemas import ControllerInput, ControllerResponse


_SENSITIVE_KEY = re.compile(r"password|access[_-]?token|refresh[_-]?token|authorization|api[_-]?key|secret|credential", re.I)
_TEXT_SECRET = re.compile(
    r"(?i)(password|access[_-]?token|refresh[_-]?token|authorization|api[_-]?key|secret|credential)"
    r"(\\?[\"']?\s*[=:]\s*\\?[\"']?)([^\s,}\]\n\\\"']+)"
)


class BranchRuntimeLogger:
    """Append direct continuation and Controller records for one branch."""

    def __init__(self, branch_dir: Path, *, branch: str, checkpoint_last_step: int) -> None:
        if checkpoint_last_step < 0:
            raise ValueError("checkpoint_last_step must be non-negative")
        self.branch_dir = Path(branch_dir)
        self.branch = branch
        self.checkpoint_last_step = checkpoint_last_step
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        self.raw_logs_dir = self.branch_dir / "raw_logs"
        self.raw_logs_dir.mkdir(exist_ok=True)
        self.continuation_path = self.branch_dir / "continuation.jsonl"
        self.controller_calls_path = self.branch_dir / "controller_calls.jsonl"
        self.continuation_path.touch(exist_ok=True)
        self.controller_calls_path.touch(exist_ok=True)
        self._agent_calls_by_step: dict[int, int] = {}
        self._controller_call_count = 0

    def record_continuation(
        self,
        *,
        step_number: int,
        agent_output: str,
        generated_code: str,
        execution_allowed: bool,
        environment_observation: Any,
        task_completed: bool,
        termination_reason: str | None,
        controller_triggered: bool = False,
        controller_call_id: str | None = None,
        controller_action: str | None = None,
        guidance: str | None = None,
    ) -> str:
        """Append exactly one post-checkpoint Agent generation record."""
        if step_number <= self.checkpoint_last_step:
            raise ValueError("Continuation step_number must be after the checkpoint boundary")
        agent_call_id = self._next_agent_call_id(step_number)
        self._append_jsonl(
            self.continuation_path,
            {
                "step_number": step_number,
                "agent_call_id": agent_call_id,
                "agent_output": _sanitize(agent_output),
                "generated_code": _sanitize(generated_code),
                "controller_triggered": controller_triggered,
                "controller_call_id": controller_call_id,
                "controller_action": controller_action,
                "guidance": _sanitize(guidance),
                "execution_allowed": execution_allowed,
                "environment_observation": _sanitize(environment_observation),
                "task_completed": task_completed,
                "termination_reason": termination_reason,
            },
        )
        return agent_call_id

    def record_controller_call(
        self,
        *,
        step_number: int,
        controller_input: ControllerInput,
        controller_response: ControllerResponse,
        guidance: str | None,
        blocked: bool,
        fail_open: bool,
        prompt_hash: str | None,
    ) -> str:
        """Append one complete Controller decision at the time it is made."""
        if step_number <= self.checkpoint_last_step:
            raise ValueError("Controller step_number must be after the checkpoint boundary")
        self._controller_call_count += 1
        controller_call_id = f"controller_call_{self._controller_call_count:04d}"
        input_tokens = controller_response.input_tokens
        output_tokens = controller_response.output_tokens
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        self._append_jsonl(
            self.controller_calls_path,
            {
                "controller_call_id": controller_call_id,
                "step_number": step_number,
                "controller_input": _sanitize(_to_jsonable(controller_input)),
                "controller_response": _sanitize(_to_jsonable(controller_response)),
                "raw_response": _sanitize(controller_response.raw_response),
                "action": controller_response.decision.action.value,
                "reason": controller_response.decision.reason,
                "guidance": _sanitize(guidance),
                "blocked": blocked,
                "fail_open": fail_open,
                "model": controller_response.model_name,
                "prompt_hash": prompt_hash,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "record_origin": "runtime_direct",
            },
        )
        return controller_call_id

    def write_checkpoint_manifest(
        self,
        path: Path,
        *,
        checkpoint_id: str,
        state_path: str | None,
        source_trajectory: str | None,
        source_logs: str | None,
    ) -> None:
        """Write the shared checkpoint boundary metadata without branch history."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_id": checkpoint_id,
            "last_step_number": self.checkpoint_last_step,
            "continuation_start_step_number": self.checkpoint_last_step + 1,
            "state_path": state_path,
            "source_trajectory": source_trajectory,
            "source_logs": source_logs,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def update_last_continuation(self, *, task_completed: bool, termination_reason: str | None) -> None:
        """Finalize the latest runtime record once its terminal state is known."""
        rows = self.continuation_path.read_text(encoding="utf-8").splitlines()
        if not rows:
            return
        latest = json.loads(rows[-1])
        latest["task_completed"] = task_completed
        latest["termination_reason"] = termination_reason
        rows[-1] = json.dumps(latest, ensure_ascii=False, sort_keys=True)
        self.continuation_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def archive_raw_logs(self, source_logs_dir: Path) -> list[str]:
        """Archive post-restore logs while redacting credentials from durable artifacts."""
        copied: list[str] = []
        for name in ("logger.jsonl", "lm_calls.jsonl", "api_calls.jsonl", "environment_io.md", "logger.log", "usage.json"):
            source = Path(source_logs_dir) / name
            if source.is_file():
                destination = self.raw_logs_dir / name
                if source.suffix == ".jsonl":
                    _archive_jsonl_with_redaction(source, destination)
                elif source.suffix == ".json":
                    _archive_json_with_redaction(source, destination)
                else:
                    destination.write_text(str(_sanitize(source.read_text(encoding="utf-8"))), encoding="utf-8")
                copied.append(name)
        return copied

    def _next_agent_call_id(self, step_number: int) -> str:
        """Return a stable call id that stays unique for repeated raw steps."""
        count = self._agent_calls_by_step.get(step_number, 0) + 1
        self._agent_calls_by_step[step_number] = count
        suffix = "" if count == 1 else f"_{count:02d}"
        return f"agent_call_{step_number:04d}{suffix}"

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        """Append one JSON object without rewriting previous calls."""
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    """Convert public schema objects to JSON-compatible values."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize(value: Any) -> Any:
    """Copy and redact credential values from structured branch records."""
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _SENSITIVE_KEY.search(str(key)) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _TEXT_SECRET.sub(r"\1\2<redacted>", value)
    return copy.deepcopy(value)


def _archive_jsonl_with_redaction(source: Path, destination: Path) -> None:
    """Archive JSON Lines records without retaining literal credentials."""
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            lines.append(str(_sanitize(line)))
        else:
            lines.append(json.dumps(_sanitize(value), ensure_ascii=False, sort_keys=True))
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _archive_json_with_redaction(source: Path, destination: Path) -> None:
    """Archive a JSON document without retaining literal credentials."""
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        destination.write_text(str(_sanitize(source.read_text(encoding="utf-8"))), encoding="utf-8")
        return
    destination.write_text(json.dumps(_sanitize(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
