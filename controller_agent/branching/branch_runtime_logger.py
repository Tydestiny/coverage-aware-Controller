"""Direct structured logging for one restored checkpoint branch."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

from controller_agent.state.schemas import ControllerInput, ControllerResponse


class BranchRuntimeLogger:
    """Append direct continuation and Controller records for one branch."""

    def __init__(self, branch_dir: Path, *, branch: str, checkpoint_last_step: int) -> None:
        if checkpoint_last_step < -1:
            raise ValueError("checkpoint_last_step must be at least -1")
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
        self._cached_checkpoint_proposals: list[dict[str, Any]] = []

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
                "agent_output": copy.deepcopy(agent_output),
                "generated_code": copy.deepcopy(generated_code),
                "controller_triggered": controller_triggered,
                "controller_call_id": controller_call_id,
                "controller_action": controller_action,
                "guidance": copy.deepcopy(guidance),
                "execution_allowed": execution_allowed,
                "environment_observation": copy.deepcopy(environment_observation),
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
                "controller_input": _to_jsonable(controller_input),
                "controller_response": _to_jsonable(controller_response),
                "raw_response": copy.deepcopy(controller_response.raw_response),
                "action": controller_response.decision.action.value,
                "reason": controller_response.decision.reason,
                "guidance": copy.deepcopy(guidance),
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

    def record_cached_checkpoint_proposal(
        self,
        *,
        step_number: int,
        agent_output: str,
        generated_code: str,
        controller_triggered: bool,
        controller_action: str | None,
        guidance: str | None,
        execution_allowed: bool,
    ) -> None:
        """Preserve a replayed checkpoint proposal in the readable raw logs."""
        if step_number <= self.checkpoint_last_step:
            raise ValueError("Checkpoint proposal must be after the checkpoint boundary")
        self._cached_checkpoint_proposals.append(
            {
                "role": "agent",
                "content": agent_output,
                "raw_message": {"role": "assistant", "content": agent_output},
                "step_number": step_number,
                "event_type": "checkpoint_proposed_action",
                "generated_code": generated_code,
                "controller_triggered": controller_triggered,
                "controller_action": controller_action,
                "guidance": guidance,
                "execution_allowed": execution_allowed,
            }
        )

    def archive_raw_logs(
        self,
        source_logs_dir: Path,
        *,
        prefix_logger_jsonl: Path | None = None,
        prefix_logger_log: Path | None = None,
    ) -> list[str]:
        """Archive branch logs and optionally join the shared logger prefix."""
        copied: list[str] = []
        for name in ("logger.jsonl", "lm_calls.jsonl", "api_calls.jsonl", "environment_io.md", "logger.log", "usage.json"):
            source = Path(source_logs_dir) / name
            if source.is_file():
                destination = self.raw_logs_dir / name
                shutil.copy2(source, destination)
                copied.append(name)
        self._prepend_cached_checkpoint_proposals()
        if prefix_logger_jsonl is not None and (self.raw_logs_dir / "logger.jsonl").is_file():
            self._join_logger_history(Path(prefix_logger_jsonl))
        if prefix_logger_log is not None and (self.raw_logs_dir / "logger.log").is_file():
            self._join_text_logger_history(Path(prefix_logger_log))
        return copied

    def _prepend_cached_checkpoint_proposals(self) -> None:
        """Place replayed proposals ahead of new raw continuation events."""
        if not self._cached_checkpoint_proposals:
            return
        proposal_jsonl = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in self._cached_checkpoint_proposals
        )
        logger_jsonl = self.raw_logs_dir / "logger.jsonl"
        existing_jsonl = logger_jsonl.read_text(encoding="utf-8") if logger_jsonl.is_file() else ""
        logger_jsonl.write_text(proposal_jsonl + existing_jsonl, encoding="utf-8", newline="\n")

        logger_log = self.raw_logs_dir / "logger.log"
        proposal_text = "".join(
            "[CHECKPOINT_PROPOSED_ACTION] "
            f"step_number={record['step_number']} execution_allowed={record['execution_allowed']}\n"
            f"{record['content']}\n\n"
            for record in self._cached_checkpoint_proposals
        )
        existing_text = logger_log.read_text(encoding="utf-8") if logger_log.is_file() else ""
        logger_log.write_text(proposal_text + existing_text, encoding="utf-8", newline="\n")

    def _join_logger_history(self, prefix_logger_jsonl: Path) -> None:
        """Make the archived logger readable from task start without losing continuation."""
        continuation_path = self.raw_logs_dir / "logger.jsonl"
        if not prefix_logger_jsonl.is_file():
            return

        prefix_records = _logger_prefix_records(prefix_logger_jsonl, self.checkpoint_last_step)
        continuation_text = continuation_path.read_text(encoding="utf-8")
        continuation_copy = self.raw_logs_dir / "logger_continuation.jsonl"
        continuation_copy.write_text(continuation_text, encoding="utf-8", newline="\n")

        prefix_text = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in prefix_records
        )
        continuation_path.write_text(prefix_text + continuation_text, encoding="utf-8", newline="\n")
        history_manifest = {
            "checkpoint_last_step_number": self.checkpoint_last_step,
            "prefix_logger_source": str(prefix_logger_jsonl),
            "prefix_record_count": len(prefix_records),
            "continuation_record_count": len(continuation_text.splitlines()),
            "combined_logger": "logger.jsonl",
            "continuation_logger": "logger_continuation.jsonl",
            "cached_checkpoint_proposal_count": len(self._cached_checkpoint_proposals),
        }
        (self.raw_logs_dir / "logger_history_manifest.json").write_text(
            json.dumps(history_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _join_text_logger_history(self, prefix_logger_log: Path) -> None:
        """Join the readable logger through the same raw JSON prefix boundary."""
        continuation_path = self.raw_logs_dir / "logger.log"
        if not prefix_logger_log.is_file():
            return
        prefix_jsonl = prefix_logger_log.with_suffix(".jsonl")
        next_agent_output = _next_agent_output(prefix_jsonl, self.checkpoint_last_step)
        if next_agent_output is None:
            return
        prefix_text = _logger_log_prefix_text(prefix_logger_log, next_agent_output)
        continuation_text = continuation_path.read_text(encoding="utf-8")
        (self.raw_logs_dir / "logger_continuation.log").write_text(
            continuation_text,
            encoding="utf-8",
            newline="\n",
        )
        separator = "\n\n===== CHECKPOINT CONTINUATION =====\n\n"
        continuation_path.write_text(prefix_text + separator + continuation_text, encoding="utf-8", newline="\n")
        history_path = self.raw_logs_dir / "logger_history_manifest.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {}
        history.update({
            "prefix_logger_log_source": str(prefix_logger_log),
            "combined_text_logger": "logger.log",
            "continuation_text_logger": "logger_continuation.log",
        })
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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


def _logger_prefix_records(path: Path, checkpoint_last_step: int) -> list[dict[str, Any]]:
    """Return raw logger records through the checkpoint's final Agent step."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("role") == "agent":
            step_number = record.get("step_number")
            if isinstance(step_number, int) and step_number > checkpoint_last_step:
                break
        records.append(record)
    return records


def _next_agent_output(path: Path, checkpoint_last_step: int) -> str | None:
    """Return the first baseline Agent output after a checkpoint boundary."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("role") == "agent" and record.get("step_number") > checkpoint_last_step:
            content = record.get("content")
            return content if isinstance(content, str) and content else None
    return None


def _logger_log_prefix_text(path: Path, next_agent_output: str) -> str:
    """Keep readable logger text before the next baseline Agent segment."""
    text = path.read_text(encoding="utf-8")
    first_line = next((line.strip() for line in next_agent_output.splitlines() if line.strip()), None)
    if first_line is None:
        raise ValueError(f"Cannot determine next Agent output boundary in {path}")
    # AppWorld wraps long output lines inside the box renderer. A short prefix
    # remains contiguous while the entire raw line may not.
    marker = first_line[:60]
    content_start = text.find(marker)
    if content_start < 0:
        raise ValueError(f"Cannot find next Agent output in logger.log: {marker!r}")
    headings = list(re.finditer(r"(?m)^.*agent \(step #\d+\).*\n", text[:content_start], flags=re.I))
    if not headings:
        raise ValueError(f"Cannot find Agent logger header before checkpoint boundary in {path}")
    return text[:headings[-1].start()].rstrip() + "\n"
