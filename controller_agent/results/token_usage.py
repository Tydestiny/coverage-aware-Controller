"""Build auditable token ledgers for full-context end-to-end runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EndToEndTokenUsageReport:
    """Summary of one run's token ledger reconstruction."""

    run_dir: Path
    agent_call_count: int
    controller_call_count: int
    agent_total_tokens: int | None
    controller_total_tokens: int | None
    combined_total_tokens: int | None
    status: str


def build_end_to_end_token_usage(run_dir: Path) -> EndToEndTokenUsageReport:
    """Write a token ledger by joining ordered raw calls to structured call IDs."""
    run_path = Path(run_dir)
    continuation = _read_jsonl(run_path / "continuation.jsonl")
    raw_calls = _read_jsonl(run_path / "raw_logs" / "lm_calls.jsonl")
    if len(continuation) != len(raw_calls):
        raise ValueError(
            f"Agent call count mismatch in {run_path}: "
            f"continuation={len(continuation)}, lm_calls={len(raw_calls)}"
        )

    records = [_agent_record(event, raw_call) for event, raw_call in zip(continuation, raw_calls, strict=True)]
    records.extend(_controller_records(run_path / "controller_calls.jsonl"))
    _assert_unique_call_ids(records, run_path)
    _write_jsonl(run_path / "token_usage.jsonl", records)

    agent_records = [record for record in records if record["source"] == "agent"]
    controller_records = [record for record in records if record["source"] == "controller"]
    branch = _branch_name(run_path)
    agent_total = _total(agent_records)
    controller_total = 0 if branch == "baseline" and not controller_records else _total(controller_records)
    combined_total = None if agent_total is None or controller_total is None else agent_total + controller_total
    status = "complete" if combined_total is not None else "partial" if any(record["available"] for record in records) else "unavailable"
    report = EndToEndTokenUsageReport(
        run_dir=run_path,
        agent_call_count=len(agent_records),
        controller_call_count=len(controller_records),
        agent_total_tokens=agent_total,
        controller_total_tokens=controller_total,
        combined_total_tokens=combined_total,
        status=status,
    )
    _update_result(run_path / "result.json", report)
    return report


def _agent_record(event: dict[str, Any], raw_call: dict[str, Any]) -> dict[str, Any]:
    """Build an Agent ledger row from one exact runtime call position."""
    call_id = event.get("agent_call_id")
    step = event.get("step_number")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Continuation record is missing agent_call_id")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"Continuation record {call_id} is missing step_number")
    input_data = raw_call.get("input") if isinstance(raw_call.get("input"), dict) else {}
    output_data = raw_call.get("output") if isinstance(raw_call.get("output"), dict) else {}
    usage = output_data.get("usage") if isinstance(output_data.get("usage"), dict) else {}
    input_tokens = _token(usage.get("prompt_tokens"))
    output_tokens = _token(usage.get("completion_tokens"))
    total_tokens = _token(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    available = input_tokens is not None and output_tokens is not None and total_tokens is not None
    model = input_data.get("model") if isinstance(input_data.get("model"), str) else output_data.get("model")
    return {
        "step": step,
        "call_id": call_id,
        "source": "agent",
        "model": model if isinstance(model, str) else None,
        "input_tokens": input_tokens if available else None,
        "output_tokens": output_tokens if available else None,
        "total_tokens": total_tokens if available else None,
        "available": available,
    }


def _controller_records(path: Path) -> list[dict[str, Any]]:
    """Build Controller ledger rows from direct runtime Controller records."""
    records: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        call_id, step = row.get("controller_call_id"), row.get("step_number")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"Controller call in {path} is missing controller_call_id")
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"Controller call {call_id} is missing step_number")
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        input_tokens = _token(usage.get("input_tokens"))
        output_tokens = _token(usage.get("output_tokens"))
        total_tokens = _token(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        available = input_tokens is not None and output_tokens is not None and total_tokens is not None
        records.append(
            {
                "step": step,
                "call_id": call_id,
                "source": "controller",
                "model": row.get("model") if isinstance(row.get("model"), str) else None,
                "input_tokens": input_tokens if available else None,
                "output_tokens": output_tokens if available else None,
                "total_tokens": total_tokens if available else None,
                "available": available,
            }
        )
    return records


def _update_result(path: Path, report: EndToEndTokenUsageReport) -> None:
    """Store totals derived only from the durable per-call ledger."""
    result = _read_json(path)
    result["token_usage"] = {
        "agent_call_count": report.agent_call_count,
        "controller_call_count": report.controller_call_count,
        "agent_total_tokens": report.agent_total_tokens,
        "controller_total_tokens": report.controller_total_tokens,
        "combined_total_tokens": report.combined_total_tokens,
        "token_usage_status": report.status,
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _branch_name(run_path: Path) -> str | None:
    """Read the branch label from the run manifest without directory inference."""
    value = _read_json(run_path / "manifest.json").get("branch")
    return value if isinstance(value, str) else None


def _total(records: list[dict[str, Any]]) -> int | None:
    """Return an exact total only when every record supplies usage."""
    values = [record.get("total_tokens") for record in records]
    return sum(values) if all(isinstance(value, int) and value >= 0 for value in values) else None


def _assert_unique_call_ids(records: list[dict[str, Any]], run_path: Path) -> None:
    """Reject duplicate IDs rather than ambiguously overwriting ledger rows."""
    call_ids = [record["call_id"] for record in records]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"Duplicate token ledger call_id in {run_path}")


def _token(value: Any) -> int | None:
    """Return a non-negative provider token value when present."""
    return value if isinstance(value, int) and value >= 0 else None


def _read_json(path: Path) -> dict[str, Any]:
    """Read one optional JSON object."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, rejecting malformed audit artifacts."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Rewrite one deterministic ledger without duplicating historical rows."""
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
