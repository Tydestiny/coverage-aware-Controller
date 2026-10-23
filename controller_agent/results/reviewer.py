"""Build an offline, read-only review page from one experiment run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read an optional JSON object without inventing missing results."""
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, required: bool = False) -> list[dict[str, Any]] | None:
    """Read JSONL and report malformed rows with their source line."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def load_review_data(run_dir: Path) -> dict[str, Any]:
    """Join existing run records by authoritative step and call IDs."""
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(run_dir / "manifest.json")
    continuation = _read_jsonl(run_dir / "continuation.jsonl", required=True)
    calls = _read_jsonl(run_dir / "controller_calls.jsonl")
    triggers = _read_jsonl(run_dir / "trigger_events.jsonl")
    tokens = _read_jsonl(run_dir / "token_usage.jsonl")
    assert continuation is not None
    warnings: list[str] = []
    if calls is None:
        warnings.append("controller_calls.jsonl is missing; Controller details are unavailable.")
        calls = []
    if triggers is None:
        warnings.append("trigger_events.jsonl is missing; trigger history is unavailable.")
        triggers = []
    if tokens is None:
        warnings.append("token_usage.jsonl is missing; per-call token usage is unavailable.")
        tokens = []

    call_by_id: dict[str, dict[str, Any]] = {}
    for call in calls:
        call_id = call.get("controller_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("controller_calls.jsonl contains a call without controller_call_id")
        if call_id in call_by_id:
            raise ValueError(f"Duplicate controller_call_id: {call_id}")
        call_by_id[call_id] = call

    triggers_by_step: dict[int, list[dict[str, Any]]] = {}
    for trigger in triggers:
        step = trigger.get("step_index")
        if isinstance(step, int) and not isinstance(step, bool):
            triggers_by_step.setdefault(step, []).append(trigger)
        else:
            warnings.append("A trigger record has no integer step_index and could not be aligned.")

    tokens_by_id: dict[str, dict[str, Any]] = {}
    for record in tokens:
        call_id = record.get("call_id")
        if isinstance(call_id, str):
            if call_id in tokens_by_id:
                raise ValueError(f"Duplicate token call_id: {call_id}")
            tokens_by_id[call_id] = record

    rows: list[dict[str, Any]] = []
    seen_agent_ids: set[str] = set()
    linked_calls: set[str] = set()
    for index, event in enumerate(continuation, 1):
        step = event.get("step_number")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError(f"continuation.jsonl row {index} has no integer step_number")
        agent_id = event.get("agent_call_id")
        if isinstance(agent_id, str):
            if agent_id in seen_agent_ids:
                raise ValueError(f"Duplicate agent_call_id: {agent_id}")
            seen_agent_ids.add(agent_id)
        call_id = event.get("controller_call_id")
        call = call_by_id.get(call_id) if isinstance(call_id, str) else None
        if call_id is not None and call is None:
            warnings.append(f"Step {step} references missing Controller call {call_id}.")
        if call is not None:
            linked_calls.add(call_id)
            if call.get("step_number") != step:
                raise ValueError(f"Controller call {call_id} step_number differs from continuation")
        rows.append({
            "event": event,
            "trigger_events": triggers_by_step.get(step, []),
            "controller_call": call,
            "agent_tokens": tokens_by_id.get(agent_id),
            "controller_tokens": tokens_by_id.get(call_id),
        })
    for call_id in call_by_id.keys() - linked_calls:
        warnings.append(f"Controller call {call_id} is not referenced by continuation.jsonl.")
    observed_steps = {row["event"]["step_number"] for row in rows}
    for step in triggers_by_step.keys() - observed_steps:
        warnings.append(f"Trigger step {step} has no continuation event.")

    raw_logs = run_dir / "raw_logs"
    raw_links = {
        name: (raw_logs / name).resolve().as_uri()
        for name in ("logger.log", "logger.jsonl", "lm_calls.jsonl", "api_calls.jsonl", "environment_io.md")
        if (raw_logs / name).is_file()
    }
    return {
        "run": {"task_id": manifest.get("task_id"), "branch": manifest.get("branch"),
                "split": manifest.get("split"), "agent_model": manifest.get("agent_model"),
                "controller_model": manifest.get("controller_model"), "path": str(run_dir)},
        "manifest": manifest,
        "result": _read_json(run_dir / "result.json"),
        "evaluator": _read_json(run_dir / "evaluator_result.json"),
        "rows": rows,
        "raw_links": raw_links,
        "warnings": warnings,
    }


def write_review_page(run_dir: Path, output: Path) -> Path:
    """Write a standalone HTML page without changing source run files."""
    data = load_review_data(run_dir)
    template = (Path(__file__).with_name("reviewer.html")).read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REVIEW_DATA__", payload), encoding="utf-8")
    return output
