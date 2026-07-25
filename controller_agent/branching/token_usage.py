"""Build deterministic token-usage ledgers for checkpoint branch experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TokenUsageReport:
    """Outcome of rebuilding one experiment's token ledgers."""

    experiment_root: Path
    prefix_ledgers: int
    branch_ledgers: int
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TokenUsageValidationReport:
    """Outcome of validating generated token ledgers."""

    experiment_root: Path
    checked_branches: int
    errors: list[str] = field(default_factory=list)


def build_experiment_token_usage(experiment_root: Path) -> TokenUsageReport:
    """Rebuild all prefix and continuation ledgers under one experiment root."""
    root = Path(experiment_root)
    errors: list[str] = []
    prefix_totals: dict[tuple[str, str], int | None] = {}
    prefix_ledgers = 0
    branch_ledgers = 0
    branch_rows: list[dict[str, Any]] = []

    for task_dir in sorted((root / "tasks").glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        checkpoint_root = task_dir / "checkpoints"
        for checkpoint_dir in sorted(checkpoint_root.glob("*")):
            if not checkpoint_dir.is_dir():
                continue
            checkpoint_id = checkpoint_dir.name
            manifest = _read_json(checkpoint_dir / "checkpoint_manifest.json")
            checkpoint_step = _checkpoint_step(manifest, checkpoint_dir)
            prefix_records = _build_prefix_records(checkpoint_dir, checkpoint_step)
            _write_jsonl(checkpoint_dir / "prefix_token_usage.jsonl", prefix_records)
            prefix_totals[(task_id, checkpoint_id)] = _total(prefix_records)
            prefix_ledgers += 1

        for branch_dir in sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name != "checkpoints"):
            result_path = branch_dir / "result.json"
            if not result_path.is_file():
                continue
            result = _read_json(result_path)
            checkpoint_id = str(result.get("checkpoint_id") or _checkpoint_id_from_branch_dir(branch_dir.name))
            checkpoint_dir = checkpoint_root / checkpoint_id
            manifest = _read_json(checkpoint_dir / "checkpoint_manifest.json")
            checkpoint_step = _checkpoint_step(manifest, checkpoint_dir)
            branch = str(result.get("branch") or _branch_from_dir(branch_dir.name))
            records = _build_branch_records(branch_dir, checkpoint_step)
            _write_jsonl(branch_dir / "token_usage.jsonl", records)
            branch_manifest = _read_json(branch_dir / "manifest.json")
            files = branch_manifest.get("files") if isinstance(branch_manifest.get("files"), dict) else {}
            continuation_known = files.get("continuation.jsonl") is not False
            _update_result(result, records, prefix_totals.get((task_id, checkpoint_id)), branch, continuation_known)
            _write_json(result_path, result)
            branch_rows.append(dict(result) | {"result_path": str(result_path.relative_to(root))})
            branch_ledgers += 1

    _write_jsonl(root / "branch_results.jsonl", sorted(branch_rows, key=_branch_sort_key))
    _write_jsonl(root / "comparison_summary.jsonl", _comparison_rows(root, branch_rows))
    return TokenUsageReport(root, prefix_ledgers, branch_ledgers, errors)


def validate_experiment_token_usage(experiment_root: Path) -> TokenUsageValidationReport:
    """Validate ledger identities, boundaries, and result totals without rewriting files."""
    root = Path(experiment_root)
    errors: list[str] = []
    checked = 0
    for task_dir in sorted((root / "tasks").glob("*")):
        if not task_dir.is_dir():
            continue
        checkpoints = task_dir / "checkpoints"
        for branch_dir in sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name != "checkpoints"):
            result_path = branch_dir / "result.json"
            if not result_path.is_file():
                continue
            checked += 1
            result = _read_json(result_path)
            checkpoint_id = str(result.get("checkpoint_id") or _checkpoint_id_from_branch_dir(branch_dir.name))
            checkpoint_step = _checkpoint_step(_read_json(checkpoints / checkpoint_id / "checkpoint_manifest.json"), checkpoints / checkpoint_id)
            rows = _read_jsonl(branch_dir / "token_usage.jsonl")
            ids = [row.get("call_id") for row in rows]
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate call_id in {branch_dir}")
            for row in rows:
                step = row.get("step")
                if not isinstance(step, int) or step <= checkpoint_step:
                    errors.append(f"Token ledger crosses checkpoint boundary in {branch_dir}")
            agent_total = _total([row for row in rows if row.get("source") == "agent"])
            branch_manifest = _read_json(branch_dir / "manifest.json")
            files = branch_manifest.get("files") if isinstance(branch_manifest.get("files"), dict) else {}
            if not [row for row in rows if row.get("source") == "agent"] and files.get("continuation.jsonl") is False:
                agent_total = None
            controller_rows = [row for row in rows if row.get("source") == "controller"]
            branch = str(result.get("branch") or _branch_from_dir(branch_dir.name))
            controller_total = 0 if branch == "baseline" and not controller_rows else (_total(controller_rows) if controller_rows else None)
            prefix_total = _total(_read_jsonl(checkpoints / checkpoint_id / "prefix_token_usage.jsonl"))
            combined_total = None if agent_total is None or controller_total is None else agent_total + controller_total
            effective_total = None if prefix_total is None or combined_total is None else prefix_total + combined_total
            if result.get("prefix_total_tokens") != prefix_total:
                errors.append(f"Prefix total does not match ledger in {branch_dir}")
            if result.get("continuation_agent_total_tokens") != agent_total:
                errors.append(f"Agent total does not match ledger in {branch_dir}")
            if result.get("continuation_controller_total_tokens") != controller_total:
                errors.append(f"Controller total does not match ledger in {branch_dir}")
            if result.get("continuation_combined_total_tokens") != combined_total:
                errors.append(f"Combined total does not match ledger in {branch_dir}")
            if result.get("effective_total_tokens") != effective_total:
                errors.append(f"Effective total does not match ledgers in {branch_dir}")
    return TokenUsageValidationReport(root, checked, errors)


def _build_prefix_records(checkpoint_dir: Path, checkpoint_step: int) -> list[dict[str, Any]]:
    """Build a shared prefix ledger by ordered Agent-call position."""
    trajectory = _read_jsonl(checkpoint_dir / "prefix_trajectory.jsonl")
    agent_steps = [_step(row) for row in trajectory if row.get("role") == "agent"]
    if any(step is None or step > checkpoint_step for step in agent_steps):
        raise ValueError(f"Prefix trajectory crosses checkpoint boundary in {checkpoint_dir}")
    manifest = _read_json(checkpoint_dir / "checkpoint_manifest.json")
    source_root = manifest.get("prefix_logs_path")
    source = Path(str(source_root)) / "lm_calls.jsonl" if source_root else checkpoint_dir / "raw_logs" / "lm_calls.jsonl"
    raw_calls = _read_jsonl(source) if source.is_file() else []
    matched = raw_calls[: len(agent_steps)] if len(raw_calls) >= len(agent_steps) else None
    return _agent_records(agent_steps, matched)


def _build_branch_records(branch_dir: Path, checkpoint_step: int) -> list[dict[str, Any]]:
    """Build one branch ledger from continuation, raw Agent calls, and Controller calls."""
    continuation = _read_jsonl(branch_dir / "continuation.jsonl")
    agent_entries: list[tuple[int, str]] = []
    seen_agent_ids: set[str] = set()
    per_step: dict[int, int] = {}
    for row in continuation:
        step = _step(row)
        if step is None or step <= checkpoint_step:
            raise ValueError(f"Continuation crosses checkpoint boundary in {branch_dir}")
        count = per_step.get(step, 0) + 1
        per_step[step] = count
        call_id = row.get("agent_call_id")
        agent_call_id = str(call_id) if isinstance(call_id, str) and call_id else _agent_call_id(step, count)
        if agent_call_id in seen_agent_ids:
            raise ValueError(f"Duplicate agent call_id {agent_call_id} in {branch_dir}")
        seen_agent_ids.add(agent_call_id)
        agent_entries.append((step, agent_call_id))
    raw_path = branch_dir / "raw_logs" / "lm_calls.jsonl"
    raw_calls = _read_jsonl(raw_path) if raw_path.is_file() else []
    matched = raw_calls if len(raw_calls) == len(agent_entries) else None
    records = _agent_records([step for step, _ in agent_entries], matched, [call_id for _, call_id in agent_entries])
    records.extend(_controller_records(branch_dir, checkpoint_step))
    return records


def _agent_records(
    steps: list[int | None], raw_calls: list[dict[str, Any]] | None, call_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Convert ordered Agent raw calls into stable ledger rows without text matching."""
    records: list[dict[str, Any]] = []
    per_step: dict[int, int] = {}
    for index, maybe_step in enumerate(steps):
        if maybe_step is None:
            raise ValueError("Agent token record is missing a step")
        step = maybe_step
        per_step[step] = per_step.get(step, 0) + 1
        call_id = call_ids[index] if call_ids is not None else _agent_call_id(step, per_step[step])
        raw_call = raw_calls[index] if raw_calls is not None else None
        model, input_tokens, output_tokens, total_tokens = _lm_usage(raw_call)
        available = all(isinstance(value, int) and value >= 0 for value in (input_tokens, output_tokens, total_tokens))
        records.append(
            {
                "step": step,
                "call_id": call_id,
                "source": "agent",
                "model": model,
                "input_tokens": input_tokens if available else None,
                "output_tokens": output_tokens if available else None,
                "total_tokens": total_tokens if available else None,
                "available": available,
            }
        )
    _assert_unique_call_ids(records)
    return records


def _controller_records(branch_dir: Path, checkpoint_step: int) -> list[dict[str, Any]]:
    """Convert direct Controller decision usage into ledger rows."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _read_jsonl(branch_dir / "controller_calls.jsonl"):
        step = _step(row)
        call_id = row.get("controller_call_id")
        if step is None or step <= checkpoint_step:
            raise ValueError(f"Controller call crosses checkpoint boundary in {branch_dir}")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"Controller call is missing controller_call_id in {branch_dir}")
        if call_id in seen:
            raise ValueError(f"Duplicate controller call_id {call_id} in {branch_dir}")
        seen.add(call_id)
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        input_tokens = _nonnegative_int(usage.get("input_tokens"))
        output_tokens = _nonnegative_int(usage.get("output_tokens"))
        total_tokens = _nonnegative_int(usage.get("total_tokens"))
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
    _assert_unique_call_ids(records)
    return records


def _lm_usage(row: dict[str, Any] | None) -> tuple[str | None, int | None, int | None, int | None]:
    """Read provider token fields from one AppWorld raw lm_calls row."""
    if row is None:
        return None, None, None, None
    input_data = row.get("input") if isinstance(row.get("input"), dict) else {}
    output_data = row.get("output") if isinstance(row.get("output"), dict) else {}
    usage = output_data.get("usage") if isinstance(output_data.get("usage"), dict) else {}
    input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
    output_tokens = _nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _nonnegative_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    model = input_data.get("model") if isinstance(input_data.get("model"), str) else output_data.get("model")
    return model if isinstance(model, str) else None, input_tokens, output_tokens, total_tokens


def _update_result(
    result: dict[str, Any],
    records: list[dict[str, Any]],
    prefix_total: int | None,
    branch: str,
    continuation_known: bool,
) -> None:
    """Set aggregate token fields only from the newly written ledger."""
    agent_records = [row for row in records if row["source"] == "agent"]
    controller_records = [row for row in records if row["source"] == "controller"]
    agent_total = _total(agent_records) if agent_records or continuation_known else None
    controller_total = (
        0
        if branch == "baseline" and not controller_records
        else (_total(controller_records) if controller_records else None)
    )
    combined = None if agent_total is None or controller_total is None else agent_total + controller_total
    effective = None if prefix_total is None or combined is None else prefix_total + combined
    available_records = [row for row in records if bool(row["available"])]
    values = (prefix_total, agent_total, controller_total)
    if all(value is not None for value in values):
        status = "complete"
    elif available_records:
        status = "partial"
    else:
        status = "unavailable"
    result.update(
        {
            "prefix_total_tokens": prefix_total,
            "continuation_agent_total_tokens": agent_total,
            "continuation_controller_total_tokens": controller_total,
            "continuation_combined_total_tokens": combined,
            "effective_total_tokens": effective,
            "token_usage_status": status,
            "token_usage_available": status == "complete",
        }
    )


def _total(records: Iterable[dict[str, Any]]) -> int | None:
    """Return a total only when every record has an available token value."""
    materialized = list(records)
    if not materialized:
        return 0
    values = [row.get("total_tokens") for row in materialized]
    if any(not isinstance(value, int) or value < 0 for value in values):
        return None
    return sum(values)


def _comparison_rows(root: Path, branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regenerate a stable checkpoint comparison index from branch result files."""
    previous = {
        (str(row.get("task_id")), str(row.get("checkpoint_id"))): row
        for row in _read_jsonl(root / "comparison_summary.jsonl")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in branch_rows:
        grouped.setdefault((str(row.get("task_id")), str(row.get("checkpoint_id"))), []).append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        task_id, checkpoint_id = key
        branches = sorted(grouped[key], key=_branch_sort_key)
        statuses = {str(row.get("branch")): row.get("termination_status") for row in branches}
        old = previous.get(key, {})
        rows.append(
            {
                "task_id": task_id,
                "checkpoint_id": checkpoint_id,
                "baseline_result": statuses.get("baseline", old.get("baseline_result", "unknown")),
                "controller_result": statuses.get("controller"),
                "branch_results": statuses,
            }
        )
    return rows


def _checkpoint_step(manifest: dict[str, Any], checkpoint_dir: Path) -> int:
    """Read the one authoritative checkpoint boundary field."""
    for field in ("last_step_number", "agent_turn_index"):
        value = manifest.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError(f"Checkpoint manifest has no step boundary: {checkpoint_dir}")


def _step(row: dict[str, Any]) -> int | None:
    """Read an absolute step from current or legacy structured data."""
    for field in ("step", "step_number", "agent_turn_index"):
        value = row.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _agent_call_id(step: int, occurrence: int) -> str:
    """Reproduce the current deterministic agent call-id convention."""
    return f"agent_call_{step:04d}" if occurrence == 1 else f"agent_call_{step:04d}_{occurrence:02d}"


def _assert_unique_call_ids(records: list[dict[str, Any]]) -> None:
    """Reject duplicate call identities within one ledger source."""
    values = [str(row["call_id"]) for row in records]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate call_id in token ledger")


def _branch_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return a deterministic branch-index sort key."""
    return (
        str(row.get("task_id", "")),
        str(row.get("checkpoint_id", "")),
        str(row.get("branch", "")),
        str(row.get("result_path", "")),
    )


def _checkpoint_id_from_branch_dir(name: str) -> str:
    """Read the checkpoint ID from the canonical branch directory name."""
    parts = name.split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer checkpoint id from branch directory: {name}")
    return parts[1]


def _branch_from_dir(name: str) -> str:
    """Read the logical branch label from a canonical branch directory name."""
    parts = name.split("__", 2)
    return parts[2] if len(parts) == 3 else name


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object or return an empty object for absent optional files."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON object rows from a JSONL file."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _nonnegative_int(value: Any) -> int | None:
    """Return a non-negative integer token value when available."""
    return value if isinstance(value, int) and value >= 0 else None
