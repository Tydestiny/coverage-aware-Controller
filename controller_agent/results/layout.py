"""Validation for the portable branch-experiment result layout."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REQUIRED_RAW_LOGS = (
    "logger.log",
    "logger.jsonl",
    "lm_calls.jsonl",
    "api_calls.jsonl",
    "environment_io.md",
)
_REQUIRED_RUN_FILES = (
    "manifest.json",
    "continuation.jsonl",
    "controller_calls.jsonl",
    "token_usage.jsonl",
    "result.json",
    "evaluator_result.json",
)
_IDENTITY_FIELDS = ("experiment_id", "run_id", "task_id", "checkpoint_id", "branch")
_CONTROLLER_INPUT_FIELDS = (
    "task",
    "trigger_reason",
    "trigger_metadata",
    "agent_context_messages",
    "proposed_agent_output",
    "proposed_execution",
    "previous_controller_decisions",
    "context_complete",
)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating one experiment directory."""

    experiment_root: Path
    checked_runs: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether no required-layout violations were found."""
        return not self.errors


def validate_experiment(experiment_root: Path) -> ValidationReport:
    """Validate result identities, required files, manifests, and token totals."""
    root = Path(experiment_root)
    errors: list[str] = []
    warnings: list[str] = []
    root_manifest = _load_json(root / "manifest.json", errors)
    experiment_id = _required_string(root_manifest, "experiment_id", "manifest.json", errors)
    _require_file(root / "branch_results.jsonl", errors)
    _require_file(root / "comparison_summary.jsonl", errors)
    run_entries = root_manifest.get("runs") if isinstance(root_manifest, dict) else None
    if not isinstance(run_entries, list) or not run_entries:
        errors.append("manifest.json.runs must be a non-empty list")
        return ValidationReport(root, 0, errors, warnings)

    checked_runs = 0
    for index, entry in enumerate(run_entries):
        if not isinstance(entry, dict):
            errors.append(f"manifest.json.runs[{index}] must be an object")
            continue
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"manifest.json.runs[{index}].path is required")
            continue
        run_dir = root / relative_path
        if not run_dir.is_dir():
            errors.append(f"manifest references missing run directory: {relative_path}")
            continue
        checked_runs += 1
        _validate_run(root, run_dir, experiment_id, entry, errors, warnings)

    _validate_root_index(root / "branch_results.jsonl", experiment_id, errors)
    _validate_root_index(root / "comparison_summary.jsonl", experiment_id, errors)
    return ValidationReport(root, checked_runs, errors, warnings)


def _validate_run(
    root: Path,
    run_dir: Path,
    experiment_id: str | None,
    root_entry: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate one flattened branch run directory."""
    label = str(run_dir.relative_to(root))
    for name in _REQUIRED_RUN_FILES:
        _require_file(run_dir / name, errors)
    for name in _REQUIRED_RAW_LOGS:
        _require_file(run_dir / "raw_logs" / name, errors)
    manifest = _load_json(run_dir / "manifest.json", errors)
    result = _load_json(run_dir / "result.json", errors)
    evaluator = _load_json(run_dir / "evaluator_result.json", errors)
    if not isinstance(manifest, dict) or not isinstance(result, dict) or not isinstance(evaluator, dict):
        return

    identity = {field: _required_string(manifest, field, f"{label}/manifest.json", errors) for field in _IDENTITY_FIELDS}
    if identity["experiment_id"] != experiment_id:
        errors.append(f"{label}: manifest experiment_id does not match root manifest")
    if root_entry.get("run_id") != identity["run_id"]:
        errors.append(f"{label}: root manifest run_id does not match run manifest")
    expected_dir_name = "__".join(str(identity[field]) for field in ("task_id", "checkpoint_id", "branch"))
    if run_dir.name != expected_dir_name:
        errors.append(f"{label}: directory name must be {expected_dir_name}")

    for field, value in identity.items():
        if result.get(field) != value:
            errors.append(f"{label}/result.json.{field} does not match run manifest")
    for field in ("task_id",):
        if evaluator.get(field) != identity[field]:
            errors.append(f"{label}/evaluator_result.json.{field} does not match run manifest")
    for field in ("success", "score", "passed", "total"):
        if field not in evaluator:
            errors.append(f"{label}/evaluator_result.json.{field} is required")
    result_evaluator = result.get("evaluator")
    if not isinstance(result_evaluator, dict):
        errors.append(f"{label}/result.json.evaluator is required")
    else:
        for field in ("success", "score", "passed", "total"):
            if result_evaluator.get(field) != evaluator.get(field):
                errors.append(f"{label}/result.json.evaluator.{field} does not match evaluator_result.json")

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append(f"{label}/manifest.json.files must be a list")
    else:
        for file_name in files:
            if not isinstance(file_name, str) or not (run_dir / file_name).is_file():
                errors.append(f"{label}/manifest.json references missing file: {file_name!r}")
    _validate_raw_log_hashes(run_dir, manifest, label, errors)

    _validate_continuation(run_dir / "continuation.jsonl", identity, label, errors)
    _validate_controller_calls(run_dir / "controller_calls.jsonl", identity, label, errors, warnings)
    _validate_token_usage(run_dir / "token_usage.jsonl", result, identity, label, errors)


def _validate_continuation(path: Path, identity: dict[str, str | None], label: str, errors: list[str]) -> None:
    """Validate ordered continuation events and their identity fields."""
    rows = _load_jsonl(path, errors)
    last_index = 0
    event_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{label}/continuation.jsonl[{index}] must be an object")
            continue
        _validate_row_identity(row, identity, f"{label}/continuation.jsonl[{index}]", errors)
        event_id = row.get("event_id")
        sequence_index = row.get("sequence_index")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"{label}/continuation.jsonl[{index}].event_id is required")
        elif event_id in event_ids:
            errors.append(f"{label}/continuation.jsonl contains duplicate event_id {event_id}")
        else:
            event_ids.add(event_id)
        if not isinstance(sequence_index, int) or sequence_index != last_index + 1:
            errors.append(f"{label}/continuation.jsonl sequence_index must be consecutive from 1")
        elif sequence_index > last_index:
            last_index = sequence_index


def _validate_controller_calls(
    path: Path,
    identity: dict[str, str | None],
    label: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate complete, sanitized Controller call records."""
    for index, row in enumerate(_load_jsonl(path, errors)):
        if not isinstance(row, dict):
            errors.append(f"{label}/controller_calls.jsonl[{index}] must be an object")
            continue
        row_label = f"{label}/controller_calls.jsonl[{index}]"
        _validate_row_identity(row, identity, row_label, errors)
        for field in ("event_id", "controller_call_id", "action", "reason", "guidance"):
            if not isinstance(row.get(field), str):
                errors.append(f"{row_label}.{field} is required")
        controller_input = row.get("controller_input")
        if not isinstance(controller_input, dict):
            errors.append(f"{row_label}.controller_input must be an object")
        else:
            for field in _CONTROLLER_INPUT_FIELDS:
                if field not in controller_input:
                    errors.append(f"{row_label}.controller_input.{field} is required")
        response = row.get("controller_response")
        if not isinstance(response, dict):
            errors.append(f"{row_label}.controller_response must be an object")
            continue
        decision = response.get("decision")
        if not isinstance(decision, dict) or any(field not in decision for field in ("action", "reason", "instruction")):
            errors.append(f"{row_label}.controller_response.decision is incomplete")
        if response.get("raw_response") is None:
            warnings.append(f"{row_label}: raw_response was unavailable in the source run")


def _validate_token_usage(
    path: Path,
    result: dict[str, Any],
    identity: dict[str, str | None],
    label: str,
    errors: list[str],
) -> None:
    """Validate per-call usage and compare its sums with result totals."""
    totals = {"agent": {"input": 0, "output": 0}, "controller": {"input": 0, "output": 0}}
    for index, row in enumerate(_load_jsonl(path, errors)):
        if not isinstance(row, dict):
            errors.append(f"{label}/token_usage.jsonl[{index}] must be an object")
            continue
        row_label = f"{label}/token_usage.jsonl[{index}]"
        _validate_row_identity(row, identity, row_label, errors)
        source = row.get("source")
        if source not in totals:
            errors.append(f"{row_label}.source must be agent or controller")
            continue
        if not isinstance(row.get("event_id"), str) or not isinstance(row.get("call_id"), str):
            errors.append(f"{row_label} requires event_id and call_id")
        for token_field, total_field in (("input_tokens", "input"), ("output_tokens", "output")):
            value = row.get(token_field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{row_label}.{token_field} must be a non-negative integer")
            else:
                totals[source][total_field] += value

    token_result = result.get("tokens")
    if not isinstance(token_result, dict):
        errors.append(f"{label}/result.json.tokens is required")
        return
    for source in ("agent", "controller"):
        expected = totals[source]
        actual = token_result.get(source)
        if not isinstance(actual, dict):
            errors.append(f"{label}/result.json.tokens.{source} is required")
            continue
        _check_total(actual, expected, f"{label}/result.json.tokens.{source}", errors)
    combined = token_result.get("combined")
    expected_combined = {
        "input": totals["agent"]["input"] + totals["controller"]["input"],
        "output": totals["agent"]["output"] + totals["controller"]["output"],
    }
    if not isinstance(combined, dict):
        errors.append(f"{label}/result.json.tokens.combined is required")
    else:
        _check_total(combined, expected_combined, f"{label}/result.json.tokens.combined", errors)


def _validate_raw_log_hashes(run_dir: Path, manifest: dict[str, Any], label: str, errors: list[str]) -> None:
    """Verify every preserved raw log against the run manifest digest."""
    hashes = manifest.get("raw_log_sha256")
    if not isinstance(hashes, dict):
        errors.append(f"{label}/manifest.json.raw_log_sha256 is required")
        return
    for name in _REQUIRED_RAW_LOGS:
        expected = hashes.get(name)
        if not isinstance(expected, str) or not expected:
            errors.append(f"{label}/manifest.json.raw_log_sha256.{name} is required")
            continue
        path = run_dir / "raw_logs" / name
        if path.is_file() and _sha256(path) != expected:
            errors.append(f"{label}/raw_logs/{name} does not match manifest SHA-256")


def _check_total(actual: dict[str, Any], expected: dict[str, int], label: str, errors: list[str]) -> None:
    """Compare input, output, and total fields with a computed token sum."""
    for field in ("input", "output"):
        if actual.get(field) != expected[field]:
            errors.append(f"{label}.{field} does not match token_usage.jsonl")
    expected_total = expected["input"] + expected["output"]
    if actual.get("total") != expected_total:
        errors.append(f"{label}.total does not match token_usage.jsonl")


def _validate_root_index(path: Path, experiment_id: str | None, errors: list[str]) -> None:
    """Validate the root index contains the experiment identity on every row."""
    for index, row in enumerate(_load_jsonl(path, errors)):
        if not isinstance(row, dict):
            errors.append(f"{path.name}[{index}] must be an object")
        elif row.get("experiment_id") != experiment_id:
            errors.append(f"{path.name}[{index}].experiment_id does not match manifest")


def _validate_row_identity(row: dict[str, Any], identity: dict[str, str | None], label: str, errors: list[str]) -> None:
    """Require each row to reference the same run identity."""
    for field in ("run_id", "task_id", "checkpoint_id", "branch"):
        if row.get(field) != identity[field]:
            errors.append(f"{label}.{field} does not match run manifest")


def _require_file(path: Path, errors: list[str]) -> None:
    """Record a missing required file."""
    if not path.is_file():
        errors.append(f"Missing required file: {path}")


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Load a JSON object and report malformed or missing files."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path} must contain a JSON object")
        return None
    return value


def _load_jsonl(path: Path, errors: list[str]) -> list[Any]:
    """Load JSONL rows while reporting malformed records."""
    if not path.is_file():
        return []
    rows: list[Any] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSONL in {path} line {number}: {exc}")
    return rows


def _required_string(value: Any, field: str, label: str, errors: list[str]) -> str | None:
    """Require a non-empty string field from one JSON object."""
    result = value.get(field) if isinstance(value, dict) else None
    if not isinstance(result, str) or not result:
        errors.append(f"{label}.{field} is required")
        return None
    return result


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one raw log file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
