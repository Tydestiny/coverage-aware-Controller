"""Validate the fixed restored-checkpoint branch experiment artifact contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.logging.branch_artifacts import RUN_REQUIRED_FILES, read_jsonl, sha256_file


class ValidationErrors:
    """Collect all contract violations before returning a failure."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        """Record one failed assertion."""
        if not condition:
            self.messages.append(message)


def parse_args() -> argparse.Namespace:
    """Parse the experiment root argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def main() -> int:
    """Validate one experiment and print an actionable report."""
    experiment_root = Path(parse_args().experiment_root).resolve()
    errors = validate_experiment(experiment_root)
    if errors.messages:
        for message in errors.messages:
            print(f"ERROR: {message}")
        print(f"Validation failed with {len(errors.messages)} error(s).")
        return 1
    print(f"Validated restored-checkpoint experiment: {experiment_root}")
    return 0


def validate_experiment(experiment_root: Path) -> ValidationErrors:
    """Apply cross-file, token, path, and raw-log checks."""
    errors = ValidationErrors()
    root_required = ("manifest.json", "branch_results.jsonl", "comparison_summary.jsonl")
    for name in root_required:
        errors.check((experiment_root / name).is_file(), f"missing root file: {name}")
    manifest_path = experiment_root / "manifest.json"
    if not manifest_path.exists():
        return errors
    manifest = load_json(manifest_path, errors)
    if manifest is None:
        return errors
    experiment_id = manifest.get("experiment_id")
    task_entries = manifest.get("tasks")
    errors.check(isinstance(task_entries, list) and len(task_entries) == 1, "manifest.tasks must contain one checkpoint")
    task_entry = task_entries[0] if isinstance(task_entries, list) and len(task_entries) == 1 else {}
    run_entries = manifest.get("runs")
    errors.check(isinstance(run_entries, list), "manifest.runs must be a list")
    run_entries = run_entries if isinstance(run_entries, list) else []
    expected_results: dict[str, dict[str, Any]] = {}
    for entry in run_entries:
        if not isinstance(entry, dict):
            errors.messages.append("manifest run entry is not an object")
            continue
        relative = entry.get("path")
        run_dir = experiment_root / str(relative)
        errors.check(run_dir.is_dir(), f"manifest run path does not exist: {relative}")
        if run_dir.is_dir():
            result = validate_run(
                run_dir=run_dir,
                experiment_id=experiment_id,
                task_id=task_entry.get("task_id"),
                checkpoint_id=task_entry.get("checkpoint_id"),
                expected_entry=entry,
                errors=errors,
            )
            if result is not None:
                expected_results[str(entry.get("branch"))] = result
    validate_root_summaries(experiment_root, experiment_id, expected_results, errors)
    return errors


def validate_run(
    *,
    run_dir: Path,
    experiment_id: Any,
    task_id: Any,
    checkpoint_id: Any,
    expected_entry: dict[str, Any],
    errors: ValidationErrors,
) -> dict[str, Any] | None:
    """Validate one branch directory and all of its ledgers."""
    label = run_dir.name
    for name in RUN_REQUIRED_FILES:
        errors.check((run_dir / name).is_file(), f"{label}: missing required file {name}")
    manifest = load_json(run_dir / "manifest.json", errors)
    result = load_json(run_dir / "result.json", errors)
    evaluator = load_json(run_dir / "evaluator_result.json", errors)
    if manifest is None or result is None or evaluator is None:
        return result
    identity = {
        "experiment_id": experiment_id,
        "run_id": expected_entry.get("run_id"),
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "branch": expected_entry.get("branch"),
    }
    validate_identity(manifest, identity, f"{label}/manifest.json", errors)
    validate_identity(result, identity, f"{label}/result.json", errors)
    validate_identity(evaluator, identity, f"{label}/evaluator_result.json", errors)
    errors.check(result.get("branch") in {"baseline", "controller", "oracle"}, f"{label}: invalid branch")
    errors.check(bool(manifest.get("agent_model")), f"{label}: agent_model is missing")
    errors.check(bool(manifest.get("controller_model")), f"{label}: controller_model is missing")
    errors.check(bool(manifest.get("prompt_hash")), f"{label}: prompt_hash is missing")
    errors.check(
        manifest.get("controller_enabled") is (result.get("branch") == "controller"),
        f"{label}: controller_enabled does not match branch",
    )
    required_paths = manifest.get("required_files")
    errors.check(isinstance(required_paths, list), f"{label}: manifest.required_files must be a list")
    if isinstance(required_paths, list):
        for relative in required_paths:
            errors.check((run_dir / str(relative)).is_file(), f"{label}: invalid required file path {relative}")

    continuation = read_rows(run_dir / "continuation.jsonl", errors)
    controller_calls = read_rows(run_dir / "controller_calls.jsonl", errors)
    token_rows = read_rows(run_dir / "token_usage.jsonl", errors)
    for index, row in enumerate(continuation):
        validate_identity(row, identity, f"{label}/continuation[{index}]", errors)
        errors.check(row.get("sequence_index") == index, f"{label}: continuation sequence is not contiguous at {index}")
        errors.check(bool(row.get("event_id")), f"{label}: continuation row {index} lacks event_id")
    event_ids = {row.get("event_id") for row in continuation}
    for index, row in enumerate(controller_calls):
        validate_identity(row, identity, f"{label}/controller_calls[{index}]", errors)
        errors.check(bool(row.get("controller_call_id")), f"{label}: Controller call {index} lacks controller_call_id")
        errors.check(row.get("event_id") in event_ids, f"{label}: Controller call {index} has unknown event_id")
        provenance = row.get("controller_input_provenance")
        errors.check(isinstance(row.get("controller_input"), dict), f"{label}: ControllerInput {index} is unavailable")
        errors.check(
            isinstance(provenance, dict) and provenance.get("reconstructed") is False,
            f"{label}: ControllerInput {index} is reconstructed or provenance is missing",
        )
        provider_raw = row.get("provider_raw_response")
        errors.check(
            isinstance(provider_raw, dict) and "available" in provider_raw,
            f"{label}: Controller call {index} lacks raw_response availability",
        )
    for index, row in enumerate(token_rows):
        validate_identity(row, identity, f"{label}/token_usage[{index}]", errors)
        errors.check(row.get("source") in {"agent", "controller"}, f"{label}: token row {index} has invalid source")
        errors.check(row.get("event_id") in event_ids, f"{label}: token row {index} has unknown event_id")
        expected_total = add_nullable(row.get("input_tokens"), row.get("output_tokens"))
        errors.check(row.get("total_tokens") == expected_total, f"{label}: token row {index} total mismatch")

    validate_agent_usage(run_dir, result, token_rows, errors)
    validate_controller_usage(label, result, controller_calls, token_rows, errors)
    validate_combined_usage(label, result, errors)
    validate_raw_hashes(run_dir, manifest, errors)
    evaluator_path = run_dir / str(result.get("evaluator_result_path"))
    raw_logs_path = run_dir / str(result.get("raw_logs_path"))
    errors.check(evaluator_path.is_file(), f"{label}: evaluator_result_path is invalid")
    errors.check(raw_logs_path.is_dir(), f"{label}: raw_logs_path is invalid")
    return result


def validate_agent_usage(
    run_dir: Path,
    result: dict[str, Any],
    token_rows: list[dict[str, Any]],
    errors: ValidationErrors,
) -> None:
    """Reconcile per-call Agent tokens with AppWorld's native usage file."""
    label = run_dir.name
    usage = load_json(run_dir / "raw_logs" / "usage.json", errors)
    if usage is None:
        return
    tokens = usage.get("tokens") if isinstance(usage.get("tokens"), dict) else {}
    native_input = int(tokens.get("input_cache_miss", 0)) + int(tokens.get("input_cache_hit", 0))
    native_output = tokens.get("output")
    agent_rows = [row for row in token_rows if row.get("source") == "agent"]
    ledger_input = sum_known(row.get("input_tokens") for row in agent_rows)
    ledger_output = sum_known(row.get("output_tokens") for row in agent_rows)
    errors.check(ledger_input == native_input, f"{label}: Agent input ledger {ledger_input} != usage.json {native_input}")
    errors.check(ledger_output == native_output, f"{label}: Agent output ledger {ledger_output} != usage.json {native_output}")
    agent_result = result.get("agent_tokens") if isinstance(result.get("agent_tokens"), dict) else {}
    errors.check(agent_result.get("input_tokens") == ledger_input, f"{label}: result Agent input mismatch")
    errors.check(agent_result.get("output_tokens") == ledger_output, f"{label}: result Agent output mismatch")
    lm_calls = read_rows(run_dir / "raw_logs" / "lm_calls.jsonl", errors)
    errors.check(len(lm_calls) == len(agent_rows), f"{label}: lm_calls count does not match Agent token calls")


def validate_controller_usage(
    label: str,
    result: dict[str, Any],
    controller_calls: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    errors: ValidationErrors,
) -> None:
    """Reconcile Controller call records, token ledger, and result summary."""
    controller_rows = [row for row in token_rows if row.get("source") == "controller"]
    errors.check(len(controller_rows) == len(controller_calls), f"{label}: Controller call/token row count mismatch")
    ledger_input = sum_known(row.get("input_tokens") for row in controller_rows)
    ledger_output = sum_known(row.get("output_tokens") for row in controller_rows)
    call_input = sum_known(row.get("input_tokens") for row in controller_calls)
    call_output = sum_known(row.get("output_tokens") for row in controller_calls)
    errors.check(ledger_input == call_input, f"{label}: Controller input ledgers disagree")
    errors.check(ledger_output == call_output, f"{label}: Controller output ledgers disagree")
    controller_result = result.get("controller_tokens") if isinstance(result.get("controller_tokens"), dict) else {}
    errors.check(controller_result.get("input_tokens") == ledger_input, f"{label}: result Controller input mismatch")
    errors.check(controller_result.get("output_tokens") == ledger_output, f"{label}: result Controller output mismatch")
    errors.check(result.get("controller_calls") == len(controller_calls), f"{label}: result controller_calls mismatch")


def validate_combined_usage(label: str, result: dict[str, Any], errors: ValidationErrors) -> None:
    """Check combined result usage equals Agent plus Controller."""
    agent = result.get("agent_tokens") if isinstance(result.get("agent_tokens"), dict) else {}
    controller = result.get("controller_tokens") if isinstance(result.get("controller_tokens"), dict) else {}
    combined = result.get("combined_tokens") if isinstance(result.get("combined_tokens"), dict) else {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        expected = add_nullable(agent.get(field), controller.get(field))
        errors.check(combined.get(field) == expected, f"{label}: combined {field} mismatch")


def validate_raw_hashes(run_dir: Path, manifest: dict[str, Any], errors: ValidationErrors) -> None:
    """Verify every archived raw log against the run manifest."""
    hashes = manifest.get("raw_log_sha256")
    errors.check(isinstance(hashes, dict), f"{run_dir.name}: raw_log_sha256 must be an object")
    if not isinstance(hashes, dict):
        return
    expected_names = {Path(path).name for path in RUN_REQUIRED_FILES if path.startswith("raw_logs/")}
    errors.check(set(hashes) == expected_names, f"{run_dir.name}: raw_log_sha256 keys are incomplete")
    for name, expected in hashes.items():
        path = run_dir / "raw_logs" / name
        errors.check(path.is_file(), f"{run_dir.name}: hashed raw log missing: {name}")
        if path.is_file():
            errors.check(sha256_file(path) == expected, f"{run_dir.name}: raw log hash mismatch: {name}")


def validate_root_summaries(
    experiment_root: Path,
    experiment_id: Any,
    expected_results: dict[str, dict[str, Any]],
    errors: ValidationErrors,
) -> None:
    """Require both implemented branches and one comparison row."""
    branch_rows = read_rows(experiment_root / "branch_results.jsonl", errors)
    branches = {row.get("branch") for row in branch_rows}
    errors.check({"baseline", "controller"}.issubset(branches), "branch_results must contain baseline and controller")
    for index, row in enumerate(branch_rows):
        errors.check(row.get("experiment_id") == experiment_id, f"branch_results row {index} experiment_id mismatch")
        result_path = experiment_root / str(row.get("result_path"))
        errors.check(result_path.is_file(), f"branch_results row {index} has invalid result_path")
    comparisons = read_rows(experiment_root / "comparison_summary.jsonl", errors)
    errors.check(len(comparisons) == 1, "comparison_summary must contain exactly one checkpoint row")
    if comparisons:
        row = comparisons[0]
        errors.check(row.get("experiment_id") == experiment_id, "comparison_summary experiment_id mismatch")
        errors.check(row.get("oracle_success") is None, "oracle_success must be null when oracle is not run")
        errors.check(row.get("checkpoint_recoverable") is None, "checkpoint_recoverable must be null without oracle")
        if "baseline" in expected_results:
            errors.check(
                row.get("baseline_success") == expected_results["baseline"].get("success"),
                "comparison baseline_success mismatch",
            )
        if "controller" in expected_results:
            errors.check(
                row.get("controller_success") == expected_results["controller"].get("success"),
                "comparison controller_success mismatch",
            )


def validate_identity(
    value: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    errors: ValidationErrors,
) -> None:
    """Require the shared experiment/run/checkpoint identity fields."""
    for key, expected_value in expected.items():
        errors.check(value.get(key) == expected_value, f"{label}: {key} mismatch")


def load_json(path: Path, errors: ValidationErrors) -> dict[str, Any] | None:
    """Load an object and convert parse failures into validation messages."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.messages.append(f"cannot parse {path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.messages.append(f"JSON root must be an object: {path}")
        return None
    return value


def read_rows(path: Path, errors: ValidationErrors) -> list[dict[str, Any]]:
    """Read JSONL and report malformed records."""
    try:
        rows = read_jsonl(path)
    except Exception as exc:
        errors.messages.append(f"cannot parse {path}: {type(exc).__name__}: {exc}")
        return []
    if not all(isinstance(row, dict) for row in rows):
        errors.messages.append(f"JSONL rows must be objects: {path}")
        return []
    return rows


def sum_known(values: Any) -> int | None:
    """Sum values and propagate unavailable per-call usage."""
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(int(value) for value in materialized)


def add_nullable(left: Any, right: Any) -> int | float | None:
    """Add only fully available metrics."""
    return None if left is None or right is None else left + right


if __name__ == "__main__":
    raise SystemExit(main())
