"""Analyze paired baseline and Controller results without changing experiment data."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


BRANCHES = frozenset({"baseline", "controller"})


@dataclass(frozen=True)
class RunRecord:
    """One completed branch result and its adjacent audit files."""

    path: Path
    root_manifest: dict[str, Any]
    manifest: dict[str, Any]
    result: dict[str, Any]
    evaluator: dict[str, Any]
    branch: str
    task_id: str
    split: str | None
    run_id: str | None
    seed: str | None


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from an optional JSONL ledger."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _first_value(*mappings: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
    return None


def _as_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def discover_runs(experiment_root: Path) -> list[RunRecord]:
    """Discover baseline and Controller result directories under an experiment root."""
    root_manifest_path = experiment_root / "manifest.json"
    root_manifest = read_json(root_manifest_path) if root_manifest_path.is_file() else {}
    records: list[RunRecord] = []
    for result_path in experiment_root.rglob("result.json"):
        path = result_path.parent
        if any(part in {"interrupted_attempts", "invalid_prompt_path_attempts"} for part in path.parts):
            continue
        manifest_path = path / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        result = read_json(result_path)
        branch = _as_text(_first_value(manifest, result, keys=("branch",)))
        if branch not in BRANCHES:
            continue
        task_id = _as_text(_first_value(manifest, result, root_manifest, keys=("task_id",)))
        if not task_id:
            continue
        evaluator_path = path / "evaluator_result.json"
        evaluator = read_json(evaluator_path) if evaluator_path.is_file() else {}
        split = _as_text(_first_value(manifest, result, root_manifest, keys=("split", "dataset_split")))
        run_id = _as_text(_first_value(manifest, result, root_manifest, keys=("run_id",)))
        seed = _as_text(_first_value(manifest, result, root_manifest, keys=("seed", "random_seed")))
        records.append(RunRecord(path, root_manifest, manifest, result, evaluator, branch, task_id, split, run_id, seed))
    return records


def discover_unpaired_results(experiment_root: Path) -> list[dict[str, Any]]:
    """Report branch results that lack the identity needed for a valid pair."""
    root_manifest_path = experiment_root / "manifest.json"
    root_manifest = read_json(root_manifest_path) if root_manifest_path.is_file() else {}
    issues: list[dict[str, Any]] = []
    for result_path in experiment_root.rglob("result.json"):
        path = result_path.parent
        if any(part in {"interrupted_attempts", "invalid_prompt_path_attempts"} for part in path.parts):
            continue
        manifest_path = path / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        result = read_json(result_path)
        branch = _as_text(_first_value(manifest, result, keys=("branch",)))
        if branch not in BRANCHES:
            continue
        task_id = _as_text(_first_value(manifest, result, root_manifest, keys=("task_id",)))
        if not task_id:
            issues.append({"path": str(path), "branch": branch, "reason": "missing_task_id"})
    return issues


def pair_key(record: RunRecord) -> tuple[str | None, str, str]:
    """Return the documented split/task/run-or-seed pairing key."""
    identity = f"run:{record.run_id}" if record.run_id is not None else f"seed:{record.seed}" if record.seed is not None else "none"
    return record.split, record.task_id, identity


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def evaluator_metrics(evaluator: dict[str, Any]) -> dict[str, Any]:
    """Extract only official evaluator metrics, returning null for absent fields."""
    official = evaluator.get("official") if isinstance(evaluator.get("official"), dict) else evaluator
    task_result = official.get("task_result") if isinstance(official.get("task_result"), dict) else {}
    aggregate = official.get("aggregate") if isinstance(official.get("aggregate"), dict) else {}
    score = _number(_first_value(official, task_result, aggregate, keys=("score", "task_goal_completion")))
    success_value = _first_value(official, task_result, keys=("success",))
    success = success_value if isinstance(success_value, bool) else (score == 100 if score is not None else None)
    passes = _first_value(official, task_result, keys=("passed_checks", "passes"))
    passed = len(passes) if isinstance(passes, list) else _number(passes)
    total = _number(_first_value(official, task_result, keys=("total_checks", "num_tests", "total")))
    return {"success": success, "score": score, "passed": passed, "total": total}


def _sum_tokens(rows: list[dict[str, Any]], source: str) -> float | int | None:
    matching = [row for row in rows if row.get("source") == source]
    if not matching:
        return None
    values = [_number(row.get("total_tokens")) for row in matching]
    return None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]


def token_metrics(record: RunRecord) -> dict[str, float | int | None]:
    """Read Agent ledger tokens and Controller-call usage without guessing missing data."""
    ledger = read_jsonl(record.path / "token_usage.jsonl")
    agent = _sum_tokens(ledger, "agent")
    calls = read_jsonl(record.path / "controller_calls.jsonl")
    if record.branch == "baseline" and not calls:
        controller: float | int | None = 0
    elif not calls:
        controller = None
    else:
        values = []
        for call in calls:
            usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
            total = _number(usage.get("total_tokens"))
            if total is None:
                inputs, outputs = _number(usage.get("input_tokens")), _number(usage.get("output_tokens"))
                total = inputs + outputs if inputs is not None and outputs is not None else None
            values.append(total)
        controller = None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]
    total = None if agent is None or controller is None else agent + controller
    return {"agent": agent, "controller": controller, "total": total}


def real_controller_calls(record: RunRecord) -> list[dict[str, Any]]:
    """Return successful runtime Controller calls, excluding fail-open fallback rows."""
    return [
        row
        for row in read_jsonl(record.path / "controller_calls.jsonl")
        if row.get("fail_open") is not True and row.get("record_origin", "runtime_direct") == "runtime_direct"
    ]


def outcome_category(baseline: bool | None, controller: bool | None) -> str | None:
    """Classify paired official outcomes when both evaluator outcomes are available."""
    if baseline is None or controller is None:
        return None
    return {
        (False, True): "fail_to_success",
        (True, False): "success_to_fail",
        (True, True): "success_to_success",
        (False, False): "fail_to_fail",
    }[(baseline, controller)]


def _delta(left: float | int | None, right: float | int | None) -> float | int | None:
    return None if left is None or right is None else right - left


def _models(baseline: RunRecord, controller: RunRecord) -> tuple[str | None, str | None]:
    agent = _as_text(_first_value(controller.manifest, controller.result, controller.root_manifest, baseline.manifest, baseline.result, baseline.root_manifest, keys=("agent_model",)))
    model = _as_text(_first_value(controller.manifest, controller.result, controller.root_manifest, keys=("controller_model",)))
    return agent, model


def paired_task_row(baseline: RunRecord, controller: RunRecord) -> dict[str, Any]:
    """Build one paired baseline-versus-Controller task report."""
    baseline_eval, controller_eval = evaluator_metrics(baseline.evaluator), evaluator_metrics(controller.evaluator)
    baseline_tokens, controller_tokens = token_metrics(baseline), token_metrics(controller)
    agent_model, controller_model = _models(baseline, controller)
    calls = real_controller_calls(controller)
    return {
        "task_id": baseline.task_id,
        "split": baseline.split,
        "run_id": baseline.run_id,
        "seed": baseline.seed,
        "baseline_success": baseline_eval["success"],
        "controller_success": controller_eval["success"],
        "outcome_category": outcome_category(baseline_eval["success"], controller_eval["success"]),
        "baseline_score": baseline_eval["score"],
        "controller_score": controller_eval["score"],
        "score_delta": _delta(baseline_eval["score"], controller_eval["score"]),
        "baseline_passed_checks": baseline_eval["passed"],
        "controller_passed_checks": controller_eval["passed"],
        "passed_checks_delta": _delta(baseline_eval["passed"], controller_eval["passed"]),
        "total_checks": controller_eval["total"] if controller_eval["total"] is not None else baseline_eval["total"],
        "controller_triggered": bool(calls),
        "controller_call_count": len(calls),
        "baseline_agent_tokens": baseline_tokens["agent"],
        "controller_agent_tokens": controller_tokens["agent"],
        "controller_tokens": controller_tokens["controller"],
        "baseline_total_tokens": baseline_tokens["total"],
        "controller_total_tokens": controller_tokens["total"],
        "token_delta": _delta(baseline_tokens["total"], controller_tokens["total"]),
        "agent_model": agent_model,
        "controller_model": controller_model,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if _number(row.get(field)) is not None]
    return mean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired task reports with null-aware arithmetic."""
    def successes(field: str) -> int:
        return sum(row.get(field) is True for row in rows)

    baseline_success, controller_success = successes("baseline_success"), successes("controller_success")
    categories = {name: sum(row.get("outcome_category") == name for row in rows) for name in ("fail_to_success", "success_to_fail", "success_to_success", "fail_to_fail")}
    baseline_total = _sum_field(rows, "baseline_total_tokens")
    controller_agent_total = _sum_field(rows, "controller_agent_tokens")
    controller_total = _sum_field(rows, "controller_tokens")
    controller_branch_total = _sum_field(rows, "controller_total_tokens")
    return {
        "task_count": len(rows),
        "baseline_success_count": baseline_success,
        "controller_success_count": controller_success,
        "baseline_success_rate": baseline_success / len(rows) if rows else None,
        "controller_success_rate": controller_success / len(rows) if rows else None,
        "success_rate_delta": (controller_success - baseline_success) / len(rows) if rows else None,
        "outcome_categories": categories,
        "baseline_mean_score": _mean(rows, "baseline_score"),
        "controller_mean_score": _mean(rows, "controller_score"),
        "mean_score_delta": _mean(rows, "score_delta"),
        "triggered_task_count": sum(row["controller_triggered"] for row in rows),
        "trigger_rate": sum(row["controller_triggered"] for row in rows) / len(rows) if rows else None,
        "triggered_tasks_baseline_success_count": sum(row["controller_triggered"] and row["baseline_success"] is True for row in rows),
        "triggered_tasks_controller_success_count": sum(row["controller_triggered"] and row["controller_success"] is True for row in rows),
        "baseline_total_tokens": baseline_total,
        "controller_agent_total_tokens": controller_agent_total,
        "controller_total_tokens": controller_total,
        "controller_branch_total_tokens": controller_branch_total,
        "total_token_overhead": _delta(baseline_total, controller_branch_total),
        "mean_controller_calls": mean([row["controller_call_count"] for row in rows]) if rows else None,
    }


def _sum_field(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [_number(row.get(field)) for row in rows]
    return None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]


def safe_component(value: str | None) -> str:
    """Make a stable filename component from a run identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "default")


def analyze(experiment_root: Path, output_dir: Path) -> dict[str, Any]:
    """Pair runs, write per-task reports, and return the full analysis payload."""
    records = discover_runs(experiment_root)
    grouped: dict[tuple[str | None, str, str], dict[str, list[RunRecord]]] = {}
    for record in records:
        grouped.setdefault(pair_key(record), {}).setdefault(record.branch, []).append(record)
    rows: list[dict[str, Any]] = []
    quality: dict[str, list[dict[str, Any]]] = {
        "missing_baseline": [],
        "missing_controller": [],
        "duplicate_branches": [],
        "unpaired": discover_unpaired_results(experiment_root),
    }
    for key, branches in grouped.items():
        baseline, controller = branches.get("baseline", []), branches.get("controller", [])
        context = {"split": key[0], "task_id": key[1], "pair_id": key[2], "baseline_count": len(baseline), "controller_count": len(controller)}
        if len(baseline) > 1 or len(controller) > 1:
            quality["duplicate_branches"].append(context)
            continue
        if not baseline:
            quality["missing_baseline"].append(context)
            continue
        if not controller:
            quality["missing_controller"].append(context)
            continue
        rows.append(paired_task_row(baseline[0], controller[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for row in rows:
        identifier = row["run_id"] or row["seed"] or "default"
        (tasks_dir / f"{safe_component(row['task_id'])}__{safe_component(str(identifier))}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"overall": summarize(rows)}
    for split in ("normal", "challenge"):
        split_rows = [row for row in rows if row["split"] == split]
        if split_rows:
            summary[split] = summarize(split_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "data_quality.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "data_quality": quality, "rows": rows}


def parse_args() -> argparse.Namespace:
    """Parse analysis paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the read-only experiment analysis."""
    args = parse_args()
    payload = analyze(args.experiment_root.resolve(), args.output_dir.resolve())
    print(json.dumps(payload["summary"]["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
