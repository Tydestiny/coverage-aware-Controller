"""Discovery and aggregation helpers for single-intervention experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from controller_agent.state.schemas import TriggerContext
from controller_agent.triggers.composite import CompositeTriggerDetectorWithBudget
from controller_agent.triggers.config import TriggerBudgetConfig, TriggerConfig


@dataclass(frozen=True)
class SingleInterventionCandidate:
    """One causal pre-action Controller intervention point."""

    task_id: str
    checkpoint_step: int
    checkpoint_id: str
    trigger_type: str
    trigger_metadata: dict[str, Any]
    proposed_execution: str
    agent_call_id: str | None
    baseline_event_index: int
    baseline_run_path: str
    source_appworld_task_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible candidate record."""
        return asdict(self)


def scan_failed_baselines(
    experiment_root: Path,
    *,
    appworld_root: Path | None = None,
) -> tuple[list[SingleInterventionCandidate], dict[str, Any]]:
    """Scan every failed baseline branch under an end-to-end experiment root."""
    return _scan_baselines(experiment_root, success=False, appworld_root=appworld_root)


def scan_successful_baselines(
    experiment_root: Path,
    *,
    appworld_root: Path | None = None,
) -> tuple[list[SingleInterventionCandidate], dict[str, Any]]:
    """Scan every successful baseline branch under an end-to-end experiment root."""
    return _scan_baselines(experiment_root, success=True, appworld_root=appworld_root)


def _scan_baselines(
    experiment_root: Path,
    *,
    success: bool,
    appworld_root: Path | None,
) -> tuple[list[SingleInterventionCandidate], dict[str, Any]]:
    """Scan baseline branches with one explicit official-evaluator outcome."""
    candidates: list[SingleInterventionCandidate] = []
    selected_runs = 0
    scanned_runs = 0
    for baseline_dir in sorted(Path(experiment_root).glob("tasks/*/*__baseline")):
        scanned_runs += 1
        if _baseline_success(baseline_dir) is not success:
            continue
        selected_runs += 1
        candidates.extend(scan_baseline_run(baseline_dir, appworld_root=appworld_root))
    counts = Counter(candidate.trigger_type for candidate in candidates)
    outcome_label = "successful" if success else "failed"
    return candidates, {
        "experiment_root": str(Path(experiment_root).resolve()),
        "baseline_runs_scanned": scanned_runs,
        f"{outcome_label}_baseline_count": selected_runs,
        "checkpoint_count": len(candidates),
        "trigger_counts": dict(sorted(counts.items())),
    }


def scan_baseline_run(
    baseline_dir: Path,
    *,
    appworld_root: Path | None = None,
) -> list[SingleInterventionCandidate]:
    """Find first-priority trigger events in one failed baseline trajectory."""
    baseline_dir = Path(baseline_dir)
    manifest = _read_json(baseline_dir / "manifest.json")
    task_id = str(manifest.get("task_id") or baseline_dir.parent.name)
    source_path = _source_appworld_task_path(manifest, task_id, appworld_root)
    events = _read_jsonl(baseline_dir / "continuation.jsonl")
    detector = _unlimited_detector()
    detector.reset()
    history: list[dict[str, Any]] = []
    candidates: list[SingleInterventionCandidate] = []
    last_outputs: list[Any] = []
    for event_index, event in enumerate(events):
        code = event.get("generated_code")
        step = event.get("step_number")
        if not isinstance(code, str) or not code.strip() or not isinstance(step, int):
            _append_history(history, event)
            last_outputs = _as_list(event.get("environment_observation"))
            continue
        context = TriggerContext(
            task_id=task_id,
            step_index=step,
            task="",
            proposed_execution=code,
            last_execution_outputs=list(last_outputs),
            recent_history=list(history),
            runtime_state={"scan_mode": True},
        )
        result = detector.detect(context)
        if result.should_call_controller:
            candidates.append(
                SingleInterventionCandidate(
                    task_id=task_id,
                    checkpoint_step=step,
                    checkpoint_id=f"{task_id}_pre_action_{step:04d}",
                    trigger_type=result.reason,
                    trigger_metadata=dict(result.metadata),
                    proposed_execution=code,
                    agent_call_id=_optional_str(event.get("agent_call_id")),
                    baseline_event_index=event_index,
                    baseline_run_path=str(baseline_dir.resolve()),
                    source_appworld_task_path=None if source_path is None else str(source_path),
                )
            )
        _append_history(history, event)
        last_outputs = _as_list(event.get("environment_observation"))
    return candidates


def branch_directory(task_dir: Path, candidate: SingleInterventionCandidate) -> Path:
    """Return the collision-free directory reserved for one candidate branch."""
    safe_trigger = candidate.trigger_type.replace("/", "_").replace("\\", "_")
    return Path(task_dir) / f"single_intervention__step_{candidate.checkpoint_step:04d}__{safe_trigger}"


def summarize_single_interventions(root: Path) -> dict[str, Any]:
    """Aggregate completed single-intervention results without inferring values."""
    results = [
        _read_json(path)
        for path in sorted(Path(root).glob("tasks/*/single_intervention__*/result.json"))
    ]
    completed = [result for result in results if result.get("status") == "completed"]
    successful = [result for result in completed if result.get("success") is True]
    by_trigger: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        trigger = str(result.get("trigger_type") or "unknown")
        bucket = by_trigger.setdefault(trigger, {"branch_count": 0, "success_count": 0})
        bucket["branch_count"] += 1
        bucket["success_count"] += int(result.get("success") is True)
        by_task[str(result.get("task_id") or "unknown")].append(result)
    for bucket in by_trigger.values():
        bucket["success_rate"] = _ratio(bucket["success_count"], bucket["branch_count"])
    restored_tasks = sum(any(item.get("restore_success") is True for item in items) for items in by_task.values())
    return {
        "branch_count": len(results),
        "completed_branch_count": len(completed),
        "single_intervention_success_count": len(successful),
        "single_intervention_success_rate": _ratio(len(successful), len(completed)),
        "trigger_results": dict(sorted(by_trigger.items())),
        "task_count": len(by_task),
        "tasks_with_at_least_one_restored_checkpoint": restored_tasks,
        "average_agent_tokens": _mean_path(results, ("tokens", "agent_total")),
        "average_controller_tokens": _mean_path(results, ("tokens", "controller_total")),
    }


def _baseline_failed(baseline_dir: Path) -> bool:
    """Return whether the official evaluator marked one baseline as failed."""
    return _baseline_success(baseline_dir) is False


def _baseline_success(baseline_dir: Path) -> bool | None:
    """Return the official baseline outcome when it is available."""
    evaluator = _read_json(baseline_dir / "evaluator_result.json")
    official = evaluator.get("official")
    if not isinstance(official, dict):
        official = evaluator.get("official_evaluator")
    task_result = official.get("task_result") if isinstance(official, dict) else None
    if isinstance(task_result, dict) and isinstance(task_result.get("success"), bool):
        return task_result["success"]
    if isinstance(evaluator.get("success"), bool):
        return evaluator["success"]
    return None


def _source_appworld_task_path(
    manifest: dict[str, Any], task_id: str, appworld_root: Path | None
) -> Path | None:
    if appworld_root is None:
        return None
    experiment = manifest.get("appworld_experiment")
    if not isinstance(experiment, str) or not experiment:
        return None
    path = Path(appworld_root) / "experiments" / "outputs" / experiment / "tasks" / task_id
    return path if path.is_dir() else None


def _unlimited_detector() -> CompositeTriggerDetectorWithBudget:
    """Create a scanner dispatcher with the live detector order and no cap."""
    config = TriggerConfig(
        budget=TriggerBudgetConfig(
            detector_order=("before_finish", "before_write", "no_progress"),
            max_triggers_per_task=1_000_000,
            min_before_finish_triggers=0,
        )
    )
    return CompositeTriggerDetectorWithBudget(full_config=config)


def _append_history(history: list[dict[str, Any]], event: dict[str, Any]) -> None:
    history.append(
        {
            "proposed_execution": str(event.get("generated_code") or ""),
            "last_execution_outputs": _as_list(event.get("environment_observation")),
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else ([] if value is None else [value])


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean_path(records: Iterable[dict[str, Any]], path: tuple[str, str]) -> float | None:
    values: list[float] = []
    for record in records:
        nested = record.get(path[0])
        value = nested.get(path[1]) if isinstance(nested, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return None if not values else sum(values) / len(values)
