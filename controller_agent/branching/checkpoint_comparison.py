"""Build one-row-per-task comparisons for checkpoint replay experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_checkpoint_comparison(
    experiment_root: Path,
    candidates_path: Path,
    *,
    controller_branch: str = "single_intervention",
    output_prefix: str = "checkpoint_comparison",
) -> dict[str, Any]:
    """Write CSV and JSON comparison summaries for paired replay branches."""
    root = Path(experiment_root)
    rows = [
        _comparison_row(root, candidate, controller_branch=controller_branch)
        for candidate in _read_jsonl(Path(candidates_path))
    ]
    rows.sort(key=lambda row: str(row["task_id"]))
    _write_csv(root / f"{output_prefix}.csv", rows)
    summary = _summary(rows)
    summary["controller_branch"] = controller_branch
    (root / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _comparison_row(
    root: Path,
    candidate: dict[str, Any],
    *,
    controller_branch: str,
) -> dict[str, Any]:
    task_id = str(candidate["task_id"])
    checkpoint_step = int(candidate["checkpoint_step"])
    trigger_type = str(candidate["trigger_type"])
    safe_trigger = trigger_type.replace("/", "_").replace("\\", "_")
    suffix = f"step_{checkpoint_step:04d}__{safe_trigger}"
    task_root = root / "tasks" / task_id
    controller_dir = task_root / f"{controller_branch}__{suffix}"
    no_guidance_dir = task_root / f"no_guidance_replay__{suffix}"
    baseline_success = _success(Path(str(candidate["baseline_run_path"])) / "evaluator_result.json")
    controller_result = _read_json(controller_dir / "result.json")
    no_guidance_result = _read_json(no_guidance_dir / "result.json")
    controller_success = _bool_or_none(controller_result.get("success"))
    no_guidance_success = _bool_or_none(no_guidance_result.get("success"))
    calls = _read_jsonl(controller_dir / "controller_calls.jsonl")
    tokens = controller_result.get("tokens") if isinstance(controller_result.get("tokens"), dict) else {}
    return {
        "task_id": task_id,
        "checkpoint_step": checkpoint_step,
        "trigger_type": trigger_type,
        "baseline_success": baseline_success,
        "no_guidance_success": no_guidance_success,
        "controller_success": controller_success,
        "comparison_type": _comparison_type(no_guidance_success, controller_success),
        "controller_action": calls[-1].get("action") if calls else None,
        "controller_tokens": tokens.get("controller_total"),
    }


def _comparison_type(no_guidance: bool | None, controller: bool | None) -> str:
    """Classify the paired outcome without interpreting missing results as success."""
    if controller is True and no_guidance is not True:
        return "controller_only_success"
    if no_guidance is True and controller is not True:
        return "no_guidance_only_success"
    if controller is True and no_guidance is True:
        return "both_success"
    return "both_fail"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired checkpoints using only recorded outcomes and token usage."""
    total = len(rows)
    counts = {name: sum(row["comparison_type"] == name for row in rows) for name in (
        "controller_only_success", "no_guidance_only_success", "both_success", "both_fail"
    )}
    controller_success = sum(row["controller_success"] is True for row in rows)
    no_guidance_success = sum(row["no_guidance_success"] is True for row in rows)
    token_values = [row["controller_tokens"] for row in rows]
    return {
        "total_tasks": total,
        "controller_success": controller_success,
        "no_guidance_success": no_guidance_success,
        **counts,
        "net_recovery": counts["controller_only_success"] - counts["no_guidance_only_success"],
        "controller_success_rate": controller_success / total if total else None,
        "no_guidance_success_rate": no_guidance_success / total if total else None,
        "total_controller_tokens": sum(value for value in token_values if isinstance(value, int) and value >= 0),
    }


def _success(path: Path) -> bool | None:
    """Read an official success value without inferring it from logs."""
    payload = _read_json(path)
    direct = _bool_or_none(payload.get("success"))
    if direct is not None:
        return direct
    official = payload.get("official")
    task_result = official.get("task_result") if isinstance(official, dict) else None
    return _bool_or_none(task_result.get("success")) if isinstance(task_result, dict) else None


def _bool_or_none(value: Any) -> bool | None:
    """Accept only explicit boolean outcome values."""
    return value if isinstance(value, bool) else None


def _read_json(path: Path) -> dict[str, Any]:
    """Read one optional JSON object."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read optional JSONL objects."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the stable user-facing task comparison table."""
    fields = [
        "task_id", "checkpoint_step", "trigger_type", "baseline_success", "no_guidance_success",
        "controller_success", "comparison_type", "controller_action", "controller_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
