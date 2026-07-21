"""Compare v1.0 and v1.1 JSONL trajectory/prefix invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.io_utils import read_records


def compare_trajectories(before_path: Path, after_path: Path) -> dict[str, Any]:
    """Compare order, IDs, steps, raw steps, labels, family, and split."""
    before = list(read_records(before_path))
    after = list(read_records(after_path))
    checks = {"trajectory_count": len(before) == len(after), "trajectory_order": [item.get("trajectory_id") for item in before] == [item.get("trajectory_id") for item in after], "family_ids": all(left.get("trajectory_family_id") == right.get("trajectory_family_id") for left, right in zip(before, after)), "split": all(left.get("source_split") == right.get("source", {}).get("split") for left, right in zip(before, after)), "step_counts": all(len(left.get("steps", [])) == len(right.get("steps", [])) for left, right in zip(before, after)), "source_step_indices": all([step.get("source_step_index") for step in left.get("steps", [])] == [step.get("source_step_index") for step in right.get("steps", [])] for left, right in zip(before, after)), "raw_steps": all([step.get("raw_step") for step in left.get("steps", [])] == [step.get("raw_step") for step in right.get("steps", [])] for left, right in zip(before, after)), "critical_error_steps": all((left.get("labels", {}).get("error_onset_step") if left.get("labels", {}).get("is_failure") else None) == (right.get("annotations", {}).get("primary_error", {}).get("step_index") if right.get("annotations", {}).get("primary_error") else None) for left, right in zip(before, after))}
    checks["passed"] = all(checks.values())
    return checks


def compare_prefixes(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    """Compare prefix counts and boolean error labels by prefix ID."""
    before = {record["prefix_id"]: record for path in before_dir.glob("*_prefixes.jsonl") for record in read_records(path)}
    after = {record["prefix_id"]: record for path in after_dir.glob("*_prefixes.jsonl") for record in read_records(path)}
    checks = {"prefix_count": len(before) == len(after), "prefix_ids": set(before) == set(after), "boolean_labels": all(before[key].get("error_occurred") == after[key].get("critical_error_occurred") for key in before if key in after)}
    checks["passed"] = all(checks.values())
    return checks


def main() -> None:
    """Write a migration consistency report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-prefix-dir", type=Path)
    parser.add_argument("--after-prefix-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {"trajectory": compare_trajectories(args.before, args.after)}
    if args.before_prefix_dir and args.after_prefix_dir:
        report["prefix"] = compare_prefixes(args.before_prefix_dir, args.after_prefix_dir)
    report["passed"] = all(section["passed"] for section in report.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

