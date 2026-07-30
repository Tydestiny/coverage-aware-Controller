"""Scan failed Baseline trajectories for causal single-intervention points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.single_intervention import scan_failed_baselines, scan_successful_baselines


def parse_args() -> argparse.Namespace:
    """Parse scan paths without requiring an AppWorld or model process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument(
        "--first-per-task",
        action="store_true",
        help="Keep only the earliest causal Trigger candidate for each failed task.",
    )
    parser.add_argument(
        "--successful-baselines",
        action="store_true",
        help="Scan successful rather than failed baseline trajectories.",
    )
    return parser.parse_args()


def main() -> int:
    """Write candidate JSONL and a small scan summary."""
    args = parse_args()
    scanner = scan_successful_baselines if args.successful_baselines else scan_failed_baselines
    candidates, summary = scanner(Path(args.experiment_root), appworld_root=Path(args.appworld_root))
    if args.first_per_task:
        selected = {}
        for candidate in candidates:
            existing = selected.get(candidate.task_id)
            if existing is None or (candidate.baseline_event_index, candidate.checkpoint_step) < (
                existing.baseline_event_index,
                existing.checkpoint_step,
            ):
                selected[candidate.task_id] = candidate
        candidates = [selected[task_id] for task_id in sorted(selected)]
        summary["selection_policy"] = "first_trigger_per_task"
        summary["selected_checkpoint_count"] = len(candidates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for item in candidates),
        encoding="utf-8",
    )
    summary_path = Path(args.summary_output) if args.summary_output else output.with_name("checkpoint_scan_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
