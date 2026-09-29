"""Evaluate one AppWorld task in a separate official evaluator process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.evaluation.official_evaluator import run_official_evaluation, update_run_summary


def parse_args() -> argparse.Namespace:
    """Parse official evaluation wrapper arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--lifecycle-error", default=None)
    return parser.parse_args()


def main() -> int:
    """Run official evaluation and update the specified run summary."""
    args = parse_args()
    lifecycle_error = None
    if args.lifecycle_error:
        lifecycle_error = {
            "known": True,
            "stage": "world.close",
            "error": args.lifecycle_error,
        }
    result = run_official_evaluation(
        Path(args.appworld_root).resolve(),
        args.experiment_name,
        args.task_id,
    )
    update_run_summary(Path(args.summary).resolve(), result, lifecycle_error)
    evaluator_path = Path(args.summary).resolve().parent / "evaluator_result.json"
    evaluator_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
