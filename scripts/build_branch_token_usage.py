"""Build and validate token-usage ledgers for one branching experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.token_usage import build_experiment_token_usage, validate_experiment_token_usage


def parse_args() -> argparse.Namespace:
    """Parse the experiment root to process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=REPO_ROOT / "results" / "branching_experiment" / "legacy_branching_v1",
    )
    parser.add_argument("--check", action="store_true", help="Validate only; do not write ledgers.")
    return parser.parse_args()


def main() -> int:
    """Build or validate deterministic prefix and continuation token ledgers."""
    args = parse_args()
    root = args.experiment_root.resolve()
    if args.check:
        validation = validate_experiment_token_usage(root)
        print(f"checked_branches={validation.checked_branches}")
        for error in validation.errors:
            print(f"ERROR: {error}")
        return 1 if validation.errors else 0
    report = build_experiment_token_usage(root)
    validation = validate_experiment_token_usage(root)
    print(f"prefix_ledgers={report.prefix_ledgers}")
    print(f"branch_ledgers={report.branch_ledgers}")
    for error in validation.errors:
        print(f"ERROR: {error}")
    return 1 if validation.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
