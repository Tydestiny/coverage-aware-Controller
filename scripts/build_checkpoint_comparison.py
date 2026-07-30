"""Create paired Controller-versus-no-guidance checkpoint comparison files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.branching.checkpoint_comparison import write_checkpoint_comparison


def main() -> int:
    """Build the durable CSV and JSON summary without running any model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--controller-branch",
        default="single_intervention",
        choices=("single_intervention", "continuous_controller"),
    )
    parser.add_argument("--output-prefix", default="checkpoint_comparison")
    args = parser.parse_args()
    summary = write_checkpoint_comparison(
        Path(args.experiment_root),
        Path(args.candidates),
        controller_branch=args.controller_branch,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
