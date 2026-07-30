"""Rebuild Agent and Controller token ledgers for full-context experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.results.token_usage import build_end_to_end_token_usage


def parse_args() -> argparse.Namespace:
    """Parse one experiment root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Rebuild all complete branch token ledgers under the supplied root."""
    root = parse_args().experiment_root.resolve()
    failures: list[str] = []
    completed = 0
    for result_path in sorted(root.rglob("result.json")):
        if any(part in {"interrupted_attempts", "invalid_prompt_path_attempts"} for part in result_path.parts):
            continue
        try:
            build_end_to_end_token_usage(result_path.parent)
            completed += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{result_path.parent}: {type(exc).__name__}: {exc}")
    print(f"completed={completed}")
    for failure in failures:
        print(f"ERROR: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
