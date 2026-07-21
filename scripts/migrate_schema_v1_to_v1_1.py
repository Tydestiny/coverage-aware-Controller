"""Migrate existing Canonical Trajectory JSONL from v1.0 to v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.migration import migrate_file


def main() -> None:
    """Run the streaming schema migration and write statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    args = parser.parse_args()
    stats = migrate_file(args.input, args.output, args.rejected)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()

