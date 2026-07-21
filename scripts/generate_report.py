"""Generate canonical-trajectory and prefix data quality reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.io_utils import read_records
from agent_error_compaction.schemas import prefix_from_dict, trajectory_from_dict
from agent_error_compaction.statistics import build_report, write_report


def main() -> None:
    """Load generated artifacts and write JSON plus Markdown statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    splits = {name: [trajectory_from_dict(record) for record in read_records(args.trajectory_dir / f"{name}.jsonl")] for name in ("train", "validation", "test") if (args.trajectory_dir / f"{name}.jsonl").exists()}
    prefixes = [prefix_from_dict(record) for path in args.prefix_dir.glob("*_prefixes.jsonl") for record in read_records(path)]
    rejected_path = args.trajectory_dir / "rejected_records.jsonl"
    rejected = list(read_records(rejected_path)) if rejected_path.exists() else []
    report = build_report(splits, prefixes, rejected)
    write_report(report, args.output_json, args.output_md)
    print(f"trajectories={report['trajectory_count']} prefixes={report['prefix_count']} rejected={report['rejected_records']['count']}")


if __name__ == "__main__":
    main()

