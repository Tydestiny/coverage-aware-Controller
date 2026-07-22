"""Script to convert raw AFTraj files into canonical JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_error_compaction.adapters.aftraj import AFTrajAdapter
from src.agent_error_compaction.io_utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AFTraj raw files to canonical JSONL")
    parser.add_argument("--input", type=str, default="data/raw/aftraj", help="Input file or directory")
    parser.add_argument("--output", type=str, default="data/processed/aftraj/aftraj_all.jsonl", help="Output file")
    parser.add_argument("--rejected", type=str, default="data/processed/aftraj/rejected_records.jsonl", help="Rejected records file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return

    if input_path.is_dir():
        raw_files = list(input_path.rglob("*.jsonl")) + list(input_path.rglob("*.json"))
    else:
        raw_files = [input_path]

    print(f"Found {len(raw_files)} raw files for AFTraj in {input_path}")

    adapter = AFTrajAdapter()
    converted_records = []
    rejected_records = []

    for file_path in raw_files:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    canonical_traj = adapter.convert_record(record)
                    converted_records.append(canonical_traj.to_dict())
                except Exception as exc:
                    rejected_records.append({"file": file_path.name, "line": line_no, "error": str(exc), "raw": line})

    write_jsonl(output_file, converted_records)
    if rejected_records:
        write_jsonl(Path(args.rejected), rejected_records)

    print(f"Successfully converted {len(converted_records)} AFTraj records to {output_file}")
    print(f"Rejected {len(rejected_records)} records (saved to {args.rejected})")


if __name__ == "__main__":
    main()