"""Script to convert Who&When Pro JSON clean files into canonical JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_error_compaction.adapters.whowhenpro import WhoWhenProAdapter
from src.agent_error_compaction.io_utils import write_jsonl


def main() -> None:
    raw_dir = Path("data/raw/who_when_pro")
    if not raw_dir.exists():
        raw_dir = Path("data/raw/whowhenpro")

    output_file = Path("data/processed/whowhenpro_all.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Error: Raw directory does not exist.")
        return

    json_files = list(raw_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON records in {raw_dir}")

    adapter = WhoWhenProAdapter()
    converted_records = []
    failed_count = 0

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            canonical_traj = adapter.convert_record(record)
            converted_records.append(canonical_traj.to_dict())
        except Exception as exc:
            failed_count += 1
            print(f"Warning: Failed to convert {file_path.name}: {exc}")

    write_jsonl(output_file, converted_records)
    print(f"Successfully converted {len(converted_records)} Who&When Pro records to {output_file}")
    if failed_count > 0:
        print(f"Skipped {failed_count} invalid records.")


if __name__ == "__main__":
    main()