"""Script to convert raw AgentRx JSONL files into canonical JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_error_compaction.adapters.agentrx import AgentRxAdapter
from src.agent_error_compaction.io_utils import write_jsonl


def main() -> None:
    # 调整为正确的 raw 与 processed 子目录
    raw_dir = Path("data/raw/agentrx")
    output_file = Path("data/processed/agentrx/agentrx_all.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Error: Directory {raw_dir} does not exist.")
        return

    # 递归扫描 agentrx 文件夹下所有 JSON / JSONL
    raw_files = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.json"))
    print(f"Found {len(raw_files)} raw files in {raw_dir}")

    adapter = AgentRxAdapter()
    converted_records = []
    failed_count = 0

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
                    failed_count += 1
                    print(f"Warning: Failed line {line_no} in {file_path.name}: {exc}")

    write_jsonl(output_file, converted_records)
    print(f"Successfully converted {len(converted_records)} AgentRx records to {output_file}")
    if failed_count > 0:
        print(f"Skipped {failed_count} invalid records.")


if __name__ == "__main__":
    main()