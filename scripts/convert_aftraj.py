"""Convert inspected AFTraj JSON/JSONL exports to canonical JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.adapters.aftraj import AFTrajAdapter
from agent_error_compaction.family_builder import assign_family
from agent_error_compaction.io_utils import read_records
from agent_error_compaction.validation import validate_trajectory
import json


def main() -> None:
    """Convert a source file and assign a family before writing canonical JSONL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--mistake-step-base", type=int, default=0, choices=(0,), help="Confirmed AFTraj-2K source index base.")
    args = parser.parse_args()
    adapter = AFTrajAdapter(args.mistake_step_base)
    stats = {"total": 0, "converted": 0, "rejected": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, args.rejected.open("w", encoding="utf-8") as rejected:
        try:
            for index, record in enumerate(read_records(args.input)):
                stats["total"] += 1
                try:
                    trajectory = assign_family(adapter.convert_record(record))
                    validate_trajectory(trajectory)
                    output.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                    stats["converted"] += 1
                except Exception as exc:
                    rejected.write(json.dumps({"record_index": index, "reason": str(exc), "raw_record": record}, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
        except Exception as exc:
            rejected.write(json.dumps({"record_index": None, "reason": str(exc), "raw_record": None}, ensure_ascii=False) + "\n")
            stats["rejected"] += 1
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
