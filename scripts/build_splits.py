"""Build deterministic family-disjoint train/validation/test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.io_utils import read_records
from agent_error_compaction.schemas import trajectory_from_dict
from agent_error_compaction.split_builder import build_group_splits


def main() -> None:
    """Read canonical JSONL, construct splits, and write data plus ID manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    trajectories = [trajectory_from_dict(record) for record in read_records(args.input)]
    splits = build_group_splits(trajectories, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for value in values:
                payload = value.to_dict()
                payload["source"]["split"] = name
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        (args.manifest_dir / f"{name}_ids.json").write_text(json.dumps([value.trajectory_id for value in values], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(values) for name, values in splits.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
