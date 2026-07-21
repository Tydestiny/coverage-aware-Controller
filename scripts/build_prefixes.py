"""Build causally valid prefixes from split canonical JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.io_utils import read_records
from agent_error_compaction.prefix_builder import DEFAULT_ONSET_WINDOW, build_prefixes
from agent_error_compaction.schemas import trajectory_from_dict


def _parse_window(value: str) -> tuple[int, ...]:
    """Parse a comma-separated onset window."""
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window must contain comma-separated integers") from exc


def main() -> None:
    """Create one prefix JSONL file per available trajectory split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "onset_window"), default="all")
    parser.add_argument("--window", type=_parse_window, default=DEFAULT_ONSET_WINDOW)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source = args.input_dir / f"{split}.jsonl"
        if not source.exists():
            continue
        trajectories = [trajectory_from_dict(record) for record in read_records(source)]
        prefixes = build_prefixes(trajectories, args.mode, args.window, args.seed)
        with (args.output_dir / f"{split}_prefixes.jsonl").open("w", encoding="utf-8") as handle:
            for prefix in prefixes:
                handle.write(json.dumps(prefix.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        counts[split] = len(prefixes)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

