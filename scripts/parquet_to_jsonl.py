"""Convert one or more Parquet files to a single JSONL source export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.parquet_utils import convert_parquet_files


def main() -> None:
    """Run streaming Parquet conversion and write its deterministic manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    manifest = convert_parquet_files(args.input, args.output, args.batch_size)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sources": manifest["source_count"], "rows": manifest["total_rows"], "output": str(args.output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

