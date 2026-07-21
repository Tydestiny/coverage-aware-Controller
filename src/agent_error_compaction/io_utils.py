"""Streaming JSON and JSONL I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object records from a JSON array/object or JSONL file."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported input format {path.suffix!r}; expected .json or .jsonl")
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".jsonl":
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}, got {type(record).__name__}")
                yield record
            return
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    records = payload if isinstance(payload, list) else [payload]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at item {index}, got {type(record).__name__}")
        yield record


def write_jsonl(path: Path, records: Iterator[dict[str, Any]]) -> int:
    """Write records as UTF-8 JSONL and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records into memory for bounded downstream construction."""
    return list(read_records(path))

