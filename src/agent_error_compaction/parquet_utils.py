"""Streaming Parquet-to-JSONL conversion utilities."""

from __future__ import annotations

import base64
from datetime import date, datetime, time
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


def _json_safe(value: Any) -> Any:
    """Recursively convert Arrow scalar values to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return {"__encoding__": "base64", "data": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported Parquet value type: {type(value).__name__}")


def _sha256(path: Path) -> str:
    """Compute a source-file digest for the conversion manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_parquet_files(input_paths: Iterable[Path], output_path: Path, batch_size: int = 1024) -> dict[str, Any]:
    """Stream Parquet rows to one JSONL file and return a deterministic manifest."""
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one Parquet input is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in paths:
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Input must use the .parquet extension: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Parquet input does not exist: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifests: list[dict[str, Any]] = []
    total_rows = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            rows_written = 0
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                for record in batch.to_pylist():
                    output.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
                    rows_written += 1
            expected_rows = parquet_file.metadata.num_rows
            if rows_written != expected_rows:
                raise RuntimeError(f"Row-count mismatch for {path}: expected {expected_rows}, wrote {rows_written}")
            total_rows += rows_written
            source_manifests.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "row_count": rows_written,
                    "columns": parquet_file.schema_arrow.names,
                    "schema": str(parquet_file.schema_arrow),
                }
            )
    return {
        "format": "parquet_to_jsonl_manifest/v1",
        "output": str(output_path),
        "batch_size": batch_size,
        "source_count": len(paths),
        "total_rows": total_rows,
        "sources": source_manifests,
    }

