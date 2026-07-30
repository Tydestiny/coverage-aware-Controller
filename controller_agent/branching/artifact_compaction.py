"""Compact completed checkpoint branches without losing audit evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_TRANSIENT_ROOT_FILES = (
    "config.json",
    "checkpoint_info.json",
    "turn_events.jsonl",
    "sanitized_messages.jsonl",
    "restore_result.json",
    "lifecycle_errors.json",
    "runtime_error.json",
)
_REDUNDANT_RAW_LOGS = ("environment_io.md", "usage.json")


def compact_completed_branch(branch_dir: Path) -> None:
    """Merge transient metadata into durable summaries and remove duplicates."""
    branch_dir = Path(branch_dir)
    manifest = _read_json(branch_dir / "manifest.json")
    checkpoint_info = _read_json(branch_dir / "checkpoint_info.json")
    if checkpoint_info:
        manifest["checkpoint"] = checkpoint_info
    _write_json(branch_dir / "manifest.json", manifest)

    result = _read_json(branch_dir / "result.json")
    restore = _read_json(branch_dir / "restore_result.json")
    lifecycle = _read_json(branch_dir / "lifecycle_errors.json")
    runtime_error = _read_json(branch_dir / "runtime_error.json")
    if restore:
        result["restore"] = restore
    if lifecycle:
        result["lifecycle_errors"] = lifecycle.get("errors", lifecycle)
    if runtime_error:
        result["runtime_error"] = runtime_error
    _write_json(branch_dir / "result.json", result)

    for name in _TRANSIENT_ROOT_FILES:
        (branch_dir / name).unlink(missing_ok=True)
    for name in _REDUNDANT_RAW_LOGS:
        (branch_dir / "raw_logs" / name).unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one optional JSON object."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one stable JSON object."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
