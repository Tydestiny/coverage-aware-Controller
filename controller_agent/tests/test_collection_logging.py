"""Tests for persisted collection-log sanitization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_collector_module():
    """Load the collection script without depending on a scripts package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "collect_with_checkpoints.py"
    spec = importlib.util.spec_from_file_location("collection_logging_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_lm_call_log_is_redacted(tmp_path: Path) -> None:
    """Native request logs retain data but remove API key values."""
    module = _load_collector_module()
    path = tmp_path / "lm_calls.jsonl"
    secret = "sk-abcdefghijklmnopqrstuvwxyz123"
    path.write_text(
        json.dumps({"input": {"api_key": secret, "message": secret}}) + "\n",
        encoding="utf-8",
    )

    module.sanitize_raw_lm_call_log(path)

    sanitized = json.loads(path.read_text(encoding="utf-8"))
    assert sanitized["input"]["api_key"] == "[REDACTED]"
    assert sanitized["input"]["message"] == "sk-[REDACTED]"
