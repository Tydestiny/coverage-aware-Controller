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


def test_collection_module_does_not_rewrite_native_lm_call_logs(tmp_path: Path) -> None:
    """Collection preserves native logs for exact checkpoint reconstruction."""
    module = _load_collector_module()
    path = tmp_path / "lm_calls.jsonl"
    secret = "sk-abcdefghijklmnopqrstuvwxyz123"
    path.write_text(
        json.dumps({"input": {"api_key": secret, "message": secret}}) + "\n",
        encoding="utf-8",
    )

    assert not hasattr(module, "sanitize_raw_lm_call_log")
    preserved = json.loads(path.read_text(encoding="utf-8"))
    assert preserved["input"]["api_key"] == secret
    assert preserved["input"]["message"] == secret
