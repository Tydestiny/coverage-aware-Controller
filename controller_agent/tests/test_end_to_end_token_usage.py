"""Tests for full-context runtime token ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controller_agent.results.token_usage import build_end_to_end_token_usage


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run_dir(tmp_path: Path, branch: str = "controller") -> Path:
    path = tmp_path / branch
    (path / "raw_logs").mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"branch": branch}), encoding="utf-8")
    (path / "result.json").write_text(json.dumps({"branch": branch}), encoding="utf-8")
    _write_jsonl(path / "continuation.jsonl", [
        {"step_number": 4, "agent_call_id": "agent_call_0004"},
        {"step_number": 5, "agent_call_id": "agent_call_0005"},
    ])
    _write_jsonl(path / "raw_logs" / "lm_calls.jsonl", [
        {"input": {"model": "agent"}, "output": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}},
        {"input": {"model": "agent"}, "output": {"usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}}},
    ])
    return path


def test_builds_exact_agent_and_controller_ledger(tmp_path: Path) -> None:
    """Raw call order is joined only to structured runtime call IDs."""
    path = _run_dir(tmp_path)
    _write_jsonl(path / "controller_calls.jsonl", [
        {"step_number": 5, "controller_call_id": "controller_call_0001", "model": "controller", "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}}
    ])

    report = build_end_to_end_token_usage(path)
    records = [json.loads(line) for line in (path / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [record["call_id"] for record in records] == ["agent_call_0004", "agent_call_0005", "controller_call_0001"]
    assert report.agent_total_tokens == 35
    assert report.controller_total_tokens == 9
    assert report.combined_total_tokens == 44
    assert json.loads((path / "result.json").read_text(encoding="utf-8"))["token_usage"]["token_usage_status"] == "complete"


def test_baseline_has_real_zero_controller_tokens(tmp_path: Path) -> None:
    """Baseline records zero Controller usage only when the call file is empty."""
    path = _run_dir(tmp_path, branch="baseline")
    (path / "controller_calls.jsonl").write_text("", encoding="utf-8")

    report = build_end_to_end_token_usage(path)

    assert report.controller_total_tokens == 0
    assert report.combined_total_tokens == 35


def test_rejects_unmatched_raw_and_structured_agent_calls(tmp_path: Path) -> None:
    """A count mismatch cannot be silently treated as a reliable token ledger."""
    path = _run_dir(tmp_path)
    (path / "raw_logs" / "lm_calls.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Agent call count mismatch"):
        build_end_to_end_token_usage(path)
