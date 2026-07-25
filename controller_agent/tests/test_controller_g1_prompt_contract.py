"""Contract tests for the latest-architecture Controller G1 prompt."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.state.schemas import Action


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "controller_agent" / "controller" / "prompts" / "g1_system.txt"


def _example_objects(prompt: str) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in prompt.splitlines()
        if line.startswith('{"action":')
    ]


def test_g1_prompt_uses_latest_three_field_contract() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    examples = _example_objects(prompt)

    assert 1500 <= len(prompt.split()) <= 2400
    assert len(examples) >= 6
    for example in examples:
        assert list(example) == ["action", "reason", "instruction"]
        assert example["action"] in {action.value for action in Action}
        assert example["reason"].strip()
        if example["action"] == Action.PROCEED.value:
            assert example["instruction"] == ""
        else:
            assert example["instruction"].strip()
            assert len(example["instruction"].split()) <= 50


def test_g1_prompt_is_before_finish_specific_and_never_examples_risk_block() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    examples = _example_objects(prompt)

    assert "only immediately before" in prompt
    assert "BLOCK_RISKY_ACTION is forbidden" in prompt
    assert all(example["action"] != Action.BLOCK_RISKY_ACTION.value for example in examples)


def test_g1_prompt_requires_diagnosis_and_self_contained_minimal_guidance() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    for status in ("SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED", "STALE"):
        assert status in prompt
    assert "specific visible mismatch" in prompt
    assert "immediate minimal correction" in prompt
    assert "The instruction is the only task-specific advice guaranteed to enter" in prompt
    assert "Do not output this diagnostic procedure" in prompt
