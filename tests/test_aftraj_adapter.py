import json

import pytest

from agent_error_compaction.adapters.aftraj import AFTrajAdapter


def record(mistake_step=None, action="call"):
    return {"conv_id": "case", "domain": "demo", "task": "task", "gold_answer": "answer", "turns": [{"actor": "assistant", "content": "first", "action": action}, {"actor": "tool", "observation": "result"}], "mistake_step": mistake_step, "mistake_agent": "assistant", "mistake_reason": "provided"}


def test_converts_safe_trajectory():
    trajectory = AFTrajAdapter().convert_record(record())
    assert trajectory.outcome == "success"
    assert trajectory.labels.error_onset_step is None


def test_converts_failed_trajectory():
    trajectory = AFTrajAdapter().convert_record(record(1))
    assert trajectory.outcome == "failure"
    assert trajectory.labels.error_onset_step == 1


def test_converts_zero_based_label():
    assert AFTrajAdapter(0).convert_record(record(1)).labels.error_onset_step == 1


def test_converts_one_based_label():
    assert AFTrajAdapter(1).convert_record(record(1)).labels.error_onset_step == 0


def test_retains_raw_step_completely():
    source = record(0, {"name": "tool", "arguments": {"x": 1}})
    assert AFTrajAdapter().convert_record(source).steps[0].raw_step == source["turns"][0]


@pytest.mark.parametrize("action", ["tool", {"name": "tool"}, None])
def test_retains_common_action_types(action):
    assert AFTrajAdapter().convert_record(record(0, action)).steps[0].action == action


def test_invalid_json_is_recorded_as_rejection(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("{not-json}\n", encoding="utf-8")
    stats = AFTrajAdapter().convert_file(source, tmp_path / "out.jsonl", tmp_path / "rejected.jsonl")
    rejected = (tmp_path / "rejected.jsonl").read_text(encoding="utf-8")
    assert stats["rejected"] == 1
    assert "Invalid JSONL" in rejected


def test_empty_action_is_not_classified_as_action():
    trajectory = AFTrajAdapter().convert_record(record(0, ""))
    assert trajectory.steps[0].event_type == "message"


def test_role_is_used_as_actor_when_actor_is_missing():
    source = record(0)
    source["turns"][0]["role"] = "MathSolver"
    source["turns"][0].pop("actor")
    trajectory = AFTrajAdapter().convert_record(source)
    assert trajectory.steps[0].actor == "MathSolver"


def test_environment_content_is_preserved_as_observation():
    source = record(0)
    source["turns"][1]["role"] = "environment"
    source["turns"][1]["content"] = "tool result"
    source["turns"][1].pop("observation")
    trajectory = AFTrajAdapter().convert_record(source)
    assert trajectory.steps[1].event_type == "observation"
    assert trajectory.steps[1].observation == "tool result"
