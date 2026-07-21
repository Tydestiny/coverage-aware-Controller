import pytest

from agent_error_compaction.adapters.aftraj import AFTrajAdapter


def record(mistake_step=None, action="call"):
    return {"conv_id": "case", "domain": "demo", "task": "task", "gold_answer": "answer", "turns": [{"actor": "assistant", "content": "first", "action": action}, {"actor": "tool", "observation": "result"}], "mistake_step": mistake_step, "mistake_agent": "assistant", "mistake_reason": "provided"}


def test_converts_safe_trajectory():
    trajectory = AFTrajAdapter().convert_record(record())
    assert trajectory.schema_version == "1.1"
    assert trajectory.outcome.status == "success"
    assert trajectory.annotations.primary_error is None


def test_converts_failed_trajectory():
    trajectory = AFTrajAdapter().convert_record(record(1))
    primary = trajectory.annotations.primary_error
    assert trajectory.outcome.status == "failure"
    assert primary is not None
    assert primary.step_index == 1
    assert primary.source_step_index == 1
    assert primary.label_semantics == "decisive_mistake_step"
    assert primary.annotation_source == "dataset_author"


def test_actor_and_message_role_are_separate():
    source = record(0)
    source["turns"][0]["role"] = "MathSolver"
    source["turns"][0].pop("actor")
    step = AFTrajAdapter().convert_record(source).steps[0]
    assert step.actor.id == "MathSolver"
    assert step.actor.type == "agent"
    assert step.message_role == "MathSolver"


@pytest.mark.parametrize("action", ["tool", {"name": "tool"}, None])
def test_raw_action_is_preserved(action):
    step = AFTrajAdapter().convert_record(record(0, action)).steps[0]
    calls = [event.tool_call for event in step.events if event.tool_call is not None]
    assert (calls[0].raw if calls else None) == action


def test_raw_observation_is_preserved():
    step = AFTrajAdapter().convert_record(record(0)).steps[1]
    results = [event.tool_result for event in step.events if event.tool_result is not None]
    assert results[0].raw == "result"
