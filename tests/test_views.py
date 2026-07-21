from copy import deepcopy

import pytest

from agent_error_compaction.views import build_evaluation_view, build_model_input_view, validate_model_input_view


FORBIDDEN = {"schema_version", "trajectory_id", "trajectory_family_id", "source", "source_split", "experiment_split", "environment", "gold_answer", "outcome", "annotations", "derived_annotations", "metadata", "labels", "is_failure", "error_onset_step", "source_error_onset_step", "critical_error_step", "responsible_actor", "error_reason", "error_type_original", "error_type_unified", "label_source", "unsafe_source", "raw_step", "source_step_index", "source_step_id", "source_unit_type", "raw_event"}


def trajectory(is_failure=True):
    return {
        "schema_version": "1.1",
        "trajectory_id": "trajectory-1",
        "trajectory_family_id": "family-1",
        "source": {"dataset": "AFTraj-2K", "original_split": None, "record_id": "trajectory-1"}, "experiment_split": "train",
        "task": {"task_id": None, "instruction": "Complete the task", "gold_answer": "secret answer"},
        "environment": {"domain": "demo", "tools": [{"name": "search"}]},
        "outcome": {"status": "failure" if is_failure else "success", "score": None, "termination_reason": None, "raw": "failure" if is_failure else "success"},
        "annotations": {"annotation_status": "source_provided", "primary_error": {"step_index": 2, "source_step_index": 2, "actor_id": "Agent", "reason": "secret reason", "error_type_original": "source", "error_type_unified": None, "label_semantics": "decisive_mistake_step", "annotation_source": "dataset_author", "annotation_method": "controlled_injection", "confidence": None, "alignment_status": "exact"} if is_failure else None, "error_events": []},
        "derived_annotations": {"first_error_step": {"value": None, "status": "not_annotated", "provenance": None}, "earliest_detectable_step": {"value": None, "status": "not_annotated", "provenance": None}, "evidence_steps": {"value": [], "status": "not_annotated", "provenance": None}},
        "steps": [{"step_index": i, "source_step_index": i, "source_step_id": None, "source_unit_type": "turn", "decision_step_index": i, "event_type": "message", "content": f"step {i}", "raw_step": {"error_reason": "secret"}, "token_count": None, "actor": {"id": "Agent", "type": "agent"}, "message_role": "assistant", "reasoning": "public", "events": [{"event_index": 0, "event_type": "message", "actor": {"id": "Agent", "type": "agent"}, "content": f"step {i}", "tool_call": None, "tool_result": None, "raw_event": {"error_reason": "secret"}}]} for i in range(4)],
        "metadata": {"secret": "do not pass"},
    }


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def test_builds_only_first_three_steps_and_whitelisted_fields():
    model_input = build_model_input_view(trajectory(), 2)
    assert set(model_input) == {"task", "tools", "steps"}
    assert len(model_input["steps"]) == 3
    assert model_input["steps"][-1]["step_index"] == 2
    assert set(model_input["steps"][0]) == {"step_index", "decision_step_index", "actor", "message_role", "event_type", "content", "reasoning", "events"}


def test_model_input_has_no_forbidden_nested_fields_or_raw_step():
    model_input = build_model_input_view(trajectory(), 2)
    assert not FORBIDDEN.intersection(walk_keys(model_input))
    validate_model_input_view(model_input)


@pytest.mark.parametrize("prefix_end_step", [-1, 4])
def test_rejects_invalid_prefix_position(prefix_end_step):
    with pytest.raises(ValueError, match="prefix_end_step"):
        build_model_input_view(trajectory(), prefix_end_step)


def test_evaluation_view_marks_error_before_and_after_onset():
    before = build_evaluation_view(trajectory(), 1)
    at_onset = build_evaluation_view(trajectory(), 2)
    after = build_evaluation_view(trajectory(), 3)
    assert before["critical_error_occurred"] is False
    assert at_onset["critical_error_occurred"] is True
    assert after["critical_error_occurred"] is True
    assert at_onset["critical_error_step"] == 2
    assert at_onset["responsible_actor"] == "Agent"


def test_safe_evaluation_view_never_reports_error():
    for end_step in range(4):
        result = build_evaluation_view(trajectory(False), end_step)
        assert result["critical_error_occurred"] is False
        assert result["critical_error_step"] is None


def test_original_trajectory_is_not_modified():
    source = trajectory()
    original = deepcopy(source)
    build_model_input_view(source, 2)
    build_evaluation_view(source, 2)
    assert source == original


def test_validation_reports_nested_forbidden_path():
    model_input = {"task": "task", "steps": [{"step_index": 0, "decision_step_index": 0, "nested": {"raw_step": True}}]}
    with pytest.raises(ValueError, match=r"steps\[0\]\.nested\.raw_step"):
        validate_model_input_view(model_input)


def test_validation_requires_task_steps_and_continuous_indices():
    with pytest.raises(ValueError, match="task"):
        validate_model_input_view({"steps": []})
    with pytest.raises(ValueError, match="continuous"):
        validate_model_input_view({"task": "task", "steps": [{"step_index": 1, "decision_step_index": 1}]})
