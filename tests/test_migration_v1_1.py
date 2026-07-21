import json

from agent_error_compaction.migration import migrate_record


def old_record(source="injected", failure=True):
    return {
        "trajectory_id": "old-1",
        "trajectory_family_id": "family-1",
        "source_dataset": "AFTraj-2K",
        "source_split": "train",
        "domain": "math",
        "task_id": None,
        "task": "Solve it",
        "gold_answer": "42",
        "outcome": "failure" if failure else "success",
        "tools": [{"name": "compute"}],
        "steps": [{"step_index": 0, "source_step_index": 0, "source_step_id": None, "actor": "MathSolver", "role": "assistant", "event_type": "action", "content": "call", "thought": "think", "action": {"name": "compute"}, "observation": {"result": "1"}, "raw_step": {"content": "call", "secret": "unchanged"}, "token_count": None}],
        "labels": {"is_failure": failure, "error_onset_step": 0 if failure else None, "source_error_onset_step": 0 if failure else None, "responsible_actor": "MathSolver", "error_type_original": "wrong", "error_type_unified": None, "error_reason": "wrong call", "label_source": source, "unsafe_source": source},
        "metadata": {"num_steps": 1, "source_turn_count": 1},
    }


def test_migrates_success_with_empty_primary_error():
    result = migrate_record(old_record(failure=False))
    assert result.schema_version == "1.1"
    assert result.annotations.primary_error is None
    assert result.annotations.error_events == []
    assert result.outcome.status == "success"


def test_migrates_injected_primary_error():
    primary = migrate_record(old_record("injected")).annotations.primary_error
    assert primary is not None
    assert primary.step_index == 0
    assert primary.source_step_index == 0
    assert primary.actor_id == "MathSolver"
    assert primary.reason == "wrong call"
    assert primary.label_semantics == "decisive_mistake_step"
    assert primary.annotation_source == "dataset_author"
    assert primary.annotation_method == "controlled_injection"


def test_migrates_diagnosed_annotation_method():
    primary = migrate_record(old_record("diagnosed")).annotations.primary_error
    assert primary is not None
    assert primary.annotation_method == "post_hoc_diagnosis"


def test_migrates_outcome_and_structured_step_raw_values():
    result = migrate_record(old_record())
    assert result.outcome.raw == "failure"
    assert result.steps[0].actor.id == "MathSolver"
    assert result.steps[0].actor.type == "agent"
    assert result.steps[0].message_role == "assistant"
    assert result.steps[0].decision_step_index == 0
    assert result.steps[0].tool_call.raw == {"name": "compute"}
    assert result.steps[0].tool_result.raw == {"result": "1"}
    assert result.steps[0].raw_step == {"content": "call", "secret": "unchanged"}
    assert result.derived_annotations.first_error_step is None
    assert result.derived_annotations.evidence_steps == []

