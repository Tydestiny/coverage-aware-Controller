from dataclasses import replace

import pytest

from agent_error_compaction.adapters.aftraj import AFTrajAdapter
from agent_error_compaction.schemas import ErrorAnnotations, PrimaryErrorAnnotation
from agent_error_compaction.validation import AFTrajValidator, CanonicalValidator, TrajectoryValidationError, validate_trajectory


def source(mistake_step=0):
    return {"conv_id": "v", "turns": [{"content": "a"}, {"content": "b"}], "mistake_step": mistake_step}


def test_rejects_error_step_out_of_range():
    with pytest.raises(TrajectoryValidationError, match="primary_error.step_index"):
        validate_trajectory(AFTrajAdapter().convert_record(source(2)))


def test_rejects_success_with_primary_error():
    trajectory = AFTrajAdapter().convert_record(source(None))
    primary = PrimaryErrorAnnotation(0, 0, "agent", "reason", None, None, "decisive_mistake_step", "dataset_author", None, None)
    bad = replace(trajectory, annotations=ErrorAnnotations("source_provided", primary, []))
    with pytest.raises(TrajectoryValidationError, match="success cannot"):
        AFTrajValidator().validate(bad)


def test_rejects_failure_without_primary_error():
    trajectory = AFTrajAdapter().convert_record(source(0))
    bad = replace(trajectory, annotations=ErrorAnnotations("source_provided", None, []))
    with pytest.raises(TrajectoryValidationError, match="failure requires"):
        AFTrajValidator().validate(bad)


def test_generic_validator_does_not_require_failure_annotation():
    trajectory = AFTrajAdapter().convert_record(source(0))
    bad = replace(trajectory, annotations=ErrorAnnotations("not_provided", None, []))
    CanonicalValidator().validate(bad)


def test_rejects_non_continuous_indices():
    trajectory = AFTrajAdapter().convert_record(source(0))
    bad = replace(trajectory, steps=[replace(trajectory.steps[0], step_index=1), trajectory.steps[1]])
    with pytest.raises(TrajectoryValidationError, match="continuous"):
        validate_trajectory(bad)


def test_rejects_empty_steps():
    trajectory = AFTrajAdapter().convert_record(source(0))
    bad = replace(trajectory, steps=[], metadata={"num_steps": 0, "source_turn_count": 0})
    with pytest.raises(TrajectoryValidationError, match="non-empty"):
        validate_trajectory(bad)
