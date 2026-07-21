from dataclasses import replace

import pytest

from agent_error_compaction.adapters.aftraj import AFTrajAdapter
from agent_error_compaction.validation import TrajectoryValidationError, validate_trajectory


def source(mistake_step=0):
    return {"conv_id": "v", "turns": [{"content": "a"}, {"content": "b"}], "mistake_step": mistake_step}


def test_rejects_error_step_out_of_range():
    with pytest.raises(TrajectoryValidationError, match="error_onset_step"):
        validate_trajectory(AFTrajAdapter().convert_record(source(2)))


def test_rejects_success_with_error_onset():
    trajectory = AFTrajAdapter().convert_record(source(None))
    bad = replace(trajectory, labels=replace(trajectory.labels, error_onset_step=0))
    with pytest.raises(TrajectoryValidationError, match="success cannot"):
        validate_trajectory(bad)


def test_rejects_failure_without_error_onset():
    trajectory = AFTrajAdapter().convert_record(source(0))
    bad = replace(trajectory, labels=replace(trajectory.labels, error_onset_step=None))
    with pytest.raises(TrajectoryValidationError, match="failure requires"):
        validate_trajectory(bad)


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

