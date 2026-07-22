"""Unit tests for Who&When Pro adapter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.agent_error_compaction.adapters.whowhenpro import WhoWhenProAdapter


def test_whowhenpro_adapter_conversion():
    adapter = WhoWhenProAdapter()
    record = {
        "id": "alfworld_test_001",
        "framework": "alfagent",
        "benchmark": "alfworld",
        "modality": "text",
        "task": {
            "query": "Put a clean mug in the cupboard",
            "answer": None
        },
        "trajectory": [
            {"kind": "user", "content": "Task description"},
            {"kind": "alfagent", "content": "go to countertop 1"},
            {"kind": "observation", "content": "You see a mug 1"}
        ],
        "ground_truth": {
            "agent": "alfagent",
            "step": 1,
            "mode": "execution_error"
        }
    }

    traj = adapter.convert_record(record)

    assert traj.trajectory_id == "alfworld_test_001"
    assert traj.source_dataset == "WhoAndWhen-Pro"
    # 无显式 task_id/family_id 时，退化为 trajectory_id，确保细粒度无泄露划分
    assert traj.trajectory_family_id == "alfworld_test_001"
    assert traj.domain == "alfworld"
    assert len(traj.steps) == 3
    assert traj.outcome == "failure"
    assert traj.labels.is_failure is True
    assert traj.labels.error_onset_step == 1
    assert traj.labels.responsible_actor == "alfagent"
    assert traj.labels.error_type_original == "execution_error"