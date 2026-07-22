"""Unit tests for AgentRx adapter."""

from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.agent_error_compaction.adapters.agentrx import AgentRxAdapter


def test_agentrx_adapter_failure_record():
    adapter = AgentRxAdapter()
    record = {
        "trajectory_id": "tau_retail_test_1",
        "instruction": "Exchange product",
        "steps": [
            {
                "index": 1,
                "substeps": [
                    {"sub_index": 1, "role": "system", "content": "Policy text"}
                ]
            },
            {
                "index": 2,
                "substeps": [
                    {"sub_index": 1, "role": "user", "content": "I want to exchange shoes"}
                ]
            },
            {
                "index": 3,
                "substeps": [
                    {"sub_index": 1, "role": "assistant", "content": "Sure", "action": {"tool": "get_user"}}
                ]
            }
        ],
        "failures": [
            {
                "failure_id": "1",
                "step_number": 3,
                "step_reason": "Assistant failed to authenticate user before providing info.",
                "failure_category": "Instruction Adherence Failure",
                "failed_agent": "Assistant"
            }
        ],
        "root_cause": {
            "failure_id": "1",
            "reason_for_root_cause": "Did not authenticate user first."
        }
    }

    traj = adapter.convert_record(record)
    
    assert traj.trajectory_id == "tau_retail_test_1"
    assert traj.trajectory_family_id == "tau_retail_test_1"
    assert traj.outcome == "failure"
    assert len(traj.steps) == 3
    # 检查平铺后的索引与 0-based onset 转换是否正确 (source step 3 对应 0-based index 2)
    assert traj.labels.is_failure is True
    assert traj.labels.error_onset_step == 2
    assert traj.labels.source_error_onset_step == 3
    assert traj.labels.responsible_actor == "Assistant"
    assert traj.labels.error_type_original == "Instruction Adherence Failure"


def test_agentrx_adapter_success_record():
    adapter = AgentRxAdapter()
    record = {
        "trajectory_id": "tau_retail_success_1",
        "instruction": "Simple query",
        "steps": [
            {
                "index": 1,
                "substeps": [
                    {"sub_index": 1, "role": "user", "content": "Hi"}
                ]
            }
        ],
        "failures": [],
        "root_cause": {}
    }

    traj = adapter.convert_record(record)
    
    assert traj.trajectory_id == "tau_retail_success_1"
    assert traj.outcome == "success"
    assert traj.labels.is_failure is False
    assert traj.labels.error_onset_step is None
    assert traj.labels.source_error_onset_step is None