"""Focused tests for complete-trajectory and Controller result validation."""

from __future__ import annotations

from controller_agent.logging.branch_artifacts import RunIdentity, build_controller_summary
from scripts.validate_results_structure import (
    ValidationErrors,
    validate_controller_summary,
    validate_trajectory,
)


def identity() -> RunIdentity:
    return RunIdentity("exp", "task__checkpoint__controller", "task", "checkpoint", "controller")


def test_complete_trajectory_and_controller_summary_reconcile() -> None:
    run_identity = identity()
    common = run_identity.to_dict()
    continuation = [
        common
        | {
            "event_id": run_identity.event_id(3),
            "sequence_index": 0,
            "agent_turn_index": 3,
        }
    ]
    trajectory = [
        common
        | {
            "event_id": run_identity.event_id(turn),
            "sequence_index": index,
            "segment_sequence_index": index,
            "trajectory_segment": "checkpoint_prefix",
            "agent_turn_index": turn,
        }
        for index, turn in enumerate((1, 2))
    ] + [
        continuation[0]
        | {
            "sequence_index": 2,
            "segment_sequence_index": 0,
            "trajectory_segment": "branch_continuation",
        }
    ]
    manifest = {
        "trajectory": {
            "path": "trajectory.jsonl",
            "prefix_event_count": 2,
            "continuation_event_count": 1,
        }
    }
    result = {
        "branch": "controller",
        "checkpoint_boundary_agent_turn_index": 2,
        "trajectory_prefix_events": 2,
        "continuation_events": 1,
        "token_accounting_scope": "branch_continuation_only",
        "controller_tokens": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        "fail_open_count": 1,
    }
    controller_calls = [
        common
        | {
            "controller_call_id": run_identity.controller_call_id(1),
            "action": "REPLAN",
            "blocked": True,
            "fail_open": False,
            "guidance": "Inspect the missing page.",
            "controller_response": {"latency_seconds": 0.5},
        }
    ]
    runtime_events = [
        common
        | {
            "event_type": "controller_fail_open",
            "reason": "controller_budget_exhausted",
        }
    ]
    token_rows = [
        common
        | {
            "source": "controller",
            "call_id": run_identity.controller_call_id(1),
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
            "cost": 0.01,
        }
    ]
    summary = build_controller_summary(
        identity=run_identity,
        enabled=True,
        model="controller-model",
        prompt_version="g1",
        prompt_hash="hash",
        controller_calls=controller_calls,
        runtime_events=runtime_events,
        token_rows=token_rows,
    )
    errors = ValidationErrors()

    validate_trajectory("run", manifest, result, trajectory, continuation, common, errors)
    validate_controller_summary(
        "run",
        result,
        summary,
        controller_calls,
        runtime_events,
        token_rows,
        errors,
    )

    assert errors.messages == []


def test_trajectory_validator_detects_checkpoint_boundary_mismatch() -> None:
    run_identity = identity()
    common = run_identity.to_dict()
    trajectory = [
        common
        | {
            "event_id": run_identity.event_id(1),
            "sequence_index": 0,
            "segment_sequence_index": 0,
            "trajectory_segment": "checkpoint_prefix",
            "agent_turn_index": 1,
        }
    ]
    errors = ValidationErrors()

    validate_trajectory(
        "run",
        {
            "trajectory": {
                "path": "trajectory.jsonl",
                "prefix_event_count": 1,
                "continuation_event_count": 0,
            }
        },
        {
            "checkpoint_boundary_agent_turn_index": 2,
            "trajectory_prefix_events": 1,
            "continuation_events": 0,
            "token_accounting_scope": "branch_continuation_only",
        },
        trajectory,
        [],
        common,
        errors,
    )

    assert any("checkpoint boundary mismatch" in message for message in errors.messages)
