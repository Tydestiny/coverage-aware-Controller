"""Tests for restored-checkpoint branch artifact ledgers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from controller_agent.logging.branch_artifacts import (
    AgentCallAuditor,
    PersistentExperimentLogger,
    RunIdentity,
    TokenUsageLedger,
    build_checkpoint_prefix_rows,
    build_continuation_rows,
    build_controller_summary,
    build_full_trajectory_rows,
    comparison_row,
    read_jsonl,
)
from controller_agent.state.schemas import (
    Action,
    ControllerDecision,
    ControllerInput,
    ControllerResponse,
    ControllerRuntimeState,
    TriggerContext,
    TriggerResult,
)


@dataclass
class Tokens:
    input_cache_hit_or_miss: int = 31
    output: int = 7


@dataclass
class Cost:
    total: float = 0.004


@dataclass
class Usage:
    tokens: Tokens = field(default_factory=Tokens)
    cost: Cost = field(default_factory=Cost)


class FakeAgent:
    """Minimal Agent exposing one model call and its new assistant output."""

    def __init__(self) -> None:
        self.step_number = 9
        self.messages: list[dict[str, str]] = []

    def next_execution_inputs_usage_and_status(self, _outputs):
        self.messages.append({"role": "assistant", "content": "apis.supervisor.complete_task()"})
        return [object()], Usage(), object()


def identity() -> RunIdentity:
    """Return one stable test identity."""
    return RunIdentity(
        experiment_id="exp",
        run_id="task__checkpoint__controller",
        task_id="task",
        checkpoint_id="checkpoint",
        branch="controller",
    )


def test_agent_call_is_written_per_call_before_finalization(tmp_path: Path) -> None:
    token_ledger = TokenUsageLedger(tmp_path / "token_usage.jsonl", identity())
    agent = FakeAgent()
    auditor = AgentCallAuditor(agent=agent, identity=identity(), ledger=token_ledger, model="agent-model")
    auditor.install()

    agent.next_execution_inputs_usage_and_status([])

    rows = read_jsonl(tmp_path / "token_usage.jsonl")
    assert len(rows) == 1
    assert rows[0]["source"] == "agent"
    assert rows[0]["input_tokens"] == 31
    assert rows[0]["output_tokens"] == 7
    assert rows[0]["total_tokens"] == 38
    assert auditor.calls[0]["agent_output"] == "apis.supervisor.complete_task()"


def test_controller_call_persists_exact_runtime_input_and_provider_response(tmp_path: Path) -> None:
    token_ledger = TokenUsageLedger(tmp_path / "token_usage.jsonl", identity())
    logger = PersistentExperimentLogger(
        path=tmp_path / "controller_calls.jsonl",
        identity=identity(),
        token_ledger=token_ledger,
        prompt_hash="prompt-hash",
    )
    controller_input = ControllerInput(
        task="Verify the target set.",
        trigger_reason="before_finish",
        recent_history=[{"role": "assistant", "content": "done"}],
        current_agent_output="apis.supervisor.complete_task()",
        known_entities=["artist"],
        completed_operations=["search_artists"],
        recent_errors=[],
    )
    response = ControllerResponse(
        decision=ControllerDecision(Action.VERIFY_TARGET_SET, "missing evidence", "Check every page."),
        model_name="controller-model",
        input_tokens=41,
        output_tokens=9,
        raw_response=json.dumps({"id": "provider-call", "choices": []}),
    )
    context = TriggerContext(
        task_id="task",
        step_index=9,
        task=controller_input.task,
        proposed_execution="apis.supervisor.complete_task()",
        last_execution_outputs=[],
        recent_history=[],
        runtime_state={},
    )
    trigger = TriggerResult(True, "before_finish", {"api_name": "complete_task"})

    logger.log_controller_event(
        context=context,
        trigger=trigger,
        controller_input=controller_input,
        response=response,
        runtime_state=ControllerRuntimeState(total_calls=1, total_blocks=1),
        blocked=True,
    )

    calls = read_jsonl(tmp_path / "controller_calls.jsonl")
    assert len(calls) == 1
    assert calls[0]["controller_input"]["current_agent_output"] == "apis.supervisor.complete_task()"
    assert calls[0]["controller_input_provenance"]["reconstructed"] is False
    assert calls[0]["provider_raw_response"] == {
        "available": True,
        "value": {"id": "provider-call", "choices": []},
    }
    assert calls[0]["controller_response"]["decision"]["action"] == "VERIFY_TARGET_SET"
    usage = read_jsonl(tmp_path / "token_usage.jsonl")
    assert usage[0]["source"] == "controller"
    assert usage[0]["total_tokens"] == 50


def test_controller_runtime_fail_open_is_persisted_and_summarized(tmp_path: Path) -> None:
    token_ledger = TokenUsageLedger(tmp_path / "token_usage.jsonl", identity())
    logger = PersistentExperimentLogger(
        path=tmp_path / "controller_calls.jsonl",
        runtime_events_path=tmp_path / "controller_runtime_events.jsonl",
        identity=identity(),
        token_ledger=token_ledger,
        prompt_hash="prompt-hash",
    )
    trigger = TriggerResult(True, "before_finish", {"api_name": "complete_task"})

    logger.log_runtime_event(
        task_id="task",
        step_index=10,
        event_type="controller_fail_open",
        reason="controller_budget_exhausted",
        trigger=trigger,
        error=None,
        runtime_state=ControllerRuntimeState(total_calls=1, total_blocks=1),
    )

    runtime_events = read_jsonl(tmp_path / "controller_runtime_events.jsonl")
    summary = build_controller_summary(
        identity=identity(),
        enabled=True,
        model="controller-model",
        prompt_version="g1",
        prompt_hash="prompt-hash",
        controller_calls=[],
        runtime_events=runtime_events,
        token_rows=[],
    )
    assert runtime_events[0]["reason"] == "controller_budget_exhausted"
    assert summary["fail_open_count"] == 1
    assert summary["fail_open_by_reason"] == {"controller_budget_exhausted": 1}
    assert summary["tokens"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_continuation_keeps_blocked_agent_call_and_shared_event_id(tmp_path: Path) -> None:
    del tmp_path
    calls = [
        {
            "event_id": identity().event_id(9),
            "turn_index": 9,
            "agent_output": "finish",
        }
    ]
    events = [
        {
            "generated_code": "apis.supervisor.complete_task()",
            "environment_execution_index": 8,
            "trigger_detected": True,
            "controller_action": "VERIFY_TARGET_SET",
            "execution_allowed": False,
            "guidance_injected": True,
            "world_batch_execute_called": False,
            "task_completed": False,
            "termination_reason": "blocked_extra_turns_exhausted",
        }
    ]

    rows = build_continuation_rows(identity(), events, calls)

    assert rows[0]["event_id"] == identity().event_id(9)
    assert rows[0]["sequence_index"] == 0
    assert rows[0]["execution_allowed"] is False
    assert rows[0]["guidance_injected"] is True


def test_full_trajectory_preserves_prefix_and_marks_branch_boundary(tmp_path: Path) -> None:
    source_task_dir = tmp_path / "source_task"
    post_dir = source_task_dir / "checkpoints" / "post_observation"
    collection_dir = source_task_dir / "collection"
    post_dir.mkdir(parents=True)
    collection_dir.mkdir(parents=True)
    checkpoints = []
    for turn in (1, 3):
        checkpoint = {
            "checkpoint_id": f"checkpoint_{turn}",
            "task_id": "task",
            "phase": "post_observation",
            "agent_turn_index": turn,
            "environment_execution_index": turn - 1,
            "agent_messages": [{"role": "assistant", "content": f"output-{turn}"}],
            "executed_code": f"code-{turn}",
            "last_execution_outputs": [f"observation-{turn}"],
        }
        checkpoints.append(checkpoint)
        (post_dir / f"checkpoint_{turn}.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    source_events = [
        {
            "agent_turn_index": 1,
            "environment_execution_index": 1,
            "generated_code": "code-1",
            "execution_allowed": True,
            "world_batch_execute_called": True,
        },
        {
            "agent_turn_index": 2,
            "environment_execution_index": 1,
            "generated_code": "blocked-code",
            "trigger_detected": True,
            "controller_action": "REPLAN",
            "execution_allowed": False,
            "guidance_injected": True,
            "world_batch_execute_called": False,
        },
        {
            "agent_turn_index": 3,
            "environment_execution_index": 2,
            "generated_code": "code-3",
            "execution_allowed": True,
            "world_batch_execute_called": True,
        },
    ]
    (collection_dir / "turn_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_events),
        encoding="utf-8",
    )
    prefix = build_checkpoint_prefix_rows(identity(), source_task_dir, checkpoints[-1])
    continuation = [
        identity().to_dict()
        | {
            "event_id": identity().event_id(4),
            "sequence_index": 0,
            "agent_turn_index": 4,
        }
    ]

    trajectory = build_full_trajectory_rows(prefix, continuation)

    assert [row["agent_turn_index"] for row in prefix] == [1, 2, 3]
    assert prefix[1]["source_checkpoint_id"] is None
    assert prefix[1]["execution_allowed"] is False
    assert [row["sequence_index"] for row in trajectory] == [0, 1, 2, 3]
    assert trajectory[-1]["trajectory_segment"] == "branch_continuation"
    assert trajectory[-1]["segment_sequence_index"] == 0


def test_comparison_score_difference_has_no_float_noise() -> None:
    row = comparison_row(
        identity(),
        {"success": False, "score": 0.8, "combined_tokens": {"total_tokens": 100}},
        {"success": False, "score": 0.6, "combined_tokens": {"total_tokens": 90}},
    )

    assert row["score_difference"] == -0.2
    assert row["token_difference"] == -10
