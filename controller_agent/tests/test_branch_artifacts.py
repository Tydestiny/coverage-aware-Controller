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
    build_continuation_rows,
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


def test_comparison_score_difference_has_no_float_noise() -> None:
    row = comparison_row(
        identity(),
        {"success": False, "score": 0.8, "combined_tokens": {"total_tokens": 100}},
        {"success": False, "score": 0.6, "combined_tokens": {"total_tokens": 90}},
    )

    assert row["score_difference"] == -0.2
    assert row["token_difference"] == -10
