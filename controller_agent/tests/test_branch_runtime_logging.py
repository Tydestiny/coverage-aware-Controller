"""Tests for direct checkpoint-branch runtime logging."""

from __future__ import annotations

import json
from pathlib import Path

from controller_agent.state.schemas import Action, ControllerDecision, ControllerInput, ControllerResponse


def _input() -> ControllerInput:
    return ControllerInput(
        task="Test task",
        trigger_reason="before_finish",
        trigger_metadata={},
        agent_context_messages=[{"role": "assistant", "content": "check state"}],
        proposed_agent_output="apis.supervisor.complete_task()",
        proposed_execution="apis.supervisor.complete_task()",
        previous_controller_decisions=[],
        context_complete=True,
    )


def _response(action: Action = Action.REPLAN) -> ControllerResponse:
    return ControllerResponse(
        decision=ControllerDecision(action, "needs verification", "Verify the target set."),
        model_name="fake-controller",
        input_tokens=13,
        output_tokens=7,
        raw_response='{"action":"REPLAN"}',
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_runtime_logger_writes_isolated_baseline_and_controller_records(tmp_path: Path) -> None:
    """Baseline remains controller-empty while Controller data is written directly."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    baseline = BranchRuntimeLogger(tmp_path / "baseline", branch="baseline", checkpoint_last_step=8)
    controller = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=8)

    baseline.record_continuation(
        step_number=9,
        agent_output="print('read')",
        generated_code="print('read')",
        execution_allowed=True,
        environment_observation="{}",
        task_completed=False,
        termination_reason=None,
    )
    call_id = controller.record_controller_call(
        step_number=9,
        controller_input=_input(),
        controller_response=_response(),
        guidance="[CONTROLLER_GUIDANCE]\nVerify the target set.",
        blocked=True,
        fail_open=False,
        prompt_hash="prompt-g0",
    )
    controller.record_continuation(
        step_number=9,
        agent_output="apis.supervisor.complete_task()",
        generated_code="apis.supervisor.complete_task()",
        controller_triggered=True,
        controller_call_id=call_id,
        controller_action="REPLAN",
        guidance="[CONTROLLER_GUIDANCE]\nVerify the target set.",
        execution_allowed=False,
        environment_observation=None,
        task_completed=False,
        termination_reason=None,
    )
    controller.record_continuation(
        step_number=10,
        agent_output="print('verify')",
        generated_code="print('verify')",
        execution_allowed=True,
        environment_observation="verified",
        task_completed=True,
        termination_reason="task_completed",
    )

    assert _rows(baseline.controller_calls_path) == []
    assert baseline.raw_logs_dir != controller.raw_logs_dir
    assert baseline.raw_logs_dir.is_dir() and controller.raw_logs_dir.is_dir()
    calls = _rows(controller.controller_calls_path)
    assert calls[0]["record_origin"] == "runtime_direct"
    assert calls[0]["controller_input"]["proposed_agent_output"] == "apis.supervisor.complete_task()"
    assert calls[0]["controller_response"]["decision"]["action"] == "REPLAN"
    assert calls[0]["usage"] == {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20}

    continuation = _rows(controller.continuation_path)
    assert [row["step_number"] for row in continuation] == [9, 10]
    assert continuation[0]["controller_call_id"] == call_id
    assert continuation[0]["execution_allowed"] is False
    assert continuation[0]["environment_observation"] is None
    assert continuation[1]["environment_observation"] == "verified"
    assert continuation[1]["termination_reason"] == "task_completed"


def test_runtime_logger_uses_unique_call_ids_and_appends_without_overwriting(tmp_path: Path) -> None:
    """Repeated writes preserve distinct agent attempts at the same raw step."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    logger = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=4)
    first = logger.record_continuation(
        step_number=5,
        agent_output="first attempt",
        generated_code="apis.supervisor.complete_task()",
        execution_allowed=False,
        environment_observation=None,
        task_completed=False,
        termination_reason=None,
    )
    second = logger.record_continuation(
        step_number=5,
        agent_output="retry attempt",
        generated_code="print('verify')",
        execution_allowed=True,
        environment_observation="ok",
        task_completed=False,
        termination_reason=None,
    )

    rows = _rows(logger.continuation_path)
    assert first != second
    assert [row["agent_call_id"] for row in rows] == [first, second]
    assert len(rows) == 2


def test_runtime_logger_rejects_prefix_steps(tmp_path: Path) -> None:
    """Continuation cannot accidentally contain checkpoint-prefix events."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    logger = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=8)
    try:
        logger.record_continuation(
            step_number=8,
            agent_output="prefix",
            generated_code="print('prefix')",
            execution_allowed=True,
            environment_observation="ignored",
            task_completed=False,
            termination_reason=None,
        )
    except ValueError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("Expected checkpoint boundary validation failure")


def test_runtime_logger_supports_initial_state_runs(tmp_path: Path) -> None:
    """An initial-state run treats step zero and later as continuation data."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    logger = BranchRuntimeLogger(tmp_path / "initial", branch="baseline", checkpoint_last_step=-1)
    logger.record_continuation(
        step_number=0,
        agent_output="print('initial')",
        generated_code="print('initial')",
        execution_allowed=True,
        environment_observation="ok",
        task_completed=False,
        termination_reason=None,
    )

    assert _rows(logger.continuation_path)[0]["step_number"] == 0


def test_checkpoint_manifest_records_shared_prefix_boundary(tmp_path: Path) -> None:
    """Checkpoint metadata explicitly separates shared prefix and continuation."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    logger = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=8)
    path = tmp_path / "checkpoint_manifest.json"
    logger.write_checkpoint_manifest(
        path,
        checkpoint_id="task_post_0008",
        state_path="state/task_post_0008",
        source_trajectory="source/logger.jsonl",
        source_logs="source/logs",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["last_step_number"] == 8
    assert manifest["continuation_start_step_number"] == 9
    assert manifest["checkpoint_id"] == "task_post_0008"


def test_raw_logs_are_copied_to_the_owning_branch_only(tmp_path: Path) -> None:
    """Archiving never shares or rewrites another branch's raw logs."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    source = tmp_path / "temporary" / "logs"
    source.mkdir(parents=True)
    (source / "logger.jsonl").write_text('{"step_number": 9}\n', encoding="utf-8")
    (source / "logger.log").write_text("branch-only log\n", encoding="utf-8")
    baseline = BranchRuntimeLogger(tmp_path / "baseline", branch="baseline", checkpoint_last_step=8)
    controller = BranchRuntimeLogger(tmp_path / "controller", branch="controller", checkpoint_last_step=8)

    assert baseline.archive_raw_logs(source) == ["logger.jsonl", "logger.log"]
    assert (baseline.raw_logs_dir / "logger.jsonl").is_file()
    assert not (controller.raw_logs_dir / "logger.jsonl").exists()


def test_raw_log_archive_joins_checkpoint_prefix_and_preserves_continuation(tmp_path: Path) -> None:
    """The readable logger contains task history while continuation stays separate."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    prefix = tmp_path / "baseline" / "logs" / "logger.jsonl"
    prefix.parent.mkdir(parents=True)
    prefix.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"role": "agent", "step_number": 1, "content": "prefix action"},
                {"role": "environment", "content": "prefix observation"},
                {"role": "agent", "step_number": 2, "content": "must not be copied"},
            )
        ) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "branch" / "logs"
    source.mkdir(parents=True)
    source_logger = '{"role":"agent","step_number":2,"content":"branch action"}\n'
    (source / "logger.jsonl").write_text(source_logger, encoding="utf-8")

    logger = BranchRuntimeLogger(tmp_path / "result", branch="controller", checkpoint_last_step=1)
    logger.archive_raw_logs(source, prefix_logger_jsonl=prefix)

    combined = _rows(logger.raw_logs_dir / "logger.jsonl")
    assert [row["content"] for row in combined] == ["prefix action", "prefix observation", "branch action"]
    assert (logger.raw_logs_dir / "logger_continuation.jsonl").read_text(encoding="utf-8") == source_logger
    metadata = json.loads((logger.raw_logs_dir / "logger_history_manifest.json").read_text(encoding="utf-8"))
    assert metadata["prefix_record_count"] == 2
    assert metadata["continuation_record_count"] == 1


def test_raw_log_archive_preserves_cached_checkpoint_proposal(tmp_path: Path) -> None:
    """A replayed pre-action remains visible before the restored continuation."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    prefix = tmp_path / "baseline" / "logger.jsonl"
    prefix.parent.mkdir(parents=True)
    prefix.write_text('{"role":"environment","step_number":6,"content":"prefix"}\n', encoding="utf-8")
    source = tmp_path / "branch" / "logs"
    source.mkdir(parents=True)
    (source / "logger.jsonl").write_text(
        '{"role":"agent","step_number":8,"content":"regenerated action"}\n',
        encoding="utf-8",
    )
    (source / "logger.log").write_text("restored continuation\n", encoding="utf-8")

    logger = BranchRuntimeLogger(tmp_path / "result", branch="controller", checkpoint_last_step=6)
    logger.record_cached_checkpoint_proposal(
        step_number=7,
        agent_output="original proposed action",
        generated_code="apis.supervisor.complete_task()",
        controller_triggered=True,
        controller_action="REPLAN",
        guidance="verify first",
        execution_allowed=False,
    )
    logger.archive_raw_logs(source, prefix_logger_jsonl=prefix)

    combined = _rows(logger.raw_logs_dir / "logger.jsonl")
    assert [row["content"] for row in combined] == ["prefix", "original proposed action", "regenerated action"]
    proposal = combined[1]
    assert proposal["event_type"] == "checkpoint_proposed_action"
    assert proposal["execution_allowed"] is False
    continuation = _rows(logger.raw_logs_dir / "logger_continuation.jsonl")
    assert [row["content"] for row in continuation] == ["original proposed action", "regenerated action"]
    assert "original proposed action" in (logger.raw_logs_dir / "logger.log").read_text(encoding="utf-8")


def test_raw_log_archive_keeps_full_readable_logger_and_continuation(tmp_path: Path) -> None:
    """Readable logs retain task context without retaining future baseline events."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    prefix_logs = tmp_path / "baseline" / "logs"
    prefix_logs.mkdir(parents=True)
    (prefix_logs / "logger.jsonl").write_text(
        '\n'.join((
            '{"role":"agent","step_number":1,"content":"prefix action"}',
            '{"role":"environment"}',
            '{"role":"agent","step_number":2,"content":"future action"}',
        )) + '\n',
        encoding="utf-8",
    )
    (prefix_logs / "logger.log").write_text(
        "task start\nagent (step #1)\nenvironment (step #2)\nagent (step #2)\nfuture action\n",
        encoding="utf-8",
    )
    source = tmp_path / "branch" / "logs"
    source.mkdir(parents=True)
    (source / "logger.jsonl").write_text('{"role":"agent","step_number":2}\n', encoding="utf-8")
    (source / "logger.log").write_text("agent (step #2) continuation\n", encoding="utf-8")

    logger = BranchRuntimeLogger(tmp_path / "result", branch="controller", checkpoint_last_step=1)
    logger.archive_raw_logs(
        source,
        prefix_logger_jsonl=prefix_logs / "logger.jsonl",
        prefix_logger_log=prefix_logs / "logger.log",
    )

    full = (logger.raw_logs_dir / "logger.log").read_text(encoding="utf-8")
    assert "task start" in full
    assert "future" not in full
    assert "CHECKPOINT CONTINUATION" in full
    assert (logger.raw_logs_dir / "logger_continuation.log").read_text(encoding="utf-8") == "agent (step #2) continuation\n"


def test_raw_log_archive_preserves_credentials_without_dropping_records(tmp_path: Path) -> None:
    """Durable raw logs preserve exact provider records for replay."""
    from controller_agent.branching.branch_runtime_logger import BranchRuntimeLogger

    source = tmp_path / "temporary" / "logs"
    source.mkdir(parents=True)
    (source / "lm_calls.jsonl").write_text(
        '{"id":"call-1","input":{"api_key":"live-secret","messages":[{"content":"access_token=abc and \\"password\\": \\"hidden\\""}]}}\n',
        encoding="utf-8",
    )
    logger = BranchRuntimeLogger(tmp_path / "branch", branch="controller", checkpoint_last_step=8)

    assert logger.archive_raw_logs(source) == ["lm_calls.jsonl"]
    rows = _rows(logger.raw_logs_dir / "lm_calls.jsonl")
    assert rows[0]["id"] == "call-1"
    assert rows[0]["input"]["api_key"] == "live-secret"
    assert "abc" in json.dumps(rows[0])
    assert "hidden" in json.dumps(rows[0])
