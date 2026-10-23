"""Regression tests for trigger inputs at the actual Agent boundary."""

from types import SimpleNamespace
import pytest

from controller_agent.state.schemas import TriggerContext
from controller_agent.triggers.before_write import BeforeWriteTrigger
from controller_agent.triggers.before_finish import BeforeFinishTrigger
from controller_agent.triggers.composite import CompositeTriggerDetectorWithBudget
from controller_agent.triggers.config import BeforeWriteConfig, TriggerBudgetConfig, TriggerConfig
from controller_agent.logging.logger import summarize_trigger_events


def context(code: str, history: list[dict] | None = None, outputs: list | None = None) -> TriggerContext:
    return TriggerContext("task-1", 4, "task", code, outputs or [], history or [], {})


def test_before_write_finds_later_api_call_without_matching_comments() -> None:
    trigger = BeforeWriteTrigger()
    assert trigger.detect(context('print("ready")\napis.spotify.delete_playlist(playlist_id=12)')).should_call_controller
    assert not trigger.detect(context('# apis.spotify.delete_playlist(playlist_id=12)\nprint("ready")')).should_call_controller
    assert not trigger.detect(context('print("apis.spotify.delete_playlist(playlist_id=12)")')).should_call_controller


@pytest.mark.parametrize("api_name", [
    "spotify.like_song", "spotify.unlike_song", "spotify.like_playlist",
    "venmo.like_transaction", "gmail.mark_thread_starred",
    "spotify.clear_song_queue", "file_system.copy_file", "file_system.move_file",
    "todoist.assign_or_unassign_task", "venmo.approve_payment_request",
    "venmo.deny_payment_request", "splitwise.record_payment",
    "gmail.reply_to_email", "amazon.place_order", "spotify.play_music",
    "spotify.download_song", "todoist.upload_attachment",
    "file_system.compress_directory", "gmail.forward_email_thread",
    "splitwise.accept_group_invitation",
])
def test_before_write_covers_additional_state_mutations(api_name: str) -> None:
    result = BeforeWriteTrigger().detect(context(f"apis.{api_name}()"))
    assert result.should_call_controller
    assert result.reason == "before_write"


@pytest.mark.parametrize("api_name", [
    "amazon.download_order_receipt", "amazon.download_prime_subscription_receipt",
    "amazon.show_last_product_purchase",
    "gmail.download_attachment", "splitwise.download_expense_receipt_file",
    "splitwise.download_payment_receipt_file", "spotify.show_song_review",
    "spotify.show_album_review", "spotify.show_playlist_review",
    "spotify.download_premium_subscription_receipt", "todoist.show_label",
    "todoist.download_attachment", "venmo.download_bank_transfer_receipt",
    "venmo.download_transaction_receipt", "spotify.show_liked_songs",
    "spotify.search_artists", "api_docs.show_api_doc", "spotify.login",
])
def test_before_write_does_not_classify_read_or_login_as_mutation(api_name: str) -> None:
    assert not BeforeWriteTrigger().detect(context(f"apis.{api_name}()")).should_call_controller


def test_before_write_detects_like_inside_loop_but_not_text() -> None:
    trigger = BeforeWriteTrigger()
    code = 'for song in queue:\n    apis.spotify.like_song(song_id=song["song_id"])'
    assert trigger.detect(context(code)).should_call_controller
    assert not trigger.detect(context('# apis.spotify.like_song(song_id=8)')).should_call_controller
    assert not trigger.detect(context('print("apis.spotify.like_song(song_id=8)")')).should_call_controller


def test_before_write_skips_read_only_api_before_write_and_preserves_preview() -> None:
    code = 'apis.spotify.show_song_review(song_id=1)\napis.spotify.like_song(song_id=8)'
    result = BeforeWriteTrigger().detect(context(code))
    assert result.should_call_controller
    assert result.metadata["api_name"] == "like"
    assert result.metadata["params_preview"] == "(song_id=8)"


def test_additional_write_still_obeys_composite_budget() -> None:
    detector = CompositeTriggerDetectorWithBudget(full_config=TriggerConfig(
        budget=TriggerBudgetConfig(detector_order=("before_write",),
                                   max_triggers_per_task=1, min_before_finish_triggers=0)
    ))
    assert detector.detect(context("apis.spotify.like_song(song_id=1)")).should_call_controller
    result = detector.detect(context("apis.spotify.like_song(song_id=2)"))
    assert not result.should_call_controller
    assert result.metadata["trigger_positive"] is True
    assert result.metadata["budget_bypassed"] is True


def test_before_write_keyword_override_remains_effective() -> None:
    trigger = BeforeWriteTrigger(BeforeWriteConfig(high_risk_keywords=frozenset({"delete"})))
    assert trigger.detect(context("apis.spotify.delete_playlist(playlist_id=1)")).should_call_controller
    assert not trigger.detect(context("apis.spotify.like_song(song_id=1)")).should_call_controller


def test_before_finish_ignores_comment_and_string_mentions() -> None:
    trigger = BeforeFinishTrigger()
    assert not trigger.detect(context('# apis.supervisor.complete_task()\nprint("ready")')).should_call_controller
    assert not trigger.detect(context('print("apis.supervisor.complete_task()")')).should_call_controller
    assert trigger.detect(context('print("ready")\napis.supervisor.complete_task()')).should_call_controller


def test_budget_still_records_positive_detection_after_limit() -> None:
    config = TriggerConfig(budget=TriggerBudgetConfig(
        detector_order=("before_write", "before_finish"),
        max_triggers_per_task=1,
        min_before_finish_triggers=1,
    ))
    detector = CompositeTriggerDetectorWithBudget(full_config=config)
    assert detector.detect(context("apis.spotify.delete_playlist(playlist_id=1)")).should_call_controller
    bypass = detector.detect(context("apis.spotify.delete_playlist(playlist_id=2)"))
    assert not bypass.should_call_controller
    assert bypass.metadata["trigger_positive"] is True
    assert bypass.metadata["budget_bypassed"] is True
    assert detector.detect(context("apis.supervisor.complete_task()")).should_call_controller


def test_finish_floor_does_not_depend_on_detector_order() -> None:
    detector = CompositeTriggerDetectorWithBudget(full_config=TriggerConfig(
        budget=TriggerBudgetConfig(detector_order=("before_write", "before_finish"),
                                   max_triggers_per_task=0, min_before_finish_triggers=1)
    ))
    result = detector.detect(context("apis.spotify.delete_playlist(playlist_id=1)\napis.supervisor.complete_task()"))
    assert result.should_call_controller
    assert result.reason == "before_finish"


def test_no_progress_uses_executed_steps_not_model_messages() -> None:
    from controller_agent.integration.controlled_react_agent import ControlledReActAgent

    class Agent:
        task_id = "task-1"
        task = "task"
        step_number = 4

        def __init__(self) -> None:
            self.messages = [{"role": "assistant", "content": "apis.x.read()"}]

    wrapped = ControlledReActAgent(Agent())
    wrapped._agent_context_messages = [{"role": "assistant", "content": "apis.x.read()"}]
    wrapped._trigger_history = [{"proposed_execution": "apis.x.read()", "last_execution_outputs": ["error"]}]
    trigger_context = wrapped._build_trigger_context([SimpleNamespace(content="apis.x.read()")], ["error"])
    assert trigger_context.recent_history == wrapped._trigger_history


def test_repeated_executed_actions_reach_no_progress_detector() -> None:
    from controller_agent.integration.controlled_react_agent import ControlledReActAgent
    from controller_agent.controller.controller import Controller

    class Agent:
        task_id = "task-1"
        task = "task"
        step_number = 1
        messages = [{"role": "user", "content": "task"}]

        def next_execution_inputs_usage_and_status(self, _outputs):
            self.step_number += 1
            return [SimpleNamespace(content="apis.spotify.show_tracks()")], None, SimpleNamespace(failed=False)

    calls: list[object] = []
    detector = CompositeTriggerDetectorWithBudget(full_config=TriggerConfig(
        budget=TriggerBudgetConfig(detector_order=("no_progress",), max_triggers_per_task=4)
    ))
    wrapped = ControlledReActAgent(Agent(), Controller(lambda value: calls.append(value) or {
        "action": "PROCEED", "reason": "continue"
    }), trigger_detector=detector)
    wrapped.next_execution_inputs_usage_and_status([])
    wrapped.next_execution_inputs_usage_and_status(["returned output"])
    wrapped.next_execution_inputs_usage_and_status(["returned output"])
    assert len(calls) == 1
    assert wrapped.logger.trigger_events[-1]["trigger_reason"] == "no_progress"


def test_trigger_ledger_counts_candidates_and_budget_bypasses() -> None:
    from controller_agent.integration.controlled_react_agent import ControlledReActAgent
    from controller_agent.controller.controller import Controller

    class Agent:
        task_id = "task-1"
        task = "task"
        step_number = 1
        messages = [{"role": "user", "content": "task"}]

        def next_execution_inputs_usage_and_status(self, _outputs):
            self.step_number += 1
            return [SimpleNamespace(content="apis.spotify.delete_playlist(playlist_id=1)")], None, SimpleNamespace(failed=False)

    detector = CompositeTriggerDetectorWithBudget(full_config=TriggerConfig(
        budget=TriggerBudgetConfig(detector_order=("before_write",), max_triggers_per_task=1,
                                   min_before_finish_triggers=0)
    ))
    wrapped = ControlledReActAgent(Agent(), Controller(lambda _: {"action": "PROCEED", "reason": "ok"}),
                                   trigger_detector=detector)
    wrapped.next_execution_inputs_usage_and_status([])
    wrapped.next_execution_inputs_usage_and_status(["done"])
    assert len(wrapped.logger.trigger_events) == 2
    assert [row["budget_bypassed"] for row in wrapped.logger.trigger_events] == [False, True]
    assert len(wrapped.logger.events) == 1
    assert summarize_trigger_events(wrapped.logger.trigger_events) == {
        "candidate_actions": 2,
        "trigger_positive_actions": 2,
        "trigger_budget_bypassed_actions": 1,
        "controller_budget_bypassed_actions": 0,
        "budget_bypassed_actions": 1,
    }


@pytest.mark.parametrize("output", [
    "Balance: 500", "404 items found",
    '{"amount": 500, "message": "Payment recorded"}',
    '{"success": true, "message": "No error occurred"}',
    {"message": "Updated successfully", "status_code": 200},
    '{"items": [{"message": "This document describes an error"}]}',
])
def test_successful_outputs_are_not_error_fingerprints(output: object) -> None:
    from controller_agent.triggers.no_progress import _extract_error_fingerprint
    assert _extract_error_fingerprint([output]) is None


@pytest.mark.parametrize("output,expected", [
    ("HTTP status 500", "http:500"),
    ("HTTP/1.1 404 Not Found", "http:404"),
    ('{"status_code": 401, "message": "Unauthorized"}', "http:401"),
    ({"success": False, "message": "Invalid request"}, "exc:invalid request"),
    ('{"error": {"message": "Access denied"}}', "exc:access denied"),
    ("Traceback (most recent call last):\nValueError: invalid value", "exc:valueerror: invalid value"),
])
def test_explicit_errors_keep_fingerprints(output: object, expected: str) -> None:
    from controller_agent.triggers.no_progress import _extract_error_fingerprint
    assert _extract_error_fingerprint([output]) == expected


def test_pagination_and_distinct_literals_are_distinct_actions() -> None:
    from controller_agent.triggers.no_progress import _same_action
    assert not _same_action("apis.x.list(page_index=0)", "apis.x.list(page_index=1)")
    assert not _same_action("apis.x.read(name='A')", "apis.x.read(name='a')")
    assert _same_action("apis.x.read(id=1)", "apis.x.read( id = 1 ) # comment")


def test_one_observation_cannot_count_as_two_repeated_errors() -> None:
    from controller_agent.triggers.no_progress import NoProgressTrigger
    from controller_agent.triggers.config import NoProgressConfig
    detector = NoProgressTrigger(NoProgressConfig(repeated_action_threshold=99))
    step = {"proposed_execution": "apis.x.read()", "last_execution_outputs": ["HTTP status 500"]}
    assert not detector.detect(context("apis.x.read()", [step], ["HTTP status 500"])).should_call_controller
    result = detector.detect(context("apis.x.read()", [step, dict(step)], ["HTTP status 500"]))
    assert result.metadata["stall_kind"] == "repeated_error"


def test_controller_guard_bypass_is_visible_after_four_calls() -> None:
    from controller_agent.controller.controller import Controller
    from controller_agent.integration.controlled_react_agent import ControlledReActAgent
    from controller_agent.state.schemas import ControllerConfig
    from controller_agent.integration.base import OfflineMockAdapter
    detector = CompositeTriggerDetectorWithBudget(full_config=TriggerConfig(
        budget=TriggerBudgetConfig(max_triggers_per_task=100)
    ))
    calls = []
    wrapped = ControlledReActAgent(
        SimpleNamespace(), Controller(lambda value: calls.append(value) or {"action": "PROCEED", "reason": "ok"}),
        trigger_detector=detector, integration_adapter=OfflineMockAdapter(),
        config=ControllerConfig(max_controller_calls_per_task=4, max_controller_blocks=4),
    )
    for _ in range(5):
        assert wrapped.inspect_proposed_execution(context("apis.supervisor.complete_task()"), (), []).allow_execution
    assert len(calls) == 4
    summary = summarize_trigger_events(wrapped.logger.trigger_events)
    assert summary["trigger_positive_actions"] == 5
    assert summary["controller_budget_bypassed_actions"] == 1
    assert summary["budget_bypassed_actions"] == 1
    assert summary["trigger_budget_bypassed_actions"] == 0
