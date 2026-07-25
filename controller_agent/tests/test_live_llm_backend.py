"""Parsing tests for the live Controller backend."""

import json

from controller_agent.controller.live_llm_backend import (
    LiveLLMBackend,
    remove_confidence_field,
    parse_response_json,
)
from controller_agent.state.schemas import Action, ControllerInput


def test_parse_response_json_accepts_prose_prefixed_json() -> None:
    payload = parse_response_json('Reasoning follows. {"action":"REPLAN","reason":"r","instruction":"i"}')
    assert payload["action"] == "REPLAN"


def test_parse_response_json_accepts_fenced_json() -> None:
    payload = parse_response_json('```json\n{"action":"PROCEED","reason":"r"}\n```')
    assert payload["action"] == "PROCEED"


def test_remove_confidence_field_removes_nested_and_top_level_values() -> None:
    payload = remove_confidence_field({"confidence": 0.9, "decision": {"action": "PROCEED", "confidence": 0.5}})
    assert payload == {"decision": {"action": "PROCEED"}}


def test_live_backend_uses_configured_system_prompt(tmp_path) -> None:
    prompt_path = tmp_path / "g1_system.txt"
    prompt_path.write_text("custom diagnosis-first G1 prompt", encoding="utf-8")
    controller_input = ControllerInput(
        task="Finish the visible task.",
        trigger_reason="before_finish",
        recent_history=[],
        current_agent_output="apis.supervisor.complete_task()",
        known_entities=[],
        completed_operations=[],
        recent_errors=[],
    )

    def fake_call(**kwargs):
        assert kwargs["system_prompt"] == "custom diagnosis-first G1 prompt"
        assert "before_finish" in kwargs["user_prompt"]
        return {
            "text": json.dumps(
                {
                    "action": "VERIFY_TARGET_SET",
                    "reason": "One required source is not established.",
                    "instruction": "Enumerate the missing source before completing.",
                }
            ),
            "model_name": "mock-controller",
        }

    response = LiveLLMBackend(
        system_prompt_path=prompt_path,
        call_model_fn=fake_call,
    ).decide(controller_input)

    assert response.decision.action == Action.VERIFY_TARGET_SET
    assert response.decision.reason == "One required source is not established."
    assert response.decision.instruction == "Enumerate the missing source before completing."


def test_live_backend_preserves_g0_as_the_default_prompt() -> None:
    controller_input = ControllerInput(
        task="Finish the visible task.",
        trigger_reason="before_finish",
        recent_history=[],
        current_agent_output="apis.supervisor.complete_task()",
        known_entities=[],
        completed_operations=[],
        recent_errors=[],
    )

    def fake_call(**kwargs):
        assert "You are the Coverage-Aware Controller" in kwargs["system_prompt"]
        return {
            "text": json.dumps(
                {
                    "action": "PROCEED",
                    "reason": "Visible evidence supports completion.",
                    "instruction": "",
                }
            ),
            "model_name": "mock-controller",
        }

    response = LiveLLMBackend(call_model_fn=fake_call).decide(controller_input)

    assert response.decision.action == Action.PROCEED
