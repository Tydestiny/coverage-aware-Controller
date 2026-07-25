"""Parsing tests for the live Controller backend."""

from controller_agent.controller.live_llm_backend import remove_confidence_field, parse_response_json


def test_parse_response_json_accepts_prose_prefixed_json() -> None:
    payload = parse_response_json('Reasoning follows. {"action":"REPLAN","reason":"r","instruction":"i"}')
    assert payload["action"] == "REPLAN"


def test_parse_response_json_accepts_fenced_json() -> None:
    payload = parse_response_json('```json\n{"action":"PROCEED","reason":"r"}\n```')
    assert payload["action"] == "PROCEED"


def test_remove_confidence_field_removes_nested_and_top_level_values() -> None:
    payload = remove_confidence_field({"confidence": 0.9, "decision": {"action": "PROCEED", "confidence": 0.5}})
    assert payload == {"decision": {"action": "PROCEED"}}
