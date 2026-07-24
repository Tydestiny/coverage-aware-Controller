"""Unit tests for LiveLLMBackend (no live API calls)."""

from __future__ import annotations

import json

from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend, strip_code_fences
from controller_agent.state.schemas import Action, ControllerInput


def _inp(**kwargs) -> ControllerInput:
    base = dict(
        task="Remove songs that are neither liked nor downloaded.",
        trigger_reason="before_first_write",
        recent_history=[],
        current_agent_output="apis.spotify.remove_song_from_library(song_id=1, access_token=t)",
        known_entities=[],
        completed_operations=[],
        recent_errors=[],
    )
    base.update(kwargs)
    return ControllerInput(**base)


def test_strip_code_fences() -> None:
    raw = '```json\n{"action":"PROCEED","reason":"ok","instruction":"","confidence":0.9}\n```'
    assert strip_code_fences(raw).startswith("{")


def test_live_backend_returns_text_via_call() -> None:
    payload = {
        "action": "VERIFY_TARGET_SET",
        "reason": "missing remove_set",
        "instruction": "Print remove_set before deleting.",
        "confidence": 0.8,
    }

    def fake_call(**kwargs):
        assert "Coverage-Aware Controller" in kwargs["system_prompt"]
        assert "before_first_write" in kwargs["user_prompt"]
        return {"text": json.dumps(payload), "model_name": "mock-model", "input_tokens": 10, "output_tokens": 5}

    backend = LiveLLMBackend(call_model_fn=fake_call)
    text = backend(_inp())
    assert json.loads(text)["action"] == "VERIFY_TARGET_SET"

    ctrl = Controller(backend=backend)
    resp = ctrl.decide(_inp())
    assert resp.decision.action == Action.VERIFY_TARGET_SET
    assert "remove_set" in resp.decision.instruction


def test_live_backend_decide_attaches_usage() -> None:
    def fake_call(**kwargs):
        return {
            "text": json.dumps(
                {
                    "action": "PROCEED",
                    "reason": "already verified",
                    "instruction": "",
                    "confidence": 0.7,
                }
            ),
            "model_name": "mock-model",
            "input_tokens": 11,
            "output_tokens": 3,
            "latency_seconds": 0.12,
        }

    resp = LiveLLMBackend(call_model_fn=fake_call).decide(_inp())
    assert resp.decision.action == Action.PROCEED
    assert resp.model_name == "mock-model"
    assert resp.input_tokens == 11
    assert resp.output_tokens == 3
    assert resp.latency_seconds == 0.12
    assert resp.raw_response is not None


def test_invalid_json_fails_open_through_controller() -> None:
    def fake_call(**kwargs):
        return {"text": "not-json", "model_name": "mock-model"}

    ctrl = Controller(backend=LiveLLMBackend(call_model_fn=fake_call))
    resp = ctrl.decide(_inp())
    assert resp.decision.action == Action.PROCEED
    assert resp.decision.reason == "controller_failure_fallback"


def test_provider_error_fails_open() -> None:
    def fake_call(**kwargs):
        raise RuntimeError("LLM HTTP 429: rate limit")

    ctrl = Controller(backend=LiveLLMBackend(call_model_fn=fake_call))
    resp = ctrl.decide(_inp())
    assert resp.decision.action == Action.PROCEED
    assert "RuntimeError" in (resp.decision.instruction or "")


def test_fenced_json_accepted() -> None:
    def fake_call(**kwargs):
        return {
            "text": '```json\n{"action":"REPLAN","reason":"recover","instruction":"Re-plan recovery.","confidence":0.6}\n```',
            "model_name": "mock-model",
        }

    resp = Controller(backend=LiveLLMBackend(call_model_fn=fake_call)).decide(_inp())
    assert resp.decision.action == Action.REPLAN
