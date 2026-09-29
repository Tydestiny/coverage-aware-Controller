"""Parsing and endpoint credential tests for the live Controller backend."""

from controller_agent.controller.live_llm_backend import (
    LiveLLMBackend,
    call_model,
    parse_response_json,
    remove_confidence_field,
    resolve_api_key,
)
from controller_agent.state.schemas import ControllerInput


def test_parse_response_json_accepts_prose_prefixed_json() -> None:
    payload = parse_response_json('Reasoning follows. {"action":"REPLAN","reason":"r","instruction":"i"}')
    assert payload["action"] == "REPLAN"


def test_parse_response_json_accepts_fenced_json() -> None:
    payload = parse_response_json('```json\n{"action":"PROCEED","reason":"r"}\n```')
    assert payload["action"] == "PROCEED"


def test_remove_confidence_field_removes_nested_and_top_level_values() -> None:
    payload = remove_confidence_field({"confidence": 0.9, "decision": {"action": "PROCEED", "confidence": 0.5}})
    assert payload == {"decision": {"action": "PROCEED"}}


def test_dashscope_endpoint_prefers_dashscope_key(monkeypatch: object) -> None:
    """A DashScope request must not accidentally use an OpenRouter key."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")  # type: ignore[attr-defined]
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")  # type: ignore[attr-defined]

    assert resolve_api_key("https://dashscope.aliyuncs.com/compatible-mode/v1") == "dashscope-key"


def test_openrouter_endpoint_prefers_openrouter_key(monkeypatch: object) -> None:
    """An OpenRouter request selects the OpenRouter credential."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")  # type: ignore[attr-defined]
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")  # type: ignore[attr-defined]

    assert resolve_api_key("https://openrouter.ai/api/v1") == "openrouter-key"


def test_live_backend_loads_explicit_system_prompt(tmp_path) -> None:
    """A versioned run can select its declared Controller system prompt."""
    prompt = tmp_path / "g1_prompt.txt"
    prompt.write_text("G1 system prompt", encoding="utf-8")
    captured = {}
    backend = LiveLLMBackend(
        system_prompt_path=prompt,
        call_model_fn=lambda **kwargs: captured.update(kwargs) or {"text": '{"action":"PROCEED","reason":"ok"}'},
    )
    backend.decide(
        ControllerInput("task", "before_finish", {}, [], "output", "code", [], True)
    )

    assert captured["system_prompt"] == "G1 system prompt"


def test_live_backend_renders_controller_input_in_explicit_model_visible_order(tmp_path) -> None:
    """The actual user prompt places current task and proposed action at the required positions."""
    captured = {}
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Development prompt", encoding="utf-8")
    backend = LiveLLMBackend(
        system_prompt_path=prompt_file,
        call_model_fn=lambda **kwargs: captured.update(kwargs) or {"text": '{"action":"PROCEED","reason":"ok"}'},
    )
    backend.decide(
        ControllerInput(
            task="TASK_VALUE",
            trigger_reason="TRIGGER_VALUE",
            trigger_metadata={"metadata": "METADATA_VALUE"},
            agent_context_messages=[{"role": "user", "content": "HISTORY_VALUE"}],
            proposed_agent_output="OUTPUT_VALUE",
            proposed_execution="EXECUTION_VALUE",
            previous_controller_decisions=[{"action": "DECISION_VALUE"}],
            context_complete=True,
            context_metadata={"scope": "current_task", "request_sha256": "synthetic-hash"},
        )
    )

    prompt = captured["user_prompt"]
    headings = (
        "## task",
        "## trigger_reason",
        "## trigger_metadata",
        "## context_complete",
        "## context_metadata",
        "## agent_context_messages",
        "## previous_controller_decisions",
        "## proposed_agent_output",
        "## proposed_execution",
    )
    assert [prompt.index(heading) for heading in headings] == sorted(prompt.index(heading) for heading in headings)
    assert prompt.index("TASK_VALUE") < prompt.index("HISTORY_VALUE") < prompt.index("OUTPUT_VALUE") < prompt.index("EXECUTION_VALUE")
    assert "case_type_hint" not in prompt
    assert "synthetic-hash" not in prompt
    assert "current_task" in prompt


def test_live_backend_requires_explicit_prompt() -> None:
    """Historical prompts must not be selected implicitly."""
    import pytest

    backend = LiveLLMBackend(call_model_fn=lambda **kwargs: pytest.fail("Model called"))
    with pytest.raises(ValueError, match="versioned controller system prompt"):
        backend.decide(ControllerInput("task", "before_finish", {}, [], "output", "code", [], True))


def test_deepseek_v4_controller_explicitly_disables_thinking(monkeypatch: object) -> None:
    """DashScope DeepSeek V4 Controller calls must not use default thinking mode."""
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"model":"deepseek-v4-pro","choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(request: object, timeout: float) -> Response:
        captured["payload"] = __import__("json").loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-pro",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
    )

    assert captured["payload"]["enable_thinking"] is False
