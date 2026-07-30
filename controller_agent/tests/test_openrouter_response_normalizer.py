"""Tests for the OpenRouter runtime response compatibility adapter."""

from controller_agent.integration.openrouter_response_normalizer import normalize_reasoning_content


def test_normalize_reasoning_content_replaces_only_null_reasoning_values() -> None:
    """Null reasoning fields become empty strings for the AppWorld agent."""
    original = {
        "reasoning_content": None,
        "choices": [{"message": {"reasoning_content": None, "content": "code"}}],
        "other": None,
    }

    normalized = normalize_reasoning_content(original)

    assert normalized["reasoning_content"] == ""
    assert normalized["choices"][0]["message"]["reasoning_content"] == ""
    assert normalized["other"] is None
    assert original["reasoning_content"] is None
