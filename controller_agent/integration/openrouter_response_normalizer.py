"""Normalize OpenRouter response fields for AppWorld's simplified agent."""

from __future__ import annotations

from types import MethodType
from typing import Any


OPENROUTER_GPT4O_MINI = "openai/gpt-4o-mini"
OPENROUTER_GPT4O = "openai/gpt-4o"


def normalize_reasoning_content(value: Any) -> Any:
    """Replace null reasoning-content values without changing other response data."""
    if isinstance(value, dict):
        return {
            key: "" if key == "reasoning_content" and item is None else normalize_reasoning_content(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_reasoning_content(item) for item in value]
    return value


def install_openrouter_response_normalizer() -> None:
    """Patch only the in-process AppWorld call path for OpenRouter GPT-4o mini."""
    from appworld_agents.code.simplified import language_model

    if getattr(language_model, "_controller_openrouter_normalizer_installed", False):
        return
    original = language_model.non_cached_lm_call

    def normalized_non_cached_lm_call(**kwargs: Any) -> Any:
        response = original(**kwargs)
        if kwargs.get("model") not in {OPENROUTER_GPT4O_MINI, OPENROUTER_GPT4O}:
            return response
        return normalize_reasoning_content(response)

    language_model.non_cached_lm_call = normalized_non_cached_lm_call
    language_model._controller_openrouter_normalizer_installed = True


def install_agent_generate_normalizer(agent: Any) -> None:
    """Normalize OpenRouter output after AppWorld's language-model processing."""
    language_model = getattr(agent, "language_model", None)
    if language_model is None or getattr(language_model, "_controller_openrouter_generate_normalized", False):
        return
    original = language_model.generate

    def normalized_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return normalize_reasoning_content(original(*args, **kwargs))

    language_model.generate = MethodType(normalized_generate, language_model)
    language_model._controller_openrouter_generate_normalized = True
