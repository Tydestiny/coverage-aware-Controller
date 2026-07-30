"""Tests for the project-local DeepSeek no-thinking runtime adapter."""

from __future__ import annotations

import sys
from types import ModuleType

from controller_agent.integration.dashscope_no_thinking import (
    install_deepseek_no_thinking_adapter,
    install_qwen36_plus_no_thinking_adapter,
)


def test_adapter_is_safe_when_imported_twice(monkeypatch: object) -> None:
    """The runtime patch is idempotent for one worker process."""
    package = ModuleType("appworld_agents")
    code = ModuleType("appworld_agents.code")
    simplified = ModuleType("appworld_agents.code.simplified")
    language_model = ModuleType("appworld_agents.code.simplified.language_model")
    language_model.non_cached_lm_call = lambda **kwargs: {"original": kwargs}  # type: ignore[attr-defined]
    language_model.to_dict = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents", package)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code", code)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code.simplified", simplified)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code.simplified.language_model", language_model)  # type: ignore[attr-defined]

    install_deepseek_no_thinking_adapter()
    first = language_model.non_cached_lm_call
    install_deepseek_no_thinking_adapter()

    assert language_model.non_cached_lm_call is first


def test_qwen_and_deepseek_share_one_runtime_patch(monkeypatch: object) -> None:
    """Both supported models use one process-local runtime patch."""
    package = ModuleType("appworld_agents")
    code = ModuleType("appworld_agents.code")
    simplified = ModuleType("appworld_agents.code.simplified")
    language_model = ModuleType("appworld_agents.code.simplified.language_model")
    language_model.non_cached_lm_call = lambda **kwargs: {"original": kwargs}  # type: ignore[attr-defined]
    language_model.to_dict = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents", package)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code", code)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code.simplified", simplified)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld_agents.code.simplified.language_model", language_model)  # type: ignore[attr-defined]

    install_qwen36_plus_no_thinking_adapter()
    first = language_model.non_cached_lm_call
    install_deepseek_no_thinking_adapter()

    assert language_model.non_cached_lm_call is first
