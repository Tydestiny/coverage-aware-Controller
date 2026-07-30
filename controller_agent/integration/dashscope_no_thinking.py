"""Runtime-only DashScope compatibility for non-thinking model calls."""

from __future__ import annotations

from typing import Any


DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
QWEN36_PLUS = "qwen3.6-plus"
_NO_THINKING_MODELS = frozenset({DEEPSEEK_V4_FLASH, QWEN36_PLUS})


def install_deepseek_no_thinking_adapter() -> None:
    """Patch AppWorld's in-process caller without editing AppWorld source files."""
    _install_no_thinking_adapter()


def install_qwen36_plus_no_thinking_adapter() -> None:
    """Send Qwen 3.6 Plus through DashScope with thinking disabled."""
    _install_no_thinking_adapter()


def _install_no_thinking_adapter() -> None:
    """Install one idempotent adapter for supported DashScope models."""
    from appworld_agents.code.simplified import language_model

    if getattr(language_model, "_controller_dashscope_no_thinking_installed", False):
        return
    original = language_model.non_cached_lm_call

    def patched_non_cached_lm_call(**kwargs: Any) -> dict[str, Any]:
        """Send supported models with DashScope's explicit disable-thinking payload."""
        if kwargs.get("model") not in _NO_THINKING_MODELS:
            return original(**kwargs)
        from openai import OpenAI

        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        kwargs.pop("client_name", None)
        kwargs.pop("api_type", None)
        kwargs.pop("make_cacheable", None)
        kwargs.pop("reasoning_effort", None)
        response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            **kwargs,
            extra_body={"enable_thinking": False},
        )
        return language_model.to_dict(response)

    language_model.non_cached_lm_call = patched_non_cached_lm_call
    language_model._controller_dashscope_no_thinking_installed = True
