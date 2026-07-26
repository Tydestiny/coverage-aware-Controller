"""Runtime-only DashScope compatibility for DeepSeek non-thinking calls."""

from __future__ import annotations

from typing import Any


DEEPSEEK_V4_FLASH = "deepseek-v4-flash"


def install_deepseek_no_thinking_adapter() -> None:
    """Patch AppWorld's in-process caller without editing AppWorld source files."""
    from appworld_agents.code.simplified import language_model

    if getattr(language_model, "_controller_deepseek_no_thinking_installed", False):
        return
    original = language_model.non_cached_lm_call

    def patched_non_cached_lm_call(**kwargs: Any) -> dict[str, Any]:
        """Send DeepSeek V4 Flash with the workspace's disable-thinking payload."""
        if kwargs.get("model") != DEEPSEEK_V4_FLASH:
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
    language_model._controller_deepseek_no_thinking_installed = True
