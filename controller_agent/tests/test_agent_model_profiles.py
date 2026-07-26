"""Tests for fixed Agent provider profiles used by live branch runners."""

from scripts.run_controller_api_smoke import (
    API_ENDPOINT,
    OPENAI_ENDPOINT,
    OPENAI_GPT4O_MINI,
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V4_FLASH_ENDPOINT,
    build_agent_model_config,
)


def test_gpt_4o_mini_uses_openai_profile() -> None:
    """GPT-4o mini must use OpenAI credentials and endpoint."""
    config = build_agent_model_config("openai/gpt-4o-mini")

    assert config["name"] == OPENAI_GPT4O_MINI
    assert config["base_url"] == OPENAI_ENDPOINT
    assert config["api_key_env_name"] == "OPENAI_API_KEY"


def test_qwen_profile_remains_on_dashscope() -> None:
    """Existing Qwen Agent runs retain their DashScope configuration."""
    config = build_agent_model_config("qwen3-coder-flash-2025-07-28")

    assert config["base_url"] == API_ENDPOINT
    assert config["api_key_env_name"] == "DASHSCOPE_API_KEY"


def test_deepseek_v4_flash_disables_thinking_via_openai_compatible_mode() -> None:
    """DeepSeek V4 Flash uses the documented non-thinking compatibility value."""
    config = build_agent_model_config(DEEPSEEK_V4_FLASH)

    assert config["base_url"] == DEEPSEEK_V4_FLASH_ENDPOINT
    assert config["api_key_env_name"] == "DASHSCOPE_API_KEY"
    assert config["drop_reasoning_content"] is True
