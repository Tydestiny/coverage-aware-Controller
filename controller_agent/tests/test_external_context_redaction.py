"""Tests for checkpoint context redaction before external model calls."""

from scripts.run_controller_api_smoke import sanitize_messages
from scripts.run_live_controller_checkpoint_smoke import (
    _candidate_agent_output,
    _install_private_context_variables,
    redact_external_context,
)


def test_external_context_redacts_profile_and_credentials() -> None:
    value = {
        "content": (
            "Email person@example.com, phone 4675246489, "
            "password='private-pass', access_token='private-token', artist_id=123."
        )
    }

    redacted = redact_external_context(value)["content"]

    assert "person@example.com" not in redacted
    assert "4675246489" not in redacted
    assert "private-pass" not in redacted
    assert "private-token" not in redacted
    assert "artist_id=123" in redacted


def test_external_context_preserves_credential_variable_references() -> None:
    content = (
        "login(password=spotify_password)\n"
        "follow(access_token=spotify_access_token)\n"
        "access_token='literal-token'\n"
        '"password": "literal-password"'
    )

    redacted = redact_external_context(content)

    assert "password=spotify_password" in redacted
    assert "access_token=spotify_access_token" in redacted
    assert "literal-token" not in redacted
    assert "literal-password" not in redacted


def test_log_sanitizer_preserves_credential_variable_references() -> None:
    content = "follow(access_token=spotify_access_token); access_token='literal-token'"

    redacted = sanitize_messages(content)

    assert "access_token=spotify_access_token" in redacted
    assert "literal-token" not in redacted


def test_runtime_pre_action_record_becomes_executable_candidate() -> None:
    output = _candidate_agent_output(
        {"proposed_execution": "apis.supervisor.complete_task(answer=None)"}
    )

    assert output == "```python\napis.supervisor.complete_task(answer=None)\n```\n"


class _FakeShell:
    def __init__(self) -> None:
        self.user_ns: dict[str, str] = {}


class _FakeWorld:
    def __init__(self) -> None:
        self.shell = _FakeShell()


def test_private_profile_values_stay_local_and_become_variable_references() -> None:
    world = _FakeWorld()
    checkpoint = {
        "agent_messages": [
            {
                "role": "user",
                "content": (
                    "My personal email is: person@example.com and phone number is 1234567890.\n"
                    "Task: Follow the artists."
                ),
            },
            {
                "role": "assistant",
                "content": "login(username='person@example.com', password=spotify_password)",
            },
        ]
    }

    private_context = _install_private_context_variables(world, checkpoint)
    redacted = redact_external_context(
        checkpoint["agent_messages"],
        private_context=private_context,
    )

    assert private_context == {
        "supervisor_email": "person@example.com",
        "supervisor_phone": "1234567890",
    }
    assert world.shell.user_ns == private_context
    assert "person@example.com" not in str(redacted)
    assert "1234567890" not in str(redacted)
    assert "username=supervisor_email" in redacted[1]["content"]
