"""Tests for deterministic ControllerInput construction."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from controller_agent.state.builder import StateBuilder
from controller_agent.state.schemas import TriggerResult


def _build(
    history: list[dict[str, str]],
    *,
    mode: str = "summary",
    current: str = "",
):
    """Build one ControllerInput with a stable trigger fixture."""
    return StateBuilder(api_history_mode=mode).build(
        task="Task text.",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=history,
        current_agent_output=current,
    )


def _records(result) -> list[dict[str, object]]:
    """Decode stable API history strings for assertions."""
    return [json.loads(value) for value in result.completed_operations]


def _before_recent_window(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Move asserted API calls before the raw recent-history window."""
    return history + [{"role": "user", "content": f"later-{index}"} for index in range(12)]


def test_known_entities_are_always_empty() -> None:
    """Entity extraction is intentionally disabled pending a narrower contract."""
    result = _build(
        [
            {"role": "assistant", "content": "```python\napis.spotify.show_playlist(playlist_id=12)\n```"},
            {"role": "user", "content": "{\"playlist_id\": 12}"},
        ],
        current="```python\napis.phone.send_message(contact_id=8)\n```",
    )

    assert result.known_entities == []


def test_summary_api_history_omits_specific_results() -> None:
    """Summary records types and counts without retaining returned items."""
    result = _build(
            _before_recent_window([
                {"role": "assistant", "content": "```python\napis.spotify.show_recommendations(page_index=0)\n```"},
                {"role": "user", "content": "[{\"song_id\": 1, \"title\": \"secret song\"}]"},
            ])
    )
    record = _records(result)[0]

    assert record == {
        "api": "spotify.show_recommendations",
        "parameters": {"page_index": "0"},
        "return_count": 1,
        "return_type": "list",
        "status": "success",
        "step": 0,
    }
    assert "secret song" not in result.completed_operations[0]


def test_structured_api_history_compresses_results_and_redacts_sensitive_fields() -> None:
    """Structured mode keeps bounded output with explicit truncation markers."""
    payload = {
        "access_token": "must-not-leak",
        "items": [{"name": "x" * 300}] * 5,
    }
    result = _build(
            _before_recent_window([
                {"role": "assistant", "content": "```python\napis.spotify.show_recommendations()\n```"},
                {"role": "user", "content": json.dumps(payload)},
            ]),
        mode="structured",
    )
    structured = _records(result)[0]["result"]

    assert structured["access_token"] == "<redacted>"
    assert structured["items"]["count"] == 5
    assert structured["items"]["truncated"] is True
    assert structured["items"]["items"][0]["name"]["__truncated__"] is True
    assert "must-not-leak" not in result.completed_operations[0]


def test_api_catalog_records_all_names_and_descriptions() -> None:
    """Documentation catalog calls preserve all returned API facts."""
    catalog = [
        {"name": "show_playlist", "description": "Show one playlist."},
        {"name": "delete_playlist", "description": "Delete one playlist."},
    ]
    result = _build(
            _before_recent_window([
                {"role": "assistant", "content": "```python\napis.api_docs.show_api_descriptions(app_name='spotify')\n```"},
                {"role": "user", "content": json.dumps(catalog)},
            ])
    )
    record = _records(result)[0]

    assert record["catalog"] == {"apis": catalog}
    assert [item["name"] for item in record["catalog"]["apis"]] == ["show_playlist", "delete_playlist"]


def test_app_catalog_records_all_names_and_descriptions() -> None:
    """Application catalog calls preserve every returned application entry."""
    catalog = [
        {"name": "spotify", "description": "Music."},
        {"name": "phone", "description": "Messaging."},
    ]
    result = _build(
            _before_recent_window([
                {"role": "assistant", "content": "```python\napis.api_docs.show_app_descriptions()\n```"},
                {"role": "user", "content": json.dumps(catalog)},
            ])
    )

    assert _records(result)[0]["catalog"] == {"apps": catalog}


def test_api_doc_records_description_parameters_and_return_structure() -> None:
    """Individual API docs retain schema facts rather than one selected field."""
    document = {
        "description": "Get recommendations.",
        "parameters": [{"name": "access_token"}, {"name": "page_index"}],
        "response_schemas": {"success": [{"song_id": 1}]},
    }
    result = _build(
            _before_recent_window([
                {"role": "assistant", "content": "```python\napis.api_docs.show_api_doc(app_name='spotify', api_name='show_recommendations')\n```"},
                {"role": "user", "content": json.dumps(document)},
            ])
    )
    catalog = _records(result)[0]["catalog"]

    assert catalog["description"] == "Get recommendations."
    assert catalog["parameters"]["items"][0]["name"] == "access_token"
    assert catalog["return_structure"]["success"]["items"][0]["song_id"] == 1


def test_history_is_collected_before_recent_window_truncation() -> None:
    """Early API actions stay in API history even when absent from recent messages."""
    history = [
        {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
        {"role": "user", "content": "{\"artist_id\": 1}"},
    ]
    history.extend({"role": "user", "content": f"later-{index}"} for index in range(12))

    result = _build(history)

    assert len(result.recent_history) == 12
    assert _records(result)[0]["api"] == "spotify.show_artist"


def test_current_task_boundary_excludes_single_shot_history() -> None:
    """Only messages after the latest explicit task enter derived task state."""
    history = [
        {"role": "user", "content": "Task: Single-shot example task."},
        {"role": "assistant", "content": "```python\napis.single_shot.show_example()\n```"},
        {"role": "user", "content": "HTTP status 500: single-shot failure"},
        {
            "role": "user",
            "content": "Use the APIs to solve the actual task:\nProfile: Kyle\nTask: Real Spotify task.",
        },
        {"role": "assistant", "content": "```python\napis.spotify.show_recommendations()\n```"},
        {"role": "user", "content": "[{\"artist\": \"A\"}]"},
        {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=2)\n```"},
        {"role": "user", "content": "HTTP status 404: current-task missing artist"},
    ]
    history.extend({"role": "user", "content": f"later-{index}"} for index in range(12))

    result = _build(history)
    records = _records(result)

    assert result.task == "Real Spotify task."
    assert [record["api"] for record in records] == [
        "spotify.show_recommendations",
        "spotify.show_artist",
    ]
    assert result.recent_errors == ["HTTP status 404: current-task missing artist"]
    assert all("single-shot" not in message["content"] for message in result.recent_history)
    assert all("single_shot" not in operation for operation in result.completed_operations)


def test_history_without_explicit_task_uses_full_history() -> None:
    """Ordinary histories retain the prior full-history behavior as fallback."""
    history = [
        {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
        {"role": "user", "content": "[{\"artist_id\": 1}]"},
    ]

    result = _build(history)

    assert result.task == "Task text."
    assert result.completed_operations == []
    assert result.recent_history == history


def test_task_selection_ignores_agent_phrase_before_code_block() -> None:
    """An Agent saying 'complete the task:' must not replace the user task."""
    history = [
        {
            "role": "user",
            "content": (
                "Using these APIs, now generate code to solve the actual task:\n\n"
                "Task: Follow all classical artists with at least 22 followers."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Perfect! I have completed the task:\n\n"
                "```python\napis.supervisor.complete_task(answer='done')\n```"
            ),
        },
    ]

    result = _build(history, current="apis.supervisor.complete_task(answer='done')")

    assert result.task == "Follow all classical artists with at least 22 followers."
    assert len(result.recent_history) == 1
    assert "complete_task" in result.recent_history[0]["content"]


def test_recent_history_is_redacted_without_mutating_source() -> None:
    """Message-window secrets are redacted while variable references remain usable."""
    history = [
        {"role": "user", "content": "Task: Real task."},
        {
            "role": "assistant",
            "content": "```python\npassword='actual-password'\naccess_token = 'actual-token'\napis.spotify.login(password=password)\nprint(login_result['access_token'])\n```",
        },
        {"role": "user", "content": '{"access_token": "returned-token", "secret": "nested-secret"}'},
    ]
    original = deepcopy(history)

    result = _build(history)
    rendered = json.dumps(result.recent_history)

    assert "actual-password" not in rendered
    assert "actual-token" not in rendered
    assert "returned-token" not in rendered
    assert "nested-secret" not in rendered
    assert "login_result['access_token']" in rendered
    assert history == original


def test_completed_operations_exclude_calls_in_recent_history_window() -> None:
    """The compact API history contains only task actions preceding the raw window."""
    history = [{"role": "user", "content": "Task: Real task."}]
    for index in range(8):
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": f"```python\napis.spotify.show_artist(artist_id={index})\n```",
                },
                {"role": "user", "content": f'{{"artist_id": {index}}}'},
            ]
        )

    result = _build(history)
    records = _records(result)

    assert len(result.recent_history) == 12
    assert [record["parameters"]["artist_id"] for record in records] == ["0", "1"]
    assert "artist_id=2" in result.recent_history[0]["content"]


def test_recent_errors_use_complete_task_history_after_task_boundary() -> None:
    """Errors remain complete task facts even when their calls predate the raw window."""
    history = [{"role": "user", "content": "Task: Real task."}]
    history.extend(
        [
            {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
            {"role": "user", "content": "HTTP status 404: early task error"},
        ]
    )
    history.extend({"role": "user", "content": f"later-{index}"} for index in range(12))

    result = _build(history)

    assert result.recent_errors == ["HTTP status 404: early task error"]


def test_multiple_calls_with_one_observation_are_unknown() -> None:
    """One result is never copied across multiple calls from the same action."""
    result = _build(
            _before_recent_window([
                {
                    "role": "assistant",
                    "content": "```python\napis.spotify.show_artist(artist_id=1)\napis.spotify.show_artist(artist_id=2)\n```",
                },
                {"role": "user", "content": "[{\"artist_id\": 1}]"},
            ])
    )

    records = _records(result)
    assert [record["status"] for record in records] == ["unknown", "unknown"]
    assert all(record["return_type"] is None for record in records)


def test_current_output_does_not_create_api_history() -> None:
    """Only executed history, never a pending current output, enters API records."""
    result = _build([], current="```python\napis.spotify.delete_playlist(playlist_id=12)\n```")

    assert result.completed_operations == []


def test_errors_require_executed_observation_and_http_context() -> None:
    """Task wording and bare amounts cannot produce an error record."""
    result = _build(
        [
            {"role": "user", "content": "Task says error 500 and amount 500."},
            {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
            {"role": "user", "content": "The amount is 500 dollars."},
            {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=2)\n```"},
            {"role": "user", "content": "HTTP status 500: server failure"},
        ]
    )

    assert result.recent_errors == ["HTTP status 500: server failure"]


def test_recent_history_preserves_leading_action_observation_pair() -> None:
    """Truncation can retain one extra action instead of an orphan observation."""
    history = [{"role": "user", "content": "old"}]
    history.extend(
        [
            {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
            {"role": "user", "content": "{\"artist_id\": 1}"},
        ]
    )
    history.extend({"role": "user", "content": f"later-{index}"} for index in range(11))

    result = _build(history)

    assert len(result.recent_history) == 13
    assert [message["role"] for message in result.recent_history[:2]] == ["assistant", "user"]


def test_build_does_not_mutate_original_history() -> None:
    """History ownership remains with the caller after construction."""
    history: list[dict[str, object]] = [
        {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```", "metadata": {"x": 1}},
        {"role": "user", "content": "{\"artist_id\": 1}", "metadata": {"x": 2}},
    ]
    original = deepcopy(history)

    result = StateBuilder().build(
        task=" Task ",
        trigger=TriggerResult(True, "before_finish"),
        recent_history=history,  # type: ignore[arg-type]
        current_agent_output="Output",
    )
    history[0]["metadata"]["x"] = 99  # type: ignore[index]

    assert original[0]["metadata"] == {"x": 1}
    assert result.recent_history == [
        {"role": "assistant", "content": "```python\napis.spotify.show_artist(artist_id=1)\n```"},
        {"role": "user", "content": "{\"artist_id\": 1}"},
    ]


def test_invalid_api_history_mode_is_rejected() -> None:
    """The two documented API history modes are the only accepted modes."""
    with pytest.raises(ValueError, match="api_history_mode"):
        StateBuilder(api_history_mode="verbose")
