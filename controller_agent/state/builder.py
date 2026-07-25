"""Deterministic construction of Controller input from visible state."""

from __future__ import annotations

import ast
import copy
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from controller_agent.state.schemas import ControllerInput, TriggerResult


_SENSITIVE_NAME_RE = re.compile(
    r"(?:access_?token|token|password|passwd|secret|api_?key|authorization|credential)",
    re.IGNORECASE,
)
_QUERY_API_RE = re.compile(r"(?:^|\.)(?:show|get|list|search)_", re.IGNORECASE)
_WRITE_SUCCESS_RE = re.compile(r"\b(?:execution successful|succeeded|successfully completed)\b", re.IGNORECASE)
_HTTP_STATUS_RE = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:_code)?)\s*[:=]?\s*(400|401|403|404|409|422|500)\b",
    re.IGNORECASE,
)
_EXPLICIT_ERROR_RE = re.compile(r"\b(?:exception|error|failed|failure)\b", re.IGNORECASE)
_FAILURE_MESSAGE_RE = re.compile(
    r"(?:\"(?:failure|error)\"\s*:|\"success\"\s*:\s*false|"
    r"\"message\"\s*:\s*\"[^\"]*(?:invalid|failed|error|unauthori[sz]ed|forbidden|not found)[^\"]*\")",
    re.IGNORECASE,
)
_MAX_LIST_ITEMS = 3
_MAX_STRING_LENGTH = 240
_SENSITIVE_TEXT_VALUE_RE = re.compile(
    r"(?P<prefix>\b(?:password|access_?token|refresh_?token|authorization|api_?key|secret|credential)\b"
    r"['\"]?\s*(?:=|:|\bis\b)\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ApiCall:
    """One statically visible API call in an Agent action."""

    name: str
    arguments: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ActionObservationPair:
    """One executed Agent action and its immediately following observation."""

    step: int
    calls: tuple[_ApiCall, ...]
    observation: str | None


class StateBuilder:
    """Build a conservative Controller view without hidden or future state."""

    def __init__(self, api_history_mode: str = "summary") -> None:
        """Select the deterministic API history detail level."""
        if api_history_mode not in {"summary", "structured"}:
            raise ValueError("api_history_mode must be 'summary' or 'structured'.")
        self._api_history_mode = api_history_mode

    def build(
        self,
        task: str,
        trigger: TriggerResult,
        recent_history: Iterable[dict[str, Any]],
        current_agent_output: str,
    ) -> ControllerInput:
        """Create ControllerInput from visible history and current Agent output."""
        if not isinstance(task, str):
            raise TypeError("task must be a string.")
        if not isinstance(current_agent_output, str):
            raise TypeError("current_agent_output must be a string.")

        full_history = _visible_history(recent_history)
        selected_task, task_message_index = _select_current_task(full_history)
        # AppWorld may prefix the real task with a complete single-shot example.
        # Only the messages after the selected task belong to the current task.
        task_history = (
            full_history[task_message_index + 1 :]
            if task_message_index is not None
            else full_history
        )
        recent_window, window_start = _truncate_history_with_start(task_history, limit=12)
        task_pairs = _action_observation_pairs(task_history)
        completed_pairs = _action_observation_pairs(task_history[:window_start])
        return ControllerInput(
            task=selected_task if selected_task is not None else task.strip(),
            trigger_reason=trigger.reason,
            recent_history=_redact_recent_history(recent_window),
            current_agent_output=current_agent_output,
            known_entities=[],
            completed_operations=_api_history(completed_pairs, self._api_history_mode),
            recent_errors=_recent_errors(task_pairs),
        )


def _visible_history(recent_history: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Retain valid model-visible role and content fields only."""
    messages: list[dict[str, str]] = []
    for record in recent_history:
        if not isinstance(record, dict):
            continue
        role = record.get("role")
        content = record.get("content")
        if not isinstance(role, str) or not role.strip():
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": copy.deepcopy(role), "content": copy.deepcopy(content)})
    return messages


def _select_current_task(history: list[dict[str, str]]) -> tuple[str | None, int | None]:
    """Return the latest explicit AppWorld task and its visible-history index."""
    selected_task: str | None = None
    selected_index: int | None = None
    for index, message in enumerate(history):
        if message["role"].lower() != "user":
            continue
        matches = re.finditer(
            r"^[ \t]*task[ \t]*:[ \t]*(\S[^\r\n]*)$",
            message["content"],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for match in matches:
            candidate = match.group(1).strip()
            if candidate:
                selected_task = candidate
                selected_index = index
    return selected_task, selected_index


def _truncate_history(history: list[dict[str, str]], limit: int = 12) -> list[dict[str, str]]:
    """Retain recent messages without leaving a leading observation orphaned."""
    window, _ = _truncate_history_with_start(history, limit)
    return window


def _truncate_history_with_start(
    history: list[dict[str, str]], limit: int = 12
) -> tuple[list[dict[str, str]], int]:
    """Return a recent history window and its start index in the source list."""
    if len(history) <= limit:
        return history, 0
    start = len(history) - limit
    if start > 0 and _is_observation(history[start]) and _is_action(history[start - 1]):
        start -= 1
    return history[start:], start


def _redact_recent_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a deep-copied, credential-redacted message window."""
    return _redact_sensitive(copy.deepcopy(history))


def _redact_sensitive(value: Any) -> Any:
    """Recursively redact values stored under sensitive keys or text assignments."""
    if isinstance(value, dict):
        return {
            copy.deepcopy(key): "<redacted>" if _is_sensitive_name(str(key)) else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive(item) for item in value)
    if isinstance(value, str):
        return _SENSITIVE_TEXT_VALUE_RE.sub(
            lambda match: f"{match.group('prefix')}{match.group('quote')}<redacted>{match.group('quote')}",
            value,
        )
    return copy.deepcopy(value)


def _is_action(message: dict[str, str]) -> bool:
    """Return whether a message can be an Agent action."""
    return message["role"].lower() in {"assistant", "agent"}


def _is_observation(message: dict[str, str]) -> bool:
    """Return whether a message can be an environment observation."""
    return not _is_action(message)


def _action_observation_pairs(history: list[dict[str, str]]) -> list[_ActionObservationPair]:
    """Pair API actions with their immediately following environment observations."""
    pairs: list[_ActionObservationPair] = []
    step = 0
    for index, action in enumerate(history):
        if not _is_action(action):
            continue
        calls = tuple(_api_calls(action["content"]))
        if not calls:
            continue
        observation = history[index + 1]["content"] if index + 1 < len(history) and _is_observation(history[index + 1]) else None
        pairs.append(_ActionObservationPair(step=step, calls=calls, observation=observation))
        step += 1
    return pairs


def _api_calls(source: str) -> list[_ApiCall]:
    """Return static `apis.*` calls and non-sensitive literal arguments."""
    calls: list[_ApiCall] = []
    for code in _code_sources(source):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            calls.extend(_regex_api_calls(code))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted_name = _dotted_name(node.func)
            if not dotted_name.startswith("apis."):
                continue
            arguments = tuple(
                (keyword.arg, value)
                for keyword in node.keywords
                if keyword.arg is not None
                and not _is_sensitive_name(keyword.arg)
                and (value := _literal_text(keyword.value)) is not None
            )
            calls.append(_ApiCall(name=dotted_name.removeprefix("apis."), arguments=arguments))
    return calls


def _regex_api_calls(code: str) -> list[_ApiCall]:
    """Conservatively recover simple literal calls after AST parse failure."""
    calls: list[_ApiCall] = []
    pattern = re.compile(r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)")
    argument_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:['\"]([^'\"]*)['\"]|(-?\d+(?:\.\d+)?))"
    )
    for app, api, raw_arguments in pattern.findall(code):
        arguments = tuple(
            (name, quoted or numeric)
            for name, quoted, numeric in argument_pattern.findall(raw_arguments)
            if not _is_sensitive_name(name)
        )
        calls.append(_ApiCall(name=f"{app}.{api}", arguments=arguments))
    return calls


def _code_sources(source: str) -> list[str]:
    """Return fenced code blocks or one parseable raw source candidate."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", source, flags=re.IGNORECASE | re.DOTALL)
    return blocks or [source]


def _dotted_name(node: ast.AST) -> str:
    """Return a dotted expression name when it is statically available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_text(node: ast.AST) -> str | None:
    """Return scalar literal text without evaluating arbitrary code."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    return str(value)


def _api_history(pairs: list[_ActionObservationPair], mode: str) -> list[str]:
    """Encode complete API history as stable JSON strings for frozen schema storage."""
    records: list[str] = []
    for pair in pairs:
        for call in pair.calls:
            record = _api_history_record(pair, call, mode)
            records.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return records


def _api_history_record(pair: _ActionObservationPair, call: _ApiCall, mode: str) -> dict[str, Any]:
    """Build one summary or structured API history record."""
    record: dict[str, Any] = {
        "api": call.name,
        "parameters": dict(call.arguments),
        "return_count": None,
        "return_type": None,
        "status": "unknown",
        "step": pair.step,
    }
    if pair.observation is None or len(pair.calls) != 1:
        return record

    errors = _errors_from_observation(pair.observation)
    if errors:
        record["status"] = "error"
        return record

    value = _output_value(pair.observation)
    if _is_query_call(call):
        if value is not None or pair.observation.strip():
            record["status"] = "success"
            record["return_type"] = _return_type(value, pair.observation)
            record["return_count"] = _return_count(value)
    elif _write_observation_confirms_success(pair.observation):
        record["status"] = "success"
        record["return_type"] = _return_type(value, pair.observation)
        record["return_count"] = _return_count(value)

    if record["status"] == "success" and _is_api_docs_call(call):
        record["catalog"] = _api_docs_catalog(call, value)
    elif record["status"] == "success" and mode == "structured":
        record["result"] = _compress_result(value if value is not None else pair.observation)
    return record


def _is_query_call(call: _ApiCall) -> bool:
    """Return whether a call name is conservatively query-like."""
    return bool(_QUERY_API_RE.search(call.name))


def _is_api_docs_call(call: _ApiCall) -> bool:
    """Return whether a call inspects the API documentation catalog."""
    return call.name.startswith("api_docs.")


def _output_value(observation: str) -> Any | None:
    """Decode one visible JSON result when the observation contains JSON."""
    candidate = re.sub(r"^\s*Output:\s*", "", observation, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"^```(?:json)?\s*|```$", "", candidate, flags=re.IGNORECASE).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _return_type(value: Any | None, observation: str) -> str:
    """Return a stable result type without interpreting its meaning."""
    if value is not None:
        return type(value).__name__
    return "text" if observation.strip() else "unknown"


def _return_count(value: Any | None) -> int | None:
    """Return a count only when the output directly exposes a collection size."""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("items", "results", "data"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return None


def _write_observation_confirms_success(observation: str) -> bool:
    """Require explicit confirmation before recording a write operation as successful."""
    return bool(_WRITE_SUCCESS_RE.search(observation) or re.search(r'"success"\s*:\s*true', observation, re.I))


def _api_docs_catalog(call: _ApiCall, value: Any | None) -> dict[str, Any] | None:
    """Preserve deterministic API documentation facts without relevance filtering."""
    if call.name == "api_docs.show_app_descriptions" and isinstance(value, list):
        return {
            "apps": [
                {"description": item.get("description"), "name": item.get("name")}
                for item in value
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
        }
    if call.name == "api_docs.show_api_descriptions" and isinstance(value, list):
        return {
            "apis": [
                {"description": item.get("description"), "name": item.get("name")}
                for item in value
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
        }
    if call.name == "api_docs.show_api_doc" and isinstance(value, dict):
        return {
            "description": value.get("description"),
            "parameters": _compress_result(value.get("parameters", [])),
            "return_structure": _compress_result(
                value.get("response_schemas", value.get("return_schema", value.get("returns")))
            ),
        }
    return None


def _compress_result(value: Any) -> Any:
    """Compress visible output with explicit list and string truncation markers."""
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LENGTH:
            return value
        return {
            "__truncated__": True,
            "original_length": len(value),
            "value": value[:_MAX_STRING_LENGTH],
        }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "count": len(value),
            "items": [_compress_result(item) for item in value[:_MAX_LIST_ITEMS]],
            "truncated": len(value) > _MAX_LIST_ITEMS,
        }
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _is_sensitive_name(str(key)) else _compress_result(item)
            for key, item in value.items()
        }
    return value


def _recent_errors(pairs: list[_ActionObservationPair]) -> list[str]:
    """Return the latest distinct errors from complete executed API history."""
    errors: list[str] = []
    for pair in pairs:
        if pair.observation is not None:
            errors.extend(_errors_from_observation(pair.observation))
    return _latest_unique(errors, limit=10)


def _errors_from_observation(observation: str) -> list[str]:
    """Extract explicit observation errors without classifying their cause."""
    errors: list[str] = []
    tracebacks = re.findall(r"Traceback \(most recent call last\):[\s\S]*?(?=\n\s*\n|\Z)", observation)
    for traceback in tracebacks:
        _append_unique(errors, traceback.strip())
    traceback_lines = {line.strip() for trace in tracebacks for line in trace.splitlines() if line.strip()}
    for line in observation.splitlines():
        stripped = line.strip()
        if not stripped or stripped in traceback_lines:
            continue
        if _EXPLICIT_ERROR_RE.search(stripped) or _HTTP_STATUS_RE.search(stripped):
            _append_unique(errors, stripped)
    if _FAILURE_MESSAGE_RE.search(observation):
        _append_unique(errors, observation.strip())
    return errors


def _is_sensitive_name(name: str) -> bool:
    """Return whether a field name may expose a credential."""
    return bool(_SENSITIVE_NAME_RE.search(name))


def _latest_unique(values: list[str], limit: int) -> list[str]:
    """Deduplicate values while retaining the newest chronological entries."""
    selected: list[str] = []
    seen: set[str] = set()
    for value in reversed(values):
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) == limit:
            break
    return list(reversed(selected))


def _append_unique(values: list[str], value: str) -> None:
    """Append a non-empty value once while preserving first-seen order."""
    if value and value not in values:
        values.append(value)
