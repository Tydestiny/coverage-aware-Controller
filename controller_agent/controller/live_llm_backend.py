"""Live LLM Controller backend using a selectable fixed prompt.

Calls an OpenAI-compatible chat API (e.g. DashScope) and returns model text
that `Controller` can parse into a validated `ControllerResponse`.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from controller_agent.controller.policy_g0 import load_g0_system_prompt, render_g0_user_prompt
from controller_agent.state.schemas import ControllerInput, ControllerResponse, response_from_mapping


DEFAULT_MODEL = "qwen3-coder-flash-2025-07-28"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "g0_system.txt"
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.I)


def call_model(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completion endpoint.

    Returns a dict with at least ``text`` and optional usage/latency fields.
    Does not send ``enable_thinking`` (incompatible with local AppWorld stacks).
    """
    model_name = model or os.environ.get("CONTROLLER_MODEL_NAME") or DEFAULT_MODEL
    endpoint = (base_url or os.environ.get("MODEL_SERVER_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = (
        api_key
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or OPENAI_API_KEY for LiveLLMBackend.")

    url = f"{endpoint}/chat/completions"
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    latency = time.perf_counter() - started

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {raw[:500]}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"LLM returned empty content: {raw[:500]}")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "text": text,
        "model_name": data.get("model") or model_name,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_seconds": latency,
        "raw": data,
    }


def strip_code_fences(text: str) -> str:
    """Remove optional markdown fences around a JSON object."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def parse_response_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from direct, fenced, or prose-prefixed model text."""
    candidate = strip_code_fences(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        if start < 0:
            raise first_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError:
            raise first_error
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON root must be an object.")
    return payload


def remove_confidence_field(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop unsupported confidence fields before parsing or audit logging."""
    normalized = dict(payload)
    normalized.pop("confidence", None)
    decision = normalized.get("decision")
    if isinstance(decision, dict):
        normalized["decision"] = {key: value for key, value in decision.items() if key != "confidence"}
    return normalized


class LiveLLMBackend:
    """Fixed-prompt LLM backend for the Coverage-Aware Controller.

    Typical use::

        controller = Controller(backend=LiveLLMBackend())
        response = controller.decide(controller_input)

    ``__call__`` returns a parsed ``ControllerResponse`` with usage metadata.
    The outer ``Controller`` retains ownership of fail-open behavior.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        system_prompt_path: str | Path | None = None,
        call_model_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.system_prompt_path = (
            DEFAULT_SYSTEM_PROMPT_PATH
            if system_prompt_path is None
            else Path(system_prompt_path)
        )
        self.call_model_fn = call_model_fn or call_model

    def __call__(self, controller_input: ControllerInput) -> ControllerResponse:
        """Match the team sketch: load prompts → call model → return text."""
        return self.decide(controller_input)

    def decide(self, controller_input: ControllerInput) -> ControllerResponse:
        """Call the model and wrap a parsed decision with usage metadata."""
        system_prompt = self._load_system_prompt()
        user_prompt = render_g0_user_prompt(controller_input)

        result = self.call_model_fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
        )
        text = strip_code_fences(str(result["text"]))
        payload = remove_confidence_field(parse_response_json(text))
        response = response_from_mapping(payload if "decision" in payload else {"decision": payload})
        return ControllerResponse(
            decision=response.decision,
            model_name=str(result.get("model_name") or self.model_name or DEFAULT_MODEL),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            latency_seconds=result.get("latency_seconds"),
            cost_cny=None,
            raw_response=json.dumps(result.get("raw"), ensure_ascii=False) if result.get("raw") is not None else None,
        )

    def _load_system_prompt(self) -> str:
        """Load the configured prompt while preserving the historical G0 default."""
        if self.system_prompt_path == DEFAULT_SYSTEM_PROMPT_PATH:
            return load_g0_system_prompt()
        return self.system_prompt_path.read_text(encoding="utf-8")


# Alias used in the team work plan naming.
QwenControllerBackend = LiveLLMBackend
