"""Detection of consecutive steps with no meaningful progress.

All tunable knobs (lookback window, repeated-action threshold, error
suffixes, etc.) live in :mod:`controller_agent.triggers.config`.  When the
trigger is instantiated without an explicit config, the package defaults
from ``config.DEFAULT_CONFIG`` are used so behaviour matches the
historically hard-coded values.
"""

from __future__ import annotations

import re

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.config import DEFAULT_CONFIG, NoProgressConfig

# ----------------------------------------------------------------------
# Error fingerprinting helpers
# ----------------------------------------------------------------------

# Default HTTP 4xx/5xx regex.  Compiled once at module import time for the
# common case; triggers constructed with a custom config use the shared
# regex from :func:`_build_fingerprinters`.
_HTTP_RE = re.compile(r"\b[45]\d\d\b", re.IGNORECASE)


# Matches individual identifier tokens (letters/digits/underscores).
_TOKEN_RE = re.compile(r"\b[a-z_][a-z0-9_]*\b", re.IGNORECASE)


def _extract_error_fingerprint(outputs: list[object]) -> str | None:
    """Return a short fingerprint from error-bearing output, or None.

    Kept as a module-level function for backwards compatibility with the
    test harness and the analysis scripts that import it directly.
    """
    # Convert ExecutionIO objects or other objects to text
    text_parts = []
    for o in outputs:
        if hasattr(o, "content"):
            # Handle ExecutionIO objects from AppWorld
            text_parts.append(str(o.content))
        else:
            text_parts.append(str(o))
    text = "".join(text_parts).strip()
    if not text:
        return None

    # Priority 0: AppWorld structured JSON errors.
    import json as _json
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            if "error" in parsed:
                error_val = parsed["error"]
                if isinstance(error_val, str) and error_val:
                    return ("exc:" + error_val[:120]).lower()
                elif isinstance(error_val, dict) and "message" in error_val:
                    return ("exc:" + error_val["message"][:120]).lower()
            if "status_code" in parsed:
                code = str(parsed["status_code"])
                if code.startswith(("4", "5")):
                    return "http:" + code
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    if "error" in item:
                        error_val = item["error"]
                        if isinstance(error_val, str) and error_val:
                            return ("exc:" + error_val[:120]).lower()
                        elif isinstance(error_val, dict) and "message" in error_val:
                            return ("exc:" + error_val["message"][:120]).lower()
                    if "status_code" in item:
                        code = str(item["status_code"])
                        if code.startswith(("4", "5")):
                            return "http:" + code
    except (_json.JSONDecodeError, TypeError, ValueError):
        pass

    # Priority 1: HTTP 4xx / 5xx.
    m = _HTTP_RE.search(text)
    if m:
        return "http:" + m.group(0)

    # Priority 2: AppWorld structured error messages.
    for pattern in (
        r'"message"\s*:\s*"([^"]{5,200})"',
        r'Error\s+(?:searching|loading|calling|parsing|finding)\s+([^:\n]{3,100})',
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            msg = m.group(1).strip()
            if msg.lower() not in ("string", "success"):
                return ("exc:" + msg[:120]).lower()

    # Priority 3: exception / error suffix words.
    for tok_m in _TOKEN_RE.finditer(text):
        word = tok_m.group(0)
        lower = word.lower()

        if lower in DEFAULT_CONFIG.no_progress.error_suffixes:
            start = tok_m.start()
            snippet = text[start:start + 60].lower()
            if '"string"' in snippet or '"success"' in snippet:
                continue
            return ("exc:" + text[start:start + 60]).lower()

        if "_" in word or any(c.isupper() for c in word[1:]):
            parts = re.split(r"(?<=[a-z])(?=[A-Z])", word)
            if len(parts) >= 2:
                last = parts[-1].lower()
                if last in DEFAULT_CONFIG.no_progress.error_suffixes:
                    start = tok_m.start()
                    snippet = text[start:start + 60].lower()
                    if '"string"' in snippet or '"success"' in snippet:
                        continue
                    return ("exc:" + text[start:start + 60]).lower()

    # Priority 4: bare "error" / "failed" / "failure" as a whole word.
    generic_re = re.compile(
        r"\b(?:" + "|".join(DEFAULT_CONFIG.no_progress.generic_error_keywords) + r")\b",
        re.IGNORECASE,
    )
    for m in generic_re.finditer(text):
        start = m.start()
        end = m.end()
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end].lower()
        if '"string"' in context or '"success"' in context:
            continue
        return "generic_error"

    return None


def _same_action(code1: str, code2: str) -> bool:
    """Return True when two code strings represent the same logical action.

    Normalises whitespace and removes function-call arguments so that
    ``fetch_data()`` and ``fetch_data(1)`` are treated as the same action.
    """
    def _strip_args(s: str) -> str:
        idx = s.find("(")
        return s[:idx] if idx != -1 else s

    n1 = re.sub(r"\s+", " ", _strip_args(code1.strip()).lower())
    n2 = re.sub(r"\s+", " ", _strip_args(code2.strip()).lower())
    return n1 == n2


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


class NoProgressTrigger:
    """Trigger when recent steps show repeated errors or repeated actions.

    Detection logic (priority order, stop on first match):

    1. **repeated_error** — the same proposed_execution has appeared in the
       last ``config.no_progress.lookback_window`` steps AND at least one of
       those steps produced the same error fingerprint as the current step.
       The Agent is looping on the same failing action.

    2. **repeated_action** — the same proposed_execution appears in the
       history >= ``config.no_progress.repeated_action_threshold`` times
       (excluding the current call when ``config.no_progress.exclude_api_docs``
       is enabled).  The Agent is re-issuing the same code without checking
       the result.
    """

    def __init__(self, config: NoProgressConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG.no_progress
        # Compile the api_docs filter once per instance.
        self._api_docs_re = (
            re.compile(self._config.api_docs_pattern, re.IGNORECASE)
            if self._config.exclude_api_docs
            else None
        )

    def detect(self, context: TriggerContext) -> TriggerResult:
        """Return a trigger result if the Agent appears stuck in recent steps."""
        window = (context.recent_history or [])[-self._config.lookback_window:]
        current_code = context.proposed_execution or ""

        # --- signal 1: repeated_error (highest priority) --------
        if self._repeated_error(window, current_code, context.last_execution_outputs):
            return TriggerResult(
                True,
                "no_progress",
                {
                    "stall_kind": "repeated_error",
                    "task_id": context.task_id,
                },
            )

        # --- signal 2: repeated_action (second priority) -------
        if self._count_action_in_window(window, current_code) >= self._config.repeated_action_threshold:
            return TriggerResult(
                True,
                "no_progress",
                {
                    "stall_kind": "repeated_action",
                    "task_id": context.task_id,
                },
            )

        return TriggerResult(False, "no_progress_not_triggered")

    # ----- helpers ------------------------------------------------

    def _is_api_docs(self, code: str) -> bool:
        if self._api_docs_re is None or not code:
            return False
        return bool(self._api_docs_re.search(code))

    def _is_same_action(self, code1: str, code2: str) -> bool:
        return _same_action(code1, code2)

    def _count_action_in_window(
        self, window: list[dict[str, object]], current_code: str
    ) -> int:
        """Count occurrences of ``current_code`` in the window.

        When ``exclude_api_docs`` is enabled, both the current code and
        the history entries are filtered to skip api_docs calls.
        """
        if not current_code:
            return 0
        if self._is_api_docs(current_code):
            return 0
        return sum(
            1
            for step in window
            if isinstance(step, dict)
            and self._is_same_action(step.get("proposed_execution") or "", current_code)
            and not self._is_api_docs(step.get("proposed_execution") or "")
        )

    def _repeated_error(
        self,
        window: list[dict[str, object]],
        current_code: str,
        current_outputs: list[object],
    ) -> bool:
        """Return True when the same action + same error appears >= 2 times."""
        current_fingerprint = _extract_error_fingerprint(current_outputs)
        if (
            current_fingerprint is None
            or len(current_fingerprint) < self._config.min_fingerprint_length
        ):
            return False

        count = 0
        for step in window:
            if not isinstance(step, dict):
                continue
            prev_code = step.get("proposed_execution") or ""
            if not _same_action(prev_code, current_code):
                continue
            prev_outputs: list[object] = step.get("last_execution_outputs") or []
            fp = _extract_error_fingerprint(prev_outputs)
            if fp is not None and fp == current_fingerprint:
                count += 1
        return count >= 1