"""Detection of consecutive steps with no meaningful progress.

All tunable knobs (lookback window, repeated-action threshold, error
suffixes, etc.) live in :mod:`controller_agent.triggers.config`.  When the
trigger is instantiated without an explicit config, the package defaults
from ``config.DEFAULT_CONFIG`` are used so behaviour matches the
historically hard-coded values.
"""

from __future__ import annotations

import ast
import json
import re

from controller_agent.state.schemas import TriggerContext, TriggerResult
from controller_agent.triggers.config import DEFAULT_CONFIG, NoProgressConfig

# ----------------------------------------------------------------------
# Error fingerprinting helpers
# ----------------------------------------------------------------------

# Require HTTP/status context so ordinary numeric results are not errors.
_HTTP_RE = re.compile(
    r"\b(?:HTTP(?:/\d(?:\.\d)?)?|status(?:_code|\s+code)?)\s*[:=]?\s*([45]\d\d)\b",
    re.IGNORECASE,
)


# Matches individual identifier tokens (letters/digits/underscores).
_TOKEN_RE = re.compile(r"\b[a-z_][a-z0-9_]*\b", re.IGNORECASE)


def _structured_error(value: object) -> str | None:
    """Read explicit failure fields without interpreting arbitrary result text."""
    if isinstance(value, list):
        return next((error for item in value if (error := _structured_error(item))), None)
    if not isinstance(value, dict):
        return None
    code = value.get("status_code", value.get("status"))
    if re.fullmatch(r"[45]\d\d", str(code)):
        return "http:" + str(code)
    error = value.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    if isinstance(error, str) and error.strip():
        return "exc:" + error.strip()[:120].lower()
    if value.get("success") is False or value.get("status") in ("error", "failed", "failure"):
        return "exc:" + str(value.get("message") or "explicit failure")[:120].lower()
    return None


def _extract_error_fingerprint(outputs: list[object]) -> str | None:
    """Return a short fingerprint from error-bearing output, or None.

    Kept as a module-level function for backwards compatibility with the
    test harness and the analysis scripts that import it directly.
    """
    # Parse each observation separately; joining JSON objects loses their structure.
    text_parts = []
    for o in outputs:
        value = getattr(o, "content", o)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                text_parts.append(value)
                continue
        error = _structured_error(value)
        if error:
            return error
    text = "\n".join(text_parts).strip()
    if not text:
        return None

    # Priority 1: HTTP 4xx / 5xx.
    m = _HTTP_RE.search(text)
    if m:
        return "http:" + m.group(1)

    # Priority 2: AppWorld structured error messages.
    for pattern in (
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

    Ignores formatting but retains arguments, literals and case.
    """
    try:
        return ast.dump(ast.parse(code1.strip())) == ast.dump(ast.parse(code2.strip()))
    except (SyntaxError, ValueError):
        return code1.strip() == code2.strip()


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


class NoProgressTrigger:
    """Trigger when recent steps show repeated errors or repeated actions.

    Detection logic (priority order, stop on first match):

    1. **repeated_error** — the same proposed_execution has appeared in the
       last ``config.no_progress.lookback_window`` steps AND at least two of
       those executed steps produced the latest error fingerprint.
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
        return 1 + sum(
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
        # The latest execution is already in window; it cannot count twice.
        return count >= 2
